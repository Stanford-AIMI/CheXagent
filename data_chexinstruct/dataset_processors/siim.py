import glob
import os

import cv2
import numpy as np
import pandas as pd
import pydicom
import torch
import tqdm
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torchvision.ops import masks_to_boxes

from .base_processor import BaseProcessor
from .templates import create_template


def cvt(coord):
    coord = min(coord, 999)
    coord = round(coord / 10)
    return str(coord)


class SIIMProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/siim"
        self.image_dir = f'{self.data_dir}/SIIM-ACR-Pneumothorax-512/train'
        train_path = f'{self.data_dir}/SIIM-ACR-Pneumothorax-orginial/train.csv'
        val_path = f'{self.data_dir}/SIIM-ACR-Pneumothorax-orginial/val.csv'
        test_path = f'{self.data_dir}/SIIM-ACR-Pneumothorax-orginial/test.csv'
        self.train_data = pd.read_csv(train_path)
        self.val_data = pd.read_csv(val_path)
        self.test_data = pd.read_csv(test_path)

    def create_pneumothorax_segmentation(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Pneumothorax Segmentation] [SIIM]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            for group_idx, group in split_data.groupby("ImageId"):
                unique_id = f"[SIIM] [Pneumothorax Segmentation] [{group_idx}]"
                study_id = f'[siim] [{group_idx}]'
                image_source = "SIIM"
                task_name = "Pneumothorax Segmentation"
                image_path = f'{self.image_dir}/{group_idx}.png'

                if "-1" in group[" EncodedPixels"].iloc[0]:
                    regions = []
                    text = "pneumothorax"
                    target_text = "No pneumothorax detected."
                else:
                    regions = [
                        ("Pneumothorax", mask2polygen(rle2mask(rle, 1024, 1024, 1, 512, 512)))
                        for rle in group[" EncodedPixels"].tolist()
                    ]
                    boxes = masks_to_boxes(torch.Tensor(np.array([
                        rle2mask(rle, 1024, 1024, 1, 512, 512)
                        for rle in group[" EncodedPixels"].tolist()
                    ])))
                    quantized_boxes = (boxes / 512 * qbins).int().tolist()
                    quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
                    text = "pneumothorax"
                    target_text = "".join([
                        f"<|ref|>pneumothorax<|/ref|><|box|>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})<|/box|>"
                        for b in quantized_boxes
                    ])

                qa_pair = form_qa_func(self.instruct)(text, target_text)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'image_source': image_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'region': regions,
                    'text': text,
                    'target_text': target_text,
                    'qa_pair': qa_pair
                }
                dataset[split].append(sample)

        return dataset


def mask2polygen(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    return segmentation[0]


def rle2mask(rle, height=1024, width=1024, fill_value=1, out_height=1024, out_width=1024):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start + index
        end = start + length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    if height != out_height or width != out_width:
        component = cv2.resize(component, (out_height, out_width)).astype(bool)
    return component.astype(np.uint8)


def mask2rle(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0] + 1
    end = np.where(component[:-1] > component[1:])[0] + 1
    length = end - start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i] - end[i - 1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle


def preprocess_pneumothorax_data(test_fac=0.15):
    # From Gloria: https://github.com/marshuang80/gloria/blob/main/gloria/datasets/preprocess_datasets.py
    try:
        df = pd.read_csv("data/siim/SIIM-ACR-Pneumothorax-orginial/train-rle.csv")
    except:
        raise Exception(
            "Please make sure the the SIIM Pneumothorax dataset is \
            stored at {PNEUMOTHORAX_DATA_DIR}"
        )

    # get image paths
    img_paths = {}
    for subdir, dirs, files in tqdm.tqdm(os.walk("data/siim/SIIM-ACR-Pneumothorax-orginial/dicom-images-train")):
        for f in files:
            if "dcm" in f:
                # remove dcm
                file_id = f[:-4]
                img_paths[file_id] = os.path.join(subdir, f)

    # no encoded pixels mean healthy
    df["Label"] = df.apply(
        lambda x: 0.0 if x[" EncodedPixels"] == " -1" else 1.0, axis=1
    )
    df["Path"] = df["ImageId"].apply(lambda x: img_paths[x])

    # split data
    train_df, test_val_df = train_test_split(df, test_size=test_fac * 2, random_state=0)
    test_df, valid_df = train_test_split(test_val_df, test_size=0.5, random_state=0)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Label"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Label"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Label"].value_counts())

    train_df.to_csv("data/siim/SIIM-ACR-Pneumothorax-orginial/train.csv")
    valid_df.to_csv("data/siim/SIIM-ACR-Pneumothorax-orginial/val.csv")
    test_df.to_csv("data/siim/SIIM-ACR-Pneumothorax-orginial/test.csv")


def read_xray(path, voi_lut=True, fix_monochrome=True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    print(path)
    dicom = pydicom.read_file(path, force=True)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    im = Image.fromarray(data)
    return im


def convert_dicom_to_png():
    data_dir = "data/siim/SIIM-ACR-Pneumothorax-orginial/"
    in_train_dir = f"{data_dir}/dicom-images-train"
    in_test_dir = f"{data_dir}/dicom-images-test"
    out_train_dir, out_test_dir = f"{data_dir}/dicom-images-train_png", f"{data_dir}/dicom-images-test_png"
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)
    transform = transforms.Compose([transforms.Resize(512)])
    train_meta, test_meta = {}, {}
    # For test set
    for file in glob.glob(f'{in_test_dir}/*/*/*'):
        if "dicom" not in file:
            continue
        in_path, out_path = file, file.replace("dicom", "png")
        image = read_xray(in_path).convert('RGB')
        test_meta[file.split("/")[-1][:-4]] = {"width": image.width, "height": image.height}
        # transform(image).save(out_path, quality=100, subsampling=0)
    test_meta = pd.DataFrame(test_meta).T
    test_meta.to_csv(f"{data_dir}/test_meta.csv")

    # For training set
    for file in glob.glob(f'{in_train_dir}/*/*/*'):
        if "dicom" not in file:
            continue
        in_path, out_path = file, file.replace("dicom", "png")
        image = read_xray(in_path).convert('RGB')
        train_meta[file.split("/")[-1][:-4]] = {"width": image.width, "height": image.height}
        print({"width": image.width, "height": image.height})
        # transform(image).save(out_path, quality=100, subsampling=0)
    train_meta = pd.DataFrame(train_meta).T
    train_meta.to_csv(f"{data_dir}/train_meta.csv")


if __name__ == '__main__':
    # convert_dicom_to_png()
    # preprocess_pneumothorax_data()
    processor = SIIMProcessor()
    processor.create_pneumothorax_segmentation()
