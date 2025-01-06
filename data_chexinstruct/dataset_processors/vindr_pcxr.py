import os
import random
from collections import defaultdict

import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut
from torchvision import transforms

from .base_processor import BaseProcessor
from .templates import create_template


def cvt(coord):
    coord = min(coord, 999)
    coord = round(coord / 10)
    return str(coord)


def get_iou(box1, box2):
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])

    inter = 0 if in_h < 0 or in_w < 0 else in_h * in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter

    iou = inter / union
    return iou


def merge_boxes(boxes, threshold=0.1):
    merged_boxes = []

    while boxes:
        # Pop the first box to compare with the others
        base_class, base_box = boxes.pop(0)
        to_merge = [base_box]

        # Check all other boxes to find the ones that overlap and are the same class
        remaining_boxes = []
        for box_class, box in boxes:
            if box_class == base_class and get_iou(base_box, box) > threshold:
                to_merge.append(box)
            else:
                remaining_boxes.append((box_class, box))

        # Merge all the overlapping boxes by finding the outermost coordinates
        x1, y1, x2, y2 = zip(*to_merge)
        merged_box = (min(x1), min(y1), max(x2), max(y2))
        merged_boxes.append((base_class, merged_box))

        # Update the list of boxes to check
        boxes = remaining_boxes

    return merged_boxes


class VinDRPCXRProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/vindr-pcxr/"
        train_path, test_path = f'{self.data_dir}/annotations_train.csv', f'{self.data_dir}/annotations_test.csv'
        train_meta_path = f'{self.data_dir}/train_meta.csv'
        test_meta_path = f'{self.data_dir}/test_meta.csv'
        self.train_data, self.test_data = pd.read_csv(train_path), pd.read_csv(test_path)
        self.train_meta_data = pd.read_csv(train_meta_path, index_col=0)
        self.test_meta_data = pd.read_csv(test_meta_path, index_col=0)

    def create_abnormality_detection(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Abnormality Detection] [VinDr-PCXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data, meta_data in zip(
                ["train", "test"], [self.train_data, self.test_data], [self.train_meta_data, self.test_meta_data]):
            split_data.dropna(subset=["class_name"])
            for group_idx, group in split_data.groupby("image_id"):
                unique_id = f"[Abnormality Detection] [VinDr-PCXR] [{group_idx}]"
                study_id = f'[vindr-pcxr] [{group_idx}]'
                data_source = "VinDr-PCXR"
                task_name = "Abnormality Detection"
                image_path = f'{self.data_dir}/{split}_png/{group_idx}.png'
                w, h = meta_data.loc[group_idx].width, meta_data.loc[group_idx].height
                ratio = min(w, h) / 512
                regions = [(
                    row.class_name,
                    [max(row.x_min, 0) / ratio, max(row.y_min, 0) / ratio,
                     min(row.x_max, w) / ratio, min(row.y_max, h) / ratio])
                    for row_idx, row in group.iterrows() if row.class_name != "No finding"]
                regions = merge_boxes(regions)

                if len(regions) == 0:
                    text = "abnormalities"
                    target_text = "No abnormalities detected."
                else:
                    text = "abnormalities"
                    image = Image.open(image_path)
                    quantized_boxes = [
                        (r[0],
                         [int((r[1][0] / image.width) * qbins), int((r[1][1] / image.height) * qbins),
                          int((r[1][2] / image.width) * qbins), int((r[1][3] / image.height) * qbins)])
                        for r in regions
                    ]
                    quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[1][0], x[1][1]))
                    if len(quantized_boxes) > 5:
                        continue
                    target_text = "".join(
                        [
                            f"<|ref|>{b[0]}<|/ref|><|box|>({cvt(b[1][0])},{cvt(b[1][1])}),({cvt(b[1][2])},{cvt(b[1][3])})<|/box|>"
                            for b in quantized_boxes
                        ]
                    )

                qa_pair = form_qa_func(self.instruct)(text, target_text)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'data_source': data_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'region': regions,
                    'text': text,
                    'target_text': target_text,
                    'qa_pair': qa_pair
                }
                dataset[split].append(sample)
        return dataset

    def create_abnormality_grounding(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Abnormality Grounding] [VinDr-PCXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data, meta_data in zip(
                ["train", "test"], [self.train_data, self.test_data], [self.train_meta_data, self.test_meta_data]):
            split_data.dropna(subset=["class_name"])

            all_classes = set(split_data.class_name.unique().tolist())
            class2neg = defaultdict(list)
            for group_idx, group in split_data.groupby("image_id"):
                for class_name in all_classes - set(group.class_name.tolist()):
                    class2neg[class_name].append(group_idx)

            for group_idx, group in split_data.groupby("image_id"):
                w, h = meta_data.loc[group_idx].width, meta_data.loc[group_idx].height
                ratio = min(w, h) / 512
                regions = [(
                    row.class_name,
                    [max(row.x_min, 0) / ratio, max(row.y_min, 0) / ratio,
                     min(row.x_max, w) / ratio, min(row.y_max, h) / ratio])
                    for row_idx, row in group.iterrows() if row.class_name != "No finding"]
                regions = merge_boxes(regions)
                for class_name in set([region[0] for region in regions]):
                    # positive
                    unique_id = f"[Abnormality Grounding] [VinDr-PCXR] [{group_idx}]"
                    study_id = f'[VinDr-PCXR] [{group_idx}]'
                    data_source = "VinDr-PCXR"
                    task_name = "Abnormality Grounding"
                    image_path = f'{self.data_dir}/{split}_png/{group_idx}.png'
                    text = class_name
                    image = Image.open(image_path)
                    selected_regions = [r[1] for r in regions if r[0] == class_name]
                    quantized_boxes = [
                        [int((r[0] / image.width) * qbins), int((r[1] / image.height) * qbins),
                         int((r[2] / image.width) * qbins), int((r[3] / image.height) * qbins)]
                        for r in selected_regions
                    ]
                    quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
                    if len(quantized_boxes) > 2:
                        continue
                    target_text = "".join(
                        [
                            f"<|ref|>{class_name}<|/ref|><|box|>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})<|/box|>"
                            for b in quantized_boxes]
                    )

                    qa_pair = form_qa_func(self.instruct)(text, target_text)
                    sample = {
                        'unique_id': unique_id,
                        'study_id': study_id,
                        'data_source': data_source,
                        'task_name': task_name,
                        'image_path': image_path,
                        'region': selected_regions,
                        'text': text,
                        'target_text': target_text,
                        'qa_pair': qa_pair
                    }
                    dataset[split].append(sample)

                    # negative
                    unique_id = f"[Abnormality Grounding] [VinDr-PCXR] [{group_idx} - negative]"
                    study_id = f'[VinDr-PCXR] [{group_idx}]'
                    data_source = "VinDr-PCXR"
                    task_name = "Abnormality Grounding"
                    image_path = f'{self.data_dir}/{split}_png/{random.choice(class2neg[class_name])}.png'
                    text = class_name
                    target_text = f"No {class_name.lower()} detected."
                    selected_regions = []

                    qa_pair = form_qa_func(self.instruct)(text, target_text)
                    sample = {
                        'unique_id': unique_id,
                        'study_id': study_id,
                        'data_source': data_source,
                        'task_name': task_name,
                        'image_path': image_path,
                        'region': selected_regions,
                        'text': text,
                        'target_text': target_text,
                        'qa_pair': qa_pair
                    }
                    dataset[split].append(sample)
        return dataset

    def create_grounded_diagnosis(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Grounded Diagnosis] [VinDr-PCXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data, meta_data in zip(
                ["train", "test"], [self.train_data, self.test_data], [self.train_meta_data, self.test_meta_data]):
            split_data.dropna(subset=["class_name"])

            all_classes = set(split_data.class_name.unique().tolist())
            class2neg = defaultdict(list)
            for group_idx, group in split_data.groupby("image_id"):
                for class_name in all_classes - set(group.class_name.tolist()):
                    class2neg[class_name].append(group_idx)

            for group_idx, group in split_data.groupby("image_id"):
                w, h = meta_data.loc[group_idx].width, meta_data.loc[group_idx].height
                ratio = min(w, h) / 512
                regions = [(
                    row.class_name,
                    [max(row.x_min, 0) / ratio, max(row.y_min, 0) / ratio,
                     min(row.x_max, w) / ratio, min(row.y_max, h) / ratio])
                    for row_idx, row in group.iterrows() if row.class_name != "No finding"]
                regions = merge_boxes(regions)
                for class_name in set([region[0] for region in regions]):
                    # positive
                    unique_id = f"[Grounded Diagnosis] [VinDr-PCXR] [{group_idx}]"
                    study_id = f'[vindr-pcxr] [{group_idx}]'
                    data_source = "VinDr-PCXR"
                    task_name = "Grounded Diagnosis"
                    image_path = f'{self.data_dir}/{split}_png/{group_idx}.png'
                    image = Image.open(image_path)
                    selected_regions = [r[1] for r in regions if r[0] == class_name]
                    quantized_boxes = [
                        [int((r[0] / image.width) * qbins), int((r[1] / image.height) * qbins),
                         int((r[2] / image.width) * qbins), int((r[3] / image.height) * qbins)]
                        for r in selected_regions
                    ]
                    quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
                    if len(quantized_boxes) > 2:
                        continue
                    target_text = class_name

                    for b in quantized_boxes:
                        text = f"<|box|>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})<|/box|>"
                        qa_pair = form_qa_func(self.instruct)(text, target_text)
                        sample = {
                            'unique_id': unique_id,
                            'study_id': study_id,
                            'data_source': data_source,
                            'task_name': task_name,
                            'image_path': image_path,
                            'region': selected_regions,
                            'text': text,
                            'target_text': target_text,
                            'qa_pair': qa_pair
                        }
                        dataset[split].append(sample)
        return dataset


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


def convert_dicom_to_png(size=1024):
    data_dir = "data/vindr-pcxr/"
    in_train_dir, in_test_dir = f"{data_dir}/train", f"{data_dir}/test"
    out_train_dir, out_test_dir = f"{data_dir}/train_png_{size}", f"{data_dir}/test_png_{size}"
    os.makedirs(out_train_dir, exist_ok=True)
    os.makedirs(out_test_dir, exist_ok=True)
    transform = transforms.Compose([transforms.Resize(size)])
    train_meta, test_meta = {}, {}

    # For test
    for file in os.listdir(in_test_dir):
        if "dicom" not in file:
            continue
        in_path, out_path = os.path.join(in_test_dir, file), os.path.join(
            out_test_dir, file.replace("dicom", "png")
        )
        image = read_xray(in_path).convert('RGB')
        test_meta[file.split(".")[0]] = {"width": image.width, "height": image.height}
        transform(image).save(out_path, quality=100, subsampling=0)
    test_meta = pd.DataFrame(test_meta).T
    test_meta.to_csv(f"{data_dir}/test_meta.csv")

    # For training
    for file in os.listdir(in_train_dir):
        if "dicom" not in file:
            continue
        in_path, out_path = os.path.join(in_train_dir, file), os.path.join(
            out_train_dir, file.replace("dicom", "png")
        )
        image = read_xray(in_path).convert('RGB')
        train_meta[file.split(".")[0]] = {"width": image.width, "height": image.height}
        transform(image).save(out_path, quality=100, subsampling=0)
    train_meta = pd.DataFrame(train_meta).T
    train_meta.to_csv(f"{data_dir}/train_meta.csv")


if __name__ == '__main__':
    # convert_dicom_to_png()
    processor = VinDRPCXRProcessor()
    processor.create_abnormality_detection()
    processor.create_abnormality_grounding()
    processor.create_grounded_diagnosis()
