import os
from pathlib import Path

import cv2
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split

from .base_processor import BaseProcessor
from .templates import create_template


class RSNAProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/rsna/"
        self.train_path = f"{self.data_dir}/train.csv"
        self.train_01_path = f"{self.data_dir}/train_01.csv"
        self.train_001_path = f"{self.data_dir}/train_001.csv"
        self.val_path = f"{self.data_dir}/val.csv"
        self.test_path = f"{self.data_dir}/test.csv"
        self.train_data, self.val_data = pd.read_csv(self.train_path), pd.read_csv(self.val_path)
        self.test_data = pd.read_csv(self.test_path)
        self.train_01_data = pd.read_csv(self.train_01_path)
        self.train_001_data = pd.read_csv(self.train_001_path)

    def create_image_classification(self):
        dataset = {"train": [], "train_01": [], "train_001": [], "val": [], "test": [],
                   "dataset_name": "[Image Classification] [RSNA]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(
                ["train", "train_01", "train_001", "val", "test"],
                [self.train_data, self.train_01_data, self.train_001_data, self.val_data, self.test_data]
        ):
            split_data.Path = split_data.Path.map(lambda x: x.replace("images", "images_png").replace("dcm", "png"))
            for row_idx, row in split_data.iterrows():
                unique_id = f'[Image Classification] [RSNA] [{row_idx}]'
                study_id = f'[rsna] [{row.Path.split("/")[-1].split(".")[0]}]'
                data_source = "RSNA"
                task_name = "Image Classification"
                image_path = row.Path
                text = ""
                target_text = ""
                options = {"Pneumonia": row.Target}
                qa_pair = form_qa_func(self.instruct)(options)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'data_source': data_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    "text": text,
                    'target_text': target_text,
                    "options": options,
                    "qa_pair": qa_pair
                }
                dataset[split].append(sample)
        return dataset


def read_dicom_save_png(img_path):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array

    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    new_img_path = str(img_path).replace(".dcm", ".png").replace("stage_2_train_images", "stage_2_train_images_png")
    cv2.imwrite(new_img_path, x)
    return new_img_path


def prepro_rsna_pneumonia(test_fac=0.15):
    PNEUMONIA_DATA_DIR = Path("data/rsna/")
    PNEUMONIA_ORIGINAL_TRAIN_CSV = PNEUMONIA_DATA_DIR / "stage_2_train_labels.csv"
    PNEUMONIA_IMG_DIR = PNEUMONIA_DATA_DIR / "stage_2_train_images"

    os.makedirs(Path(str(PNEUMONIA_IMG_DIR).replace("stage_2_train_images", "stage_2_train_images_png")), exist_ok=True)

    df = pd.read_csv(PNEUMONIA_ORIGINAL_TRAIN_CSV)

    # create bounding boxes
    def create_bbox(row):
        if row["Target"] == 0:
            return 0
        else:
            x1 = row["x"]
            y1 = row["y"]
            x2 = x1 + row["width"]
            y2 = y1 + row["height"]
            return [x1, y1, x2, y2]

    df["bbox"] = df.apply(lambda x: create_bbox(x), axis=1)

    # aggregate multiple boxes
    df = df[["patientId", "bbox"]]
    df = df.groupby("patientId").agg(list)
    df = df.reset_index()
    df["bbox"] = df["bbox"].apply(lambda x: None if x == [0] else x)

    # create labels
    df["Target"] = df["bbox"].apply(lambda x: 0 if x == None else 1)

    # no encoded pixels mean healthy
    df["Path"] = df["patientId"].apply(lambda x: PNEUMONIA_IMG_DIR / (x + ".dcm"))

    # split data
    train_df, test_val_df = train_test_split(df, test_size=test_fac * 2, random_state=0)
    test_df, valid_df = train_test_split(test_val_df, test_size=0.5, random_state=0)

    print(f"Number of train samples: {len(train_df)}")
    print(train_df["Target"].value_counts())
    print(f"Number of valid samples: {len(valid_df)}")
    print(valid_df["Target"].value_counts())
    print(f"Number of test samples: {len(test_df)}")
    print(test_df["Target"].value_counts())

    train_df_001 = train_df.sample(frac=0.01)
    train_df_01 = train_df.sample(frac=0.1)

    data = {
        "train": [],
        "train_001": [],
        "train_01": [],
        "val": [],
        "test": [],
    }
    for split, split_data in zip(["train", "train_001", "train_01", "val", "test"],
                                 [train_df, train_df_001, train_df_01, valid_df, test_df]):
        for sample_idx, sample in split_data.iterrows():
            img_path = read_dicom_save_png(sample["Path"])
            texts = ["None"]
            label = [sample["Target"]]
            if len(texts) > 0:
                data[split].append({
                    "img_path": img_path,
                    "texts": texts,
                    "pnsa_pneumonia": label
                })
    train_df_001.to_csv("data/rsna/train_001.csv")
    train_df_01.to_csv("data/rsna/train_01.csv")
    train_df.to_csv("data/rsna/train.csv")
    valid_df.to_csv("data/rsna/val.csv")
    test_df.to_csv("data/rsna/test.csv")


if __name__ == '__main__':
    # prepro_rsna_pneumonia()
    processor = RSNAProcessor()
    processor.create_image_classification()
