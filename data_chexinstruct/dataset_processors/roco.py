import os
import random

import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template


class ROCOProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/roco"
        self.image_dir = "data/roco/{}/radiology/images"
        self.data = {}
        self.cxr_lists = {}
        for split in ["train", "validation", "test"]:
            cxr_list = set(open(f"{self.data_dir}/{split}/radiology/list_chest_xray.txt").read().strip().split("\n"))
            files = [file.replace(".jpg", "") for file in os.listdir(self.image_dir.format(split))]
            data = pd.read_csv(f"{self.data_dir}/{split}/radiology/captions.txt", delimiter="\t", header=None)
            data = data[data[0].map(lambda x: x in files)]
            data[0] = data[0].map(lambda x: f'{self.image_dir.format(split)}/{x}.jpg')
            data = data.reset_index(drop=True)
            data = data[data[0].map(lambda x: x.split("/")[-1] in cxr_list)]
            data = data.reset_index(drop=True)
            self.data[split.replace("validation", "val")] = data

    def create_caption_generation(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Caption Generation] [ROCO]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split in ["train", "val", "test"]:
            for row_idx, row in self.data[split].iterrows():
                unique_id = f'[Caption Generation] [ROCO] [{row_idx}]'
                study_id = f'[roco] [{row[0].split(".")[0].split("_")[-1]}]'
                data_source = "ROCO"
                task_name = "Caption Generation"
                image_path = row[0]
                text = ""
                target_text = row[1]
                qa_pair = form_qa_func(self.instruct)(target_text)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'data_source': data_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    "text": text,
                    'target_text': target_text,
                    "qa_pair": qa_pair
                }
                dataset[split].append(sample)
        return dataset

    def create_image_text_matching(self):
        def randint_with_excluded_value(a, b, e):
            while True:
                randint = random.randint(a, b)
                if randint != e:
                    break
            return randint

        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image-Text Matching] [ROCO]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split in ["train", "val", "test"]:
            a, b = 0, len(self.data[split]) - 1

            self.data[split]["n_text"] = self.data[split].index.map(
                lambda e: self.data[split][1].loc[randint_with_excluded_value(a, b, e)]
            )
            for row_id, row in self.data[split].iterrows():
                unique_id = f'[Image-Text Matching] [ROCO] [{row_id}]'
                study_id = f'[roco] [{row[0].split(".")[0].split("_")[-1]}]'
                data_source = "ROCO"
                task_name = "Image-Text Matching"
                image_path = row[0]
                # Positive
                text = row[1]
                target_text = ""
                options = {"Matched": True, "Not matched": False}
                qa_pair = form_qa_func(self.instruct)(text, options)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'data_source': data_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'text': text,
                    'options': options,
                    'target_text': target_text,
                    "qa_pair": qa_pair,
                }
                dataset[split].append(sample)
                # Negative
                text = row.n_text
                target_text = ""
                options = {"Matched": False, "Not matched": True}
                qa_pair = form_qa_func(self.instruct)(text, options)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'data_source': data_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'text': text,
                    'options': options,
                    'target_text': target_text,
                    "qa_pair": qa_pair
                }
                dataset[split].append(sample)
        return dataset

    def create_image_text_selection(self):
        def randint_with_excluded_value(a, b, e):
            while True:
                randint = random.randint(a, b)
                if randint != e:
                    break
            return randint

        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image-Text Selection] [ROCO]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split in ["train", "val", "test"]:
            a, b = 0, len(self.data[split]) - 1

            self.data[split]["n_text_1"] = self.data[split].index.map(
                lambda e: self.data[split][1].loc[randint_with_excluded_value(a, b, e)]
            )
            self.data[split]["n_text_2"] = self.data[split].index.map(
                lambda e: self.data[split][1].loc[randint_with_excluded_value(a, b, e)]
            )
            self.data[split]["n_text_3"] = self.data[split].index.map(
                lambda e: self.data[split][1].loc[randint_with_excluded_value(a, b, e)]
            )
            for row_id, row in self.data[split].iterrows():
                unique_id = f'[Image-Text Selection] [ROCO] [{row_id}]'
                study_id = f'[roco] [{row[0].split(".")[0].split("_")[-1]}]'
                data_source = "ROCO"
                task_name = "Image-Text Selection"
                image_path = row[0]
                text = ""
                target_text = ""
                # options = {row[1]: True, row.n_text_1: False, row.n_text_2: False, row.n_text_3: False}
                options = [(row[1], True), (row.n_text_1, False)]
                ordered_options = [(k, v) for k, v in options]
                random.shuffle(ordered_options)
                qa_pair = form_qa_func(self.instruct)(ordered_options)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'data_source': data_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'text': text,
                    'options': options,
                    'target_text': target_text,
                    "qa_pair": qa_pair,
                }
                dataset[split].append(sample)
        return dataset


if __name__ == '__main__':
    processor = ROCOProcessor()
    processor.create_caption_generation()
    processor.create_image_text_matching()
    processor.create_image_text_selection()
