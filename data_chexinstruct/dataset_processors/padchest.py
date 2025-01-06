import os
import random
import re

import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template


def remove_variable_dots(text):
    text = re.sub(r'^\s*\.\s*(?:\.\s*)+', '', text)
    text = re.sub(r'\s*\.\s*(?:\.\s*)+', '. ', text)
    return text


class PadChestProcessor(BaseProcessor):
    @BaseProcessor.timeit
    def __init__(self):
        super().__init__()
        self.data_dir = "data/padchest/"
        self.image_dir = f'{self.data_dir}/images'
        self.ann_path = f"{self.data_dir}/PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
        self.ann_data = pd.read_csv(self.ann_path)
        self.ann_data = self.ann_data.dropna(subset=["Labels"])
        self.ann_data = self.ann_data[self.ann_data.ImageID.map(lambda x: os.path.exists(f"{self.image_dir}/{x}"))]
        ranked_views = ["POSTEROANTERIOR", "PA", "ANTEROPOSTERIOR", "AP", '', "LATERAL", "LL", "OBLICUA", "RL", "LLD",
                        "GENERICA"]
        self.ann_data.ViewPosition_DICOM = self.ann_data.ViewPosition_DICOM.fillna("")
        self.ann_data["ViewPositionRank"] = self.ann_data.ViewPosition_DICOM.map(lambda x: ranked_views.index(x))
        self.ann_data = self.ann_data.sort_values("ViewPositionRank")
        self.ann_data = self.ann_data.reset_index(drop=True)

        self.reports = [(image_line, report_line) for image_line, report_line
                        in zip(open(f'{self.data_dir}/image.tok'), open(f'{self.data_dir}/report.en.tok'))]
        self.image2report = {}
        for images, report in self.reports:
            for image in images.strip().split(","):
                self.image2report[image] = report.strip()

    @BaseProcessor.timeit
    def create_image_classification(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image Classification] [PadChest]"}
        label_set = set([label.strip() for labels in self.ann_data.Labels for label in eval(labels)])
        form_qa_func = create_template(dataset["dataset_name"])
        for group_idx, group in self.ann_data.groupby("StudyID"):
            unique_id = f'[Image Classification] [PadChest] [{group_idx}]'
            study_id = f'[padchest] [{group_idx}]'
            data_source = "PadChest"
            task_name = "Image Classification"
            image_path = [f'{self.image_dir}/{image_id}' for image_id in group.ImageID.tolist()]
            text = ""
            target_text = ""
            options = {label.strip(): 1 for label in eval(group.iloc[0].Labels)}
            options.update(
                {label: 0 for label in random.sample([label for label in label_set if label not in options], 10)}
            )
            options = {k: v for k, v in options.items() if k}
            if len(options) > 26:
                continue
            qa_pair = form_qa_func(self.instruct)(options)

            if isinstance(qa_pair, list):
                if any([len(qa_pair[i]["a"]) == 0 for i in range(len(qa_pair))]):
                    continue
            else:
                if len(qa_pair["a"]) == 0:
                    continue

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
            dataset["train"].append(sample)
        return dataset

    @BaseProcessor.timeit
    def create_impression_generation(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Impression Generation] [PadChest]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for group_idx, group in self.ann_data.groupby("StudyID"):
            unique_id = f'[Report Generation] [PadChest] [{group_idx}]'
            study_id = f'[padchest] [{group_idx}]'
            data_source = "PadChest"
            task_name = "Report Generation"
            image_path = [f'{self.image_dir}/{image_id}' for image_id in group.ImageID.tolist()]
            text = ""
            if image_path[0].split("/")[-1] not in self.image2report:
                continue
            target_text = self.image2report[image_path[0].split("/")[-1]]
            qa_pair = form_qa_func(self.instruct)(target_text)
            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'data_source': data_source,
                'task_name': task_name,
                'image_path': image_path,
                "text": text,
                'target_text': target_text,
                "qa_pair": qa_pair,
            }
            dataset["train"].append(sample)
        return dataset


if __name__ == '__main__':
    processor = PadChestProcessor()
    processor.create_image_classification()
    processor.create_impression_generation()
