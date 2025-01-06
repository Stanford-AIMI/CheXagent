import random

import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template


class ChestXray14Processor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/chestxray14"
        self.image_dir = f"{self.data_dir}/images-512"
        self.data_path = f"{self.data_dir}/Data_Entry_2017.csv"
        self.split_path = f"{self.data_dir}/split_by_image.csv"
        self.ann_data = pd.read_csv(self.data_path)
        self.split_data = pd.read_csv(self.split_path)
        self.split_data = self.split_data.set_index("image")
        self.split_data.loc[self.split_data.split == "valid", "split"] = "val"
        self.split_data.rename(columns={"pleural_thickening": "pleural thickening"}, inplace=True)
        self.negative_ratio = 10000 / (self.split_data.iloc[:, 3:].sum(1) == 0).sum()

    def create_image_classification(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image Classification] [ChestXray14]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for row_idx, row in self.split_data.iterrows():
            unique_id = f'[Image Classification] [ChestXray14] [{row_idx}]'
            study_id = f'[chestxray14] [{row_idx.replace(".png", "")}]'
            data_source = "ChestXray14"
            task_name = "Image Classification"
            image_path = f'{self.image_dir}/{row_idx.replace("png", "jpg")}'
            text = ""
            target_text = ""

            options = row.iloc[3:].astype(int).to_dict()

            if len([k for k, v in options.items() if v >= 1]) == 0:
                options.update({"no findings": 1})
                if row.split == "train" and random.random() > self.negative_ratio:
                    continue
            else:
                options.update({"no findings": 0})

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
            dataset[row.split].append(sample)
        return dataset


if __name__ == '__main__':
    processor = ChestXray14Processor()
    processor.create_image_classification()
