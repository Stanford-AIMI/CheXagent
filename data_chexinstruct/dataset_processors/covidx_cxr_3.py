import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template


class COVIDXCXR3Processor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/covidx-cxr-3"
        self.train_path = f"{self.data_dir}/train.txt"
        self.test_path = f"{self.data_dir}/test.txt"
        self.train_data = pd.read_csv(self.train_path, delimiter=" ", header=None)
        self.test_data = pd.read_csv(self.test_path, delimiter=" ", header=None)

    def create_image_classification(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image Classification] [COVIDX-CXR-3]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "test"], [self.train_data, self.test_data]):
            for row_idx, row in split_data.iterrows():
                unique_id = f'[Image Classification] [COVIDX-CXR-3] [{row_idx}]'
                study_id = f'[{row[3]}] [{row[1].split(".")[0]}]'
                data_source = "COVIDX-CXR-3"
                task_name = "Image Classification"
                image_path = f'{self.data_dir}/{split}/{row[1]}'
                text = ""
                target_text = ""
                options = {"Covid (positive)": row[2] == "positive",
                           "Covid (negative)": row[2] != "positive"}
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


if __name__ == '__main__':
    processor = COVIDXCXR3Processor()
    processor.create_image_classification()
