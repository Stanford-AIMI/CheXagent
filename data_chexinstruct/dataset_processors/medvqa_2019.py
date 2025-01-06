import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template


class MedVQA2019Processor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/medvqa-2019"
        data = {}
        self.cxr_lists = {}
        for split in ["train", "val", "test"]:
            categories = ["Abnormality"]
            csvs = [
                pd.read_csv(f'{self.data_dir}/{split}/QA/{cat}.csv', delimiter="|", header=None) for cat in categories
            ]
            data[split] = pd.concat(csvs, axis=0)
            self.cxr_lists[split] = open(f'{self.data_dir}/{split}/list_chest_xray.txt').read().strip().split("\n")
        self.train_data, self.val_data, self.test_data = data["train"], data["val"], data["test"]

    def create_open_ended_vqa(self):

        dataset = {"dataset_name": "[Open-Ended VQA] [MedVQA-2019]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            samples = []
            for row_idx, row in split_data.iterrows():
                if row[0] + ".jpg" not in self.cxr_lists[split]:
                    continue
                unique_id = f'[Open-Ended VQA] [MedVQA-2019] [{row_idx}]'
                study_id = f'[medvqa2019] [{row[0]}]'
                data_source = "MedVQA-2019"
                task_name = "Open-Ended VQA"
                image_path = f'{self.data_dir}/{split}/images/{row[0]}.jpg'
                text = row[1]
                target_text = row[2].capitalize()
                qa_pair = form_qa_func(self.instruct)(text, target_text)
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
                samples.append(sample)
            dataset[split] = samples
        return dataset

    def create_close_ended_vqa(self):
        dataset = {"dataset_name": "[Close-Ended VQA] [MedVQA-2019]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            samples = []
            for row_idx, row in split_data.iterrows():
                if row[0] + ".jpg" not in self.cxr_lists[split]:
                    continue
                if str(row[2]).lower() not in ["yes", "no"]:
                    continue
                unique_id = f'[Close-Ended VQA] [MedVQA-2019] [{row_idx}]'
                study_id = f'[medvqa2019] [{row[0]}]'
                data_source = "MedVQA-2019"
                task_name = "Close-Ended VQA"
                image_path = f'{self.data_dir}/{split}/images/{row[0]}.jpg'
                text = row[1]
                target_text = ""
                options = {"yes": row[2].lower() == "yes", "no": row[2].lower() == "no"}
                qa_pair = form_qa_func(self.instruct)(text, options)
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
                samples.append(sample)
            dataset[split] = samples
        return dataset


if __name__ == '__main__':
    processor = MedVQA2019Processor()
    processor.create_open_ended_vqa()
    processor.create_close_ended_vqa()
