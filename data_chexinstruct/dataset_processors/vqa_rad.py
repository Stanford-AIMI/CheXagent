import json

from .base_processor import BaseProcessor
from .templates import create_template


class VQARADProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/vqa-rad"
        self.image_dir = f"{self.data_dir}/images"
        self.train_path = f'{self.data_dir}/trainset.json'
        self.test_path = f'{self.data_dir}/testset.json'
        self.train_data = json.load(open(self.train_path))
        self.test_data = json.load(open(self.test_path))
        self.val_data = self.test_data
        self.cxr_list = open(f'{self.data_dir}/list_chest_xray.txt').read().strip().split("\n")

    def create_open_ended_vqa(self):
        dataset = {"dataset_name": "[Open-Ended VQA] [VQA-RAD]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            samples = []
            for pair_idx, pair in enumerate(split_data):
                if pair["image_name"] not in self.cxr_list:
                    continue
                unique_id = f'[Open-Ended VQA] [VQA-RAD] [{pair_idx}]'
                study_id = f'[vqa-rad] [{pair["image_name"].split(".")[0]}]'
                data_source = "VQA-RAD"
                task_name = "Open-Ended VQA"
                image_path = f'{self.image_dir}/{pair["image_name"]}'
                text = pair["question"]
                target_text = pair["answer"]
                qa_pair = form_qa_func(self.instruct)(text, target_text)
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
                samples.append(sample)
            dataset[split] = samples
        return dataset

    def create_close_ended_vqa(self):
        dataset = {"dataset_name": "[Close-Ended VQA] [VQA-RAD]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            samples = []
            for pair_idx, pair in enumerate(split_data):
                if pair["answer_type"].lower() != "closed" or str(pair["answer"]).lower() not in ["yes", "no"]:
                    continue
                if pair["image_name"] not in self.cxr_list:
                    continue
                unique_id = f'[Close-Ended VQA] [VQA-RAD] [{pair_idx}]'
                study_id = f'[vqa-rad] [{pair["image_name"].split(".")[0]}]'
                data_source = "VQA-RAD"
                task_name = "Close-Ended VQA"
                image_path = f'{self.image_dir}/{pair["image_name"]}'
                text = pair["question"]
                target_text = ""
                options = {"yes": pair["answer"].lower() == "yes", "no": pair["answer"].lower() == "no"}
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
    processor = VQARADProcessor()
    processor.create_open_ended_vqa()
    processor.create_close_ended_vqa()
