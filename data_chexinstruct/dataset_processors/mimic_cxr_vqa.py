import json
import random

from .base_processor import BaseProcessor
from .templates import create_template


class MIMICCXRVQAProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/mimic-cxr-vqa/mimiccxrvqa/dataset/"
        self.image_dir = "data/mimic-cxr/files"
        self.train_path = f'{self.data_dir}/train.json'
        self.val_path = f'{self.data_dir}/valid.json'
        self.test_path = f'{self.data_dir}/test.json'
        self.train_data = json.load(open(self.train_path))
        random.shuffle(self.train_data)
        self.val_data = json.load(open(self.val_path))
        self.test_data = json.load(open(self.test_path))

    def create_open_ended_vqa(self):
        dataset = {"dataset_name": "[Open-Ended VQA] [MIMIC-CXR-VQA]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            samples = []
            for pair_idx, pair in enumerate(split_data):
                if len(pair["answer"]) == 0:
                    continue
                unique_id = f'[Open-Ended VQA] [MIMIC-CXR-VQA] [{pair_idx}]'
                study_id = f'[mimic-cxr] [{int(pair["study_id"])}]'
                data_source = "MIMIC-CXR-VQA"
                task_name = "Open-Ended VQA"
                image_path = f'{self.image_dir}/{pair["image_path"]}'
                text = pair["question"]
                target_text = ", ".join(pair["answer"]).capitalize()
                if target_text.lower() == "f":
                    assert "gender" in text or "sex" in text or "male" in text or "man" in text
                    target_text = "Female"
                elif target_text.lower() == "m":
                    assert "gender" in text or "sex" in text or "male" in text or "man" in text
                    target_text = "Male"
                if "any abnormalities" in text and " in " not in text:
                    continue

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
        dataset = {"dataset_name": "[Close-Ended VQA] [MIMIC-CXR-VQA]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            samples = []
            for pair_idx, pair in enumerate(split_data):
                if len(pair["answer"]) == 0:
                    continue
                if str(pair["answer"][0]).lower() not in ["yes", "no"]:
                    continue
                unique_id = f'[Close-Ended VQA] [MIMIC-CXR-VQA] [{pair_idx}]'
                study_id = f'[mimic-cxr] [{int(pair["study_id"])}]'
                data_source = "MIMIC-CXR-VQA"
                task_name = "Close-Ended VQA"
                image_path = f'{self.image_dir}/{pair["image_path"]}'
                text = pair["question"]
                target_text = ""
                if "any abnormalities" in text and " in " not in text:
                    continue

                options = {"yes": pair["answer"][0].lower() == "yes", "no": pair["answer"][0].lower() == "no"}
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
    processor = MIMICCXRVQAProcessor()
    processor.create_open_ended_vqa()
    processor.create_close_ended_vqa()
