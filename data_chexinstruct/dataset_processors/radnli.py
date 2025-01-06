import json

from .base_processor import BaseProcessor
from .templates import create_template


class RadNLIProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/radnli/"
        self.val_path = f'{self.data_dir}/radnli_dev_v1.jsonl'
        self.test_path = f'{self.data_dir}/radnli_test_v1.jsonl'
        self.val_data = [json.loads(line) for line in open(self.val_path)]
        self.test_data = [json.loads(line) for line in open(self.test_path)]

    def create_natural_language_inference(self):
        categories = ["neutral", "entailment", "contradiction"]
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Natural Language Inference] [RadNLI]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["val", "test"], [self.val_data, self.test_data]):
            for pair_idx, pair in enumerate(split_data):
                unique_id = f'[Natural Language Inference] [RadNLI] [{pair_idx}]'
                study_id = f'[radnli] [{split}-{pair_idx}]'
                data_source = "RadNLI"
                task_name = "Natural Language Inference"
                image_path = None
                text = [pair["sentence1"], pair["sentence2"]]
                target_text = ""
                options = {cat: cat == pair["gold_label"] for cat in categories}
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
                dataset[split].append(sample)
        return dataset


if __name__ == '__main__':
    processor = RadNLIProcessor()
    processor.create_natural_language_inference()
