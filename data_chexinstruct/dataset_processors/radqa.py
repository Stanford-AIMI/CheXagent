import json
import re

from .base_processor import BaseProcessor
from .templates import create_template


class RadQAProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/radqa/"
        self.train_path = f'{self.data_dir}/train.json'
        self.val_path = f'{self.data_dir}/dev.json'
        self.test_path = f'{self.data_dir}/test.json'
        self.train_data, self.test_data = json.load(open(self.train_path)), json.load(open(self.test_path))
        self.val_data = json.load(open(self.val_path))

    def create_text_qa(self):
        dataset = {"dataset_name": "[Text QA] [RadQA]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.test_data, self.test_data]):
            samples = []
            for pairs_idx, pairs in enumerate(split_data["data"]):
                for pair_idx, pair in enumerate(pairs["paragraphs"]):
                    for qa in pair["qas"]:
                        unique_id = f'[Text QA] [RadQA] [{pairs_idx}-{pair_idx}]'
                        study_id = f'[radqa] [{split}-{pairs_idx}-{pair_idx}]'
                        data_source = "RadQA"
                        task_name = "Text QA"
                        image_path = None
                        text = f'Context: {pair["context"]}\n Q: {qa["question"]}'
                        target_text = "Not answerable" if qa["is_impossible"] else \
                            re.sub("\s+", " ", qa["answers"][0]["text"])
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


if __name__ == '__main__':
    processor = RadQAProcessor()
    processor.create_text_qa()
