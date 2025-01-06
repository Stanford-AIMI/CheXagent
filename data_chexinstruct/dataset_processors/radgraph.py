import json

from .base_processor import BaseProcessor
from .templates import create_template
from .utils import create_study_to_split_mapping


class RadGraphProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/radgraph"
        self.train_path = f'{self.data_dir}/train.json'
        self.val_path = f'{self.data_dir}/dev.json'
        self.test_path = f'{self.data_dir}/test.json'
        self.train_data = json.load(open(self.train_path))
        self.val_data = json.load(open(self.val_path))
        self.test_data = json.load(open(self.test_path))
        self.study2split = create_study_to_split_mapping()

    def create_named_entity_recognition(self):
        name_mapping = {
            "OBS-DA": "Observation (Definitely Absent)",
            "OBS-DP": "Observation (Definitely Present)",
            "OBS-U": " Observation (Uncertain)",
            "ANAT-DP": "Anatomy"
        }

        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Named Entity Recognition] [RadGraph]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            for sample_idx, sample in split_data.items():
                unique_id = f'[Named Entity Recognition] [RadGraph] [{sample_idx}]'
                study_id = f'[mimic-cxr] [{sample_idx.split("/")[-1].split(".")[0][1:]}]'
                data_source = "RadGraph"
                task_name = "Named Entity Recognition"
                image_path = None
                text = sample["text"]
                target_text = ""
                try:
                    options = [
                        [name_mapping[sample["entities"][k]["label"]], sample["entities"][k]["tokens"]]
                        for k in sample["entities"]
                    ]
                except:
                    options = [
                        [
                            name_mapping[sample["labeler_1"]["entities"][k]["label"]],
                            sample["labeler_1"]["entities"][k]["tokens"]
                        ]
                        for k in sample["labeler_1"]["entities"]
                    ]
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
                dataset[
                    self.study2split[int(sample_idx.split("/")[-1].split(".")[0][1:])]
                    if len(sample_idx.split("/")[-1].split(".")[0][1:]) >= 5 else "test"
                ].append(sample)
        return dataset


if __name__ == '__main__':
    processor = RadGraphProcessor()
    processor.create_named_entity_recognition()
