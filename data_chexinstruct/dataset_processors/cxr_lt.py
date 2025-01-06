import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template
from .utils import create_study_to_paths_mapping

CXRLT_TASK = [
    'Calcification of the Aorta',
    'Emphysema',
    'Fibrosis',
    'Hernia',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pleural Thickening',
    'Pneumomediastinum',
    'Pneumoperitoneum',
    'Subcutaneous Emphysema',
    'Tortuous Aorta'
]


class CXRLTProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/cxr-lt/"
        self.train_path = f"{self.data_dir}/train.csv"
        self.train_data = pd.read_csv(self.train_path)
        self.train_data = self.train_data.dropna(subset=["study_id"])
        self.study2path = create_study_to_paths_mapping()

    def create_image_classification(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image Classification] [CXR-LT]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train"], [self.train_data]):
            for group_id, group in split_data.groupby("study_id"):
                unique_id = f'[Image Classification] [CXR-LT] [{group_id}]'
                study_id = f'[mimic-cxr] [{int(group_id)}]'
                data_source = "CXR-LT"
                task_name = "Image Classification"
                image_path = self.study2path[group_id]
                text = ""
                target_text = ""
                options = group.loc[:, CXRLT_TASK].iloc[0].to_dict()
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
                dataset[split].append(sample)
        return dataset


if __name__ == '__main__':
    processor = CXRLTProcessor()
    processor.create_image_classification()
