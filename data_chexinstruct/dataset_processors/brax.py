import random

import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template


class BraxProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/brax/"
        self.data_path = f"{self.data_dir}/master_spreadsheet_update.csv"
        self.data = pd.read_csv(self.data_path)

        ranked_views = ['PA', 'AP', 'AP LLD', 'L', 'RL', 'LT-DECUB', 'RLO', ""]
        self.data.ViewPosition = self.data.ViewPosition.fillna("")
        self.data["ViewPositionRank"] = self.data.ViewPosition.map(lambda x: ranked_views.index(x))
        self.data = self.data.sort_values(by='ViewPositionRank')
        self.data = self.data.reset_index(drop=True)
        self.data = self.data.fillna(0)
        self.negative_ratio = 8000 / (self.data.loc[:, "No Finding"] == 1).sum()

    def create_image_classification(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image Classification] [Brax]"}
        form_qa_func = create_template(dataset["dataset_name"])

        for group_id, group in self.data.groupby("AccessionNumber"):
            unique_id = f'[Image Classification] [Brax] [{group_id}]'
            study_id = f"[brax] [{group_id}]"
            data_source = "Brax"
            task_name = "Image Classification"
            image_path = [f'{self.data_dir}/{row.PngPath}' for _, row in group.iterrows()]
            text = ""
            target_text = ""
            if group.iloc[0, 8:22]["No Finding"] == 1:
                if random.random() > self.negative_ratio:
                    continue
            options = group.iloc[0, 8:22].to_dict()
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


if __name__ == '__main__':
    processor = BraxProcessor()
    processor.create_image_classification()
