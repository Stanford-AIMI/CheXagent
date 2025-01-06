import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template


class PMCVQAProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/pmc-vqa"
        self.data = {}
        self.cxr_lists = {}
        for split in ["train", "test"]:
            self.data[split] = pd.read_csv(f'{self.data_dir}/{split}_2.csv')
            self.cxr_lists[split] = open(f'{self.data_dir}/list_chest_xray.txt').read().strip().split("\n")
            self.data[split] = self.data[split][
                self.data[split]["Figure_path"].map(lambda x: x in self.cxr_lists[split])
            ]

    def create_open_ended_vqa(self):

        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Open-Ended VQA] [PMC-VQA]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split in ["train", "test"]:
            split_data = self.data[split]
            for row_idx, row in split_data.iterrows():
                unique_id = f'[Open-Ended VQA] [PMC-VQA] [{row_idx}]'
                study_id = f'[pmc-vqa] [{row.Figure_path.split(".")[0]}]'
                data_source = "PMC-VQA"
                task_name = "Open-Ended VQA"
                image_path = f'{self.data_dir}/figures/{row.Figure_path}'
                text = row["Question"].strip()
                target_text = row["Choice " + row["Answer"]].replace(row["Answer"] + ":", "").strip()
                qa_pair = form_qa_func(self.instruct)(text, target_text)
                if "CT" in qa_pair["q"] or "CT" in qa_pair["a"]:
                    continue
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
                dataset[split].append(sample)
        return dataset

    def create_close_ended_vqa(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Close-Ended VQA] [PMC-VQA]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split in ["train", "test"]:
            split_data = self.data[split]
            for row_idx, row in split_data.iterrows():
                unique_id = f'[Close-Ended VQA] [PMC-VQA] [{row_idx}]'
                study_id = f'[pmc-vqa] [{row[0]}]'
                data_source = "PMC-VQA"
                task_name = "Close-Ended VQA"
                image_path = f'{self.data_dir}/figures/{row.Figure_path}'
                text = row["Question"].strip()
                target_text = ""
                options = {
                    row["Choice A"].strip(): row["Answer"] + ":" in row["Choice A"],
                    row["Choice B"].strip(): row["Answer"] + ":" in row["Choice B"],
                    row["Choice C"].strip(): row["Answer"] + ":" in row["Choice C"],
                    row["Choice D"].strip(): row["Answer"] + ":" in row["Choice D"]
                }
                qa_pair = form_qa_func(self.instruct)(text, options)
                if "CT" in qa_pair["q"] or "CT" in qa_pair["a"]:
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
    processor = PMCVQAProcessor()
    processor.create_open_ended_vqa()
    processor.create_close_ended_vqa()
