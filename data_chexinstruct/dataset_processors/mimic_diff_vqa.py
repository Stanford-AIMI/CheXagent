import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template
from .utils import create_dicom_to_path_mapping, create_study_to_split_mapping


class MIMICDiffVQAProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/mimic-diff-vqa"
        self.image_dir = "data/mimic-cxr/files"
        self.ann_path = f'{self.data_dir}/code/temp/mimic_pair_questions_temp.csv'
        self.ann_data = pd.read_csv(open(self.ann_path))
        self.data_mimic_all = pd.read_csv(f"{self.data_dir}/mimic_all.csv")
        self.data_mimic_all = self.data_mimic_all.set_index("study_id")
        self.dicom2path = create_dicom_to_path_mapping()[0]
        self.study2split = create_study_to_split_mapping()

        self.ann_data = self.ann_data[
            self.ann_data.question_type.map(lambda x: x in ["location", "level", "type"])
        ]

    def create_difference_vqa(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Difference VQA] [MIMIC-Diff-VQA]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for row_idx, row in self.ann_data.iterrows():
            if self.study2split[row.study_id] != self.study2split[row.ref_id]:
                continue
            unique_id = f'[Difference VQA] [MIMIC-Diff-VQA] [{row_idx}]'
            study_id = f'[mimic-cxr] [{int(row.study_id)}]'
            data_source = "MIMIC-Diff-VQA"
            task_name = "Difference VQA"
            image_path = [self.dicom2path[self.data_mimic_all.loc[row.study_id].dicom_id]]
            text = row["question"].capitalize()
            target_text = ". ".join([sent.capitalize() for sent in row["answer"].strip().split(". ")])
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
            dataset[self.study2split[row.study_id]].append(sample)
        return dataset


if __name__ == '__main__':
    processor = MIMICDiffVQAProcessor()
    processor.create_difference_vqa()
