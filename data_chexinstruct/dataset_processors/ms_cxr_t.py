import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template
from .utils import create_study_to_split_mapping


class MSCXRTProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/ms-cxr-t/"
        # Temporal Image Classification
        self.image_dir = "data/mimic-cxr/files"
        self.test_path = f'{self.data_dir}/MS_CXR_T_temporal_image_classification_v1.0.0.csv'
        self.test_data = pd.read_csv(self.test_path)
        # Temporal Sentence Similarity
        self.test_tss_path = f'{self.data_dir}/MS_CXR_T_temporal_sentence_similarity_v1.0.0.csv'
        self.test_tss_data = pd.read_csv(self.test_tss_path)
        self.study2split = create_study_to_split_mapping()

    def create_temporal_image_classification(self):
        diseases = ["consolidation", "edema", "pleural_effusion", "pneumonia", "pneumothorax"]
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Temporal Image Classification] [MS-CXR-T]"}

        option_set = ["worsening", "stable", "improving"]
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["test"], [self.test_data]):
            for row_idx, row in split_data.iterrows():
                unique_id = f'[Temporal Image Classification] [MS-CXR-T] [{row_idx}]'
                study_id = f'[ms-cxr-t-tic] [{split}-{row_idx}]'
                data_source = "MS-CXR-T"
                task_name = "Temporal Image Classification"
                image_path = [f'{self.image_dir}/{row.previous_dicom_id}.jpg', f'{self.image_dir}/{row.dicom_id}.jpg']
                text = ""
                target_text = ""
                for disease in diseases:
                    if isinstance(row[f'{disease}_progression'], float):
                        continue
                    if row[f'{disease}_progression'] not in option_set:
                        continue
                    options = {option: option == row[f'{disease}_progression'] for option in option_set}
                    qa_pair = form_qa_func(self.instruct)(options, disease.replace("_", " "))

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
                    dataset[self.study2split[row.study_id]].append(sample)
        return dataset

    def create_temporal_sentence_similarity(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Temporal Sentence Similarity] [MS-CXR-T]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["test"], [self.test_tss_data]):
            for row_idx, row in split_data.iterrows():
                unique_id = f'[Temporal Sentence Similarity] [MS-CXR-T] [{row_idx}]'
                study_id = f'[ms-cxr-t-tss] [{split}-{row_idx}]'
                data_source = "MS-CXR-T"
                task_name = "Temporal Sentence Similarity"
                text = [row.sentence_1, row.sentence_2]
                target_text = None
                options = {
                    'contradiction': row.category == 'contradiction',
                    'paraphrase': row.category == "paraphrase"
                }
                qa_pair = form_qa_func(self.instruct)(text, options)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'data_source': data_source,
                    'task_name': task_name,
                    'image_path': None,
                    "text": text,
                    'target_text': target_text,
                    "options": options,
                    "qa_pair": qa_pair
                }
                dataset[split].append(sample)
        return dataset


if __name__ == '__main__':
    processor = MSCXRTProcessor()
    processor.create_temporal_image_classification()
    processor.create_temporal_sentence_similarity()
