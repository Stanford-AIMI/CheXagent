import json

from .base_processor import BaseProcessor
from .templates import create_template
from .utils import create_dicom_to_path_mapping, create_dicom_to_meta_mapping, create_study_to_texts_mapping
from .utils import create_study_to_split_mapping, create_study_to_paths_mapping

MIMIC_DIAGNOSIS2LABEL = {
    'Atelectasis': 0,
    'Consolidation': 1,
    'Edema': 2,
    'Enlarged Cardiomediastinum': 3,
    'Lung Lesion': 4,
    'Lung Opacity': 5,
    'Pleural Effusion': 6,
    'Pleural Other': 7,
    'Pneumonia': 8,
    'Pneumothorax': 9
}
MIMIC_LABEL2DIAGNOSIS = {v: k for k, v in MIMIC_DIAGNOSIS2LABEL.items()}

MIMIC_CAT2ONEHOT = {
    'nan': [1, 0, 0],
    '0.0': [1, 0, 0],
    '-1.0': [0, 1, 0],
    '1.0': [0, 0, 1]
}

MIMIC_STR2ONEHOT = {
    'nan': [1, 0, 0],
    'Negative': [1, 0, 0],
    'Uncertain': [0, 1, 0],
    'Positive': [0, 0, 1]
}


class MIMICNLEProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        data_dir = "data/mimic-nle/mimic-nle"
        self.train_data = [json.loads(line) for line in open(f'{data_dir}/mimic-nle-train.json')]
        self.val_data = [json.loads(line) for line in open(f'{data_dir}/mimic-nle-dev.json')]
        self.test_data = [json.loads(line) for line in open(f'{data_dir}/mimic-nle-test.json')]
        self.dicom2path, self.path2dicom = create_dicom_to_path_mapping()
        self.dicom2meta = create_dicom_to_meta_mapping()
        self.study2texts = create_study_to_texts_mapping()
        self.study2split = create_study_to_split_mapping()
        self.study2path = create_study_to_paths_mapping()

    def create_natural_language_explanation(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Natural Language Explanation] [MIMIC-NLE]",
                   "diagnosis2label": MIMIC_DIAGNOSIS2LABEL, "label2diagnosis": MIMIC_LABEL2DIAGNOSIS,
                   "cat2onehot": MIMIC_CAT2ONEHOT, "str2onehot": MIMIC_STR2ONEHOT}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            for item_idx, item in enumerate(split_data):
                study_id = int(item.pop("sentence_ID").split("#")[0][1:])
                split = self.study2split[study_id]
                unique_id = f"[Natural Language Explanation] [MIMIC-NLE] [{split}-{item_idx}]"
                data_source = "MIMIC-NLE"
                task_name = "Natural Language Explanation"
                image_path = self.study2path[study_id]
                text = ""
                target_text = ""
                options = item
                qa_pair = form_qa_func(self.instruct)(options)
                sample = {
                    'unique_id': unique_id,
                    'study_id': f'[mimic-cxr] [{study_id}]',
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
    processor = MIMICNLEProcessor()
    processor.create_natural_language_explanation()
