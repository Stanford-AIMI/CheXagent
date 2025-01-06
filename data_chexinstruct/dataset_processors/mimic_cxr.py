import copy
import random

import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template
from .utils import create_dicom_to_meta_mapping, create_dicom_to_path_mapping
from .utils import create_study_to_paths_mapping, create_study_to_split_mapping
from .utils import create_study_to_texts_mapping

CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]

CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}


class MIMICCXRProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/mimic-cxr"
        self.split_path = f"{self.data_dir}/mimic-cxr-2.0.0-split.csv"
        self.chexpert_path = f"{self.data_dir}/mimic-cxr-2.0.0-chexpert.csv"
        self.chexpert_data = pd.read_csv(self.chexpert_path)
        self.study2split = create_study_to_split_mapping()
        self.study2paths = create_study_to_paths_mapping()
        self.dicom2meta = create_dicom_to_meta_mapping()
        self.dicom2path, self.path2dicom = create_dicom_to_path_mapping()
        self.study2texts = create_study_to_texts_mapping()

        self.chexpert_data = self.chexpert_data.fillna(0)
        uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        self.chexpert_data = self.chexpert_data.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

        self.study2labels = {}
        for idx, row in self.chexpert_data.iterrows():
            labels = set([k for k, v in row[2:].to_dict().items() if v > 0])
            if len(labels) == 0:
                labels = set(['No Finding'])
            self.study2labels[str(int(row['study_id']))] = labels

        self.negative_ratio = 50000 / (self.chexpert_data.loc[:, "No Finding"] == 1).sum()

    def create_image_classification(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image Classification] [MIMIC-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for group_id, group in self.chexpert_data.groupby("study_id"):
            unique_id = f'[Image Classification] [MIMIC-CXR] [{group_id}]'
            study_id = f'[mimic-cxr] [{int(group_id)}]'
            data_source = "MIMIC-CXR"
            task_name = "Image Classification"
            image_path = self.study2paths[group_id]
            text = ""
            target_text = ""
            if group.iloc[0, 2:]["No Finding"] == 1:
                if self.study2split[group_id] == "train" and random.random() > self.negative_ratio:
                    continue

            options = group.iloc[0, 2:].to_dict()
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
            split = self.study2split[group_id]
            dataset[split].append(sample)
        return dataset

    def create_view_classification(self):
        categories = self.dicom2meta.ViewPosition.dropna().unique().tolist()
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[View Classification] [MIMIC-CXR]"}

        option_sets = ["AP", "PA", "LATERAL", "LL"]
        form_qa_func = create_template(dataset["dataset_name"])
        for row_idx, row in self.dicom2meta.iterrows():
            if not isinstance(row.ViewPosition, str):
                continue
            if isinstance(row.ViewPosition, float):
                continue
            if row.ViewPosition not in option_sets:
                continue
            unique_id = f'[View Classification] [MIMIC-CXR] [{row_idx}]'
            study_id = f'[mimic-cxr] [{int(row.study_id)}]'
            data_source = "MIMIC-CXR"
            task_name = "View Classification"
            image_path = self.dicom2path[row_idx]
            text = ""
            target_text = ""
            _option_sets = copy.deepcopy(option_sets)
            view_position = row.ViewPosition
            if random.random() > 0.5:
                _option_sets = ["AP", "PA", "LATERAL"]
                view_position = view_position.replace("LL", "LATERAL")
            options = {option: option == view_position for option in _option_sets}
            qa_pair = form_qa_func(self.instruct)(options)
            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'data_source': data_source,
                'task_name': task_name,
                'image_path': image_path,
                "text": text,
                'target_text': target_text,
                "options": options,
                "qa_pair": qa_pair,
            }
            split = self.study2split[row.study_id]
            dataset[split].append(sample)
        return dataset

    def create_view_matching(self):
        def filter_func(group):
            return len(group) > 2 and (any(group['ViewPosition'] == "AP") or any(group['ViewPosition'] == "PA")) \
                and any(group['ViewPosition'] == "LATERAL")

        dicom2meta = self.dicom2meta[self.dicom2meta.ViewPosition.map(lambda x: x in ["AP", "PA", "LATERAL"])]
        dicom2meta = dicom2meta.groupby(["study_id"]).filter(filter_func)
        dicom2meta["split"] = dicom2meta.study_id.map(lambda x: self.study2split[x])

        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[View Matching] [MIMIC-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split in ["train", "val", "test"]:
            front_cases = dicom2meta[dicom2meta.ViewPosition.map(lambda x: x in ["PA", "PA"])]
            lateral_cases = dicom2meta[dicom2meta.ViewPosition.map(lambda x: x in ["LATERAL"])]
            lateral_cases = lateral_cases.reset_index("dicom_id").set_index("study_id")
            front_cases = front_cases[front_cases.split == split]
            lateral_cases = lateral_cases[lateral_cases.split == split]
            for row_idx, row in front_cases.iterrows():
                data_source = "MIMIC-CXR"
                # Positive
                unique_id = f'[View Matching] [MIMIC-CXR] [{row_idx}-positive]'
                study_id = f'[mimic-cxr] [{int(row.study_id)}]'
                selected = lateral_cases.loc[row.study_id]
                selected = (selected.sample().dicom_id.item()
                            if isinstance(selected, pd.DataFrame) else selected.dicom_id)
                image_path = [self.dicom2path[row_idx], self.dicom2path[selected]]
                task_name = "View Matching"
                text = ""
                target_text = ""
                options = {"Yes": True, "No": False}
                qa_pair = form_qa_func(self.instruct)(options)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'data_source': data_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    "text": text,
                    'target_text': target_text,
                    "options": options,
                    "qa_pair": qa_pair,
                }
                dataset[split].append(sample)
                # Negative
                unique_id = f'[MIMIC-CXR] [View Matching] [{row_idx}-negative]'
                study_id = f'[mimic-cxr] [{int(row.study_id)}]'
                image_path = [
                    self.dicom2path[row_idx],
                    self.dicom2path[lateral_cases[lateral_cases.index != row.study_id].sample().dicom_id.item()]
                ]
                task_name = "View Matching"
                text = ""
                target_text = ""
                options = {"Yes": False, "No": True}
                qa_pair = form_qa_func(self.instruct)(options)
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

    def create_findings_summarization(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Findings Summarization] [MIMIC-CXR]"}
        study2texts = self.study2texts.dropna(subset=["findings", "impression"])
        study2texts = study2texts[study2texts.findings.map(lambda x: len(x) != 0)]
        study2texts = study2texts[study2texts.impression.map(lambda x: len(x) != 0)]

        form_qa_func = create_template(dataset["dataset_name"])
        for study_id, study in study2texts.iterrows():
            unique_id = f'[Findings Summarization] [MIMIC-CXR] [{study_id}]'
            data_source = "MIMIC-CXR"
            task_name = "Findings Summarization"
            image_path = None
            text = study.findings
            target_text = study.impression
            if any([ignore in target_text for ignore in ["___", "a.m.", "p.m."]]):
                continue
            qa_pair = form_qa_func(self.instruct)(text, target_text)
            sample = {
                'unique_id': unique_id,
                'study_id': f'[mimic-cxr] [{int(study_id[1:])}]',
                'data_source': data_source,
                'task_name': task_name,
                'image_path': image_path,
                "text": text,
                'target_text': target_text,
                "qa_pair": qa_pair
            }
            split = self.study2split[int(study_id[1:])]
            dataset[split].append(sample)
        return dataset

    def create_image_text_matching(self):
        def randint_with_excluded_value(a, b, e, labels):
            while True:
                randint = random.randint(a, b)
                if randint != e and labels[randint] != labels[e]:
                    break
            return randint

        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image-Text Matching] [MIMIC-CXR]"}
        study2findings = self.study2texts.findings.dropna()
        study2impression = self.study2texts.impression.dropna()
        study2texts = pd.concat([study2findings, study2impression], axis=0)
        study2texts = study2texts[study2texts.map(lambda x: len(x) != 0)]
        study2texts = pd.DataFrame({
            "text": study2texts,
            "split": study2texts.index.map(lambda x: self.study2split[int(x[1:])])
        })

        form_qa_func = create_template(dataset["dataset_name"])
        for split in ["train", "val", "test"]:
            split_data = study2texts.text[study2texts.split.map(lambda x: x == split)]
            split_data = split_data.reset_index("study")
            a, b = 0, len(split_data) - 1
            labels = split_data["study"].apply(lambda x: self.study2labels[x[1:]]).values.tolist()
            split_data["n_text"] = split_data.index.map(
                lambda e: split_data.text.loc[randint_with_excluded_value(a, b, e, labels)]
            )
            split_data = split_data.set_index("study")
            for study_id, row in split_data.iterrows():
                unique_id = f'[Image-Text Matching] [MIMIC-CXR] [{study_id}]'
                data_source = "MIMIC-CXR"
                task_name = "Image-Text Matching"
                image_path = self.study2paths[int(study_id[1:])]
                # Positive
                if len(row.text.split(". ")) > 1:
                    if random.random() < 0.2:
                        splits = row.text.split(". ")
                        text = ". ".join(random.sample(splits, random.randint(1, len(splits))))
                    else:
                        text = row.text
                else:
                    text = row.text
                target_text = ""
                options = {"Matched": True, "Not matched": False}
                qa_pair = form_qa_func(self.instruct)(text, options)
                sample = {
                    'unique_id': unique_id,
                    'study_id': f'[mimic-cxr] [{int(study_id[1:])}]',
                    'data_source': data_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'text': text,
                    'options': options,
                    'target_text': target_text,
                    "qa_pair": qa_pair,
                }
                split = self.study2split[int(study_id[1:])]
                dataset[split].append(sample)
                # Negative
                text = row.n_text
                target_text = ""
                options = {"Matched": False, "Not matched": True}
                qa_pair = form_qa_func(self.instruct)(text, options)
                sample = {
                    'unique_id': unique_id,
                    'study_id': f'[mimic-cxr] [{int(study_id[1:])}]',
                    'data_source': data_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'text': text,
                    'options': options,
                    'target_text': target_text,
                    "qa_pair": qa_pair
                }
                split = self.study2split[int(study_id[1:])]
                dataset[split].append(sample)
        return dataset

    def create_image_text_selection(self):
        def randint_with_excluded_value(a, b, e, labels):
            while True:
                randint = random.randint(a, b)
                if randint != e and labels[randint] != labels[e]:
                    break
            return randint

        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image-Text Selection] [MIMIC-CXR]"}
        study2findings = self.study2texts.findings.dropna()
        study2impression = self.study2texts.impression.dropna()
        study2texts = pd.concat([study2findings, study2impression], axis=0)
        study2texts = study2texts[study2texts.map(lambda x: len(x) != 0)]
        study2texts = pd.DataFrame({
            "text": study2texts,
            "split": study2texts.index.map(lambda x: self.study2split[int(x[1:])])
        })
        form_qa_func = create_template(dataset["dataset_name"])
        for split in ["train", "val", "test"]:
            split_data = study2texts.text[study2texts.split.map(lambda x: x == split)]
            split_data = split_data.reset_index("study")
            a, b = 0, len(split_data) - 1
            labels = split_data["study"].apply(lambda x: self.study2labels[x[1:]]).values.tolist()
            split_data["n_text_1"] = split_data.index.map(
                lambda e: split_data.text.loc[randint_with_excluded_value(a, b, e, labels)]
            )
            split_data = split_data.set_index("study")
            for study_id, row in split_data.iterrows():
                unique_id = f'[Image-Text Selection] [MIMIC-CXR] [{study_id}]'
                data_source = "MIMIC-CXR"
                task_name = "Image-Text Selection"
                image_path = self.study2paths[int(study_id[1:])]
                text = row.text
                target_text = ""
                options = [(row.text, True), (row.n_text_1, False)]
                ordered_options = [(k, v) for k, v in options]
                random.shuffle(ordered_options)
                qa_pair = form_qa_func(self.instruct)(ordered_options)
                sample = {
                    'unique_id': unique_id,
                    'study_id': f'[mimic-cxr] [{int(study_id[1:])}]',
                    'data_source': data_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'text': text,
                    'options': options,
                    'target_text': target_text,
                    "qa_pair": qa_pair,
                }
                split = self.study2split[int(study_id[1:])]
                dataset[split].append(sample)
        return dataset


if __name__ == '__main__':
    processor = MIMICCXRProcessor()
    processor.create_image_classification()
    processor.create_view_classification()
    processor.create_view_matching()
    processor.create_findings_summarization()
    processor.create_image_text_matching()
    processor.create_image_text_selection()
