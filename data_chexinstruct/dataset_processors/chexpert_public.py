import json
import os
import random
import re
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template

keywords = [
    "new", "change", "unchanged", "prior", "stable", "interval", "previous", "again", "increased", "improve",
    "remain", "worse", "persistent", "remov", "decrease", "similar", "earlier", "recurrence", "redemonstrate"
]

label_set = [
    'No Finding', 'Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation',
    'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices'
]

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


def format_report(report):
    ignores = ["prior", "previous", "remov", "earlier", "recurrence", "redemonstrate"]
    report = ". ".join(
        [sentence for sentence in report.split(". ") if all([ignore not in sentence.lower() for ignore in ignores])]
    )
    report = report.replace(" again", "").replace("again ", "")
    report = report.replace(" new.", " present.")
    report = report.replace(" new ", " ")
    return report


def parse_report(x):
    # Regular expression pattern to match [Section: Subsection] and associated descriptions.
    pattern = re.compile(r'\[(?P<section>[^\]]+): (?P<subsection>[^\]]+)\] (?P<description>[^\[]+)')

    # Extract matches and organize them into a dictionary
    anatomy_description = defaultdict(list)
    for match in pattern.finditer(x):
        section = match.group('section')
        subsection = match.group('subsection')
        value = match.group('description').strip()
        anatomy_description[section].append(value)
        anatomy_description[subsection].append(value)

    src_key = 'Everything else'
    tgt_key = "Everything else (e.g., mediastinal contours, bones, soft tissues, tubes, valves, and pacemakers)"
    if 'Everything else' in anatomy_description:
        anatomy_description[tgt_key] = anatomy_description.pop(src_key)
    return anatomy_description


class CheXpertPublicProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/chexpert-public"
        self.train_path = f"{self.data_dir}/train.csv"
        self.val_path = f"{self.data_dir}/valid.csv"
        self.test_path = f"{self.data_dir}/test.csv"
        self.train_data = pd.read_csv(self.train_path)
        self.val_data = pd.read_csv(self.val_path)
        self.test_data = pd.read_csv(self.test_path)

        self.train_data.Path = self.train_data.Path.map(lambda x: x.replace("CheXpert-v1.0-small/", ""))
        self.val_data.Path = self.val_data.Path.map(lambda x: x.replace("CheXpert-v1.0-small/", ""))
        self.text_data = pd.read_csv(f"{self.data_dir}/df_master_77_for_release_with_dcm.csv")
        self.text_data = self.text_data.sort_values("patient_report_date_order")

        self.train_data.loc[:, "AP/PA"] = self.train_data["AP/PA"].fillna("Lateral")
        self.train_data = self.train_data.fillna(0)
        self.val_data["AP/PA"] = self.val_data["AP/PA"].fillna("Lateral")
        self.test_data["AP/PA"] = self.val_data["AP/PA"].fillna("Lateral")
        self.text_data["AP/PA"] = self.text_data["ap_pa"].fillna("Lateral")

        self.text_data["study_id"] = self.text_data.path_to_image.map(lambda x: "_".join(x.split("/")[2:4]))
        order = ["PA", "AP", "Lateral", "LL", "RL"]
        for df in [self.train_data, self.val_data, self.test_data, self.text_data]:
            df['AP/PA'] = pd.Categorical(df['AP/PA'], categories=order, ordered=True)
            df.sort_values('AP/PA', inplace=True)

        self.text_data = self.text_data.set_index("study_id")
        self.study2original = {k: v["original_index"] for k, v in self.text_data.iterrows()}
        self.findings = json.load(open(f"{self.data_dir}/texts/findings_clean.json"))
        self.impression = json.load(open(f"{self.data_dir}/texts/impression_clean.json"))
        self.raw_findings = json.load(open(f"{self.data_dir}/texts/dict_findings.json"))
        self.raw_impression = json.load(open(f"{self.data_dir}/texts/dict_impression.json"))

        uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        self.train_data = self.train_data.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

    def create_view_classification(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[View Classification] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            for row_idx, row in split_data.iterrows():
                if row["AP/PA"] not in ["AP", "PA", "Lateral"]:
                    continue
                unique_id = f'[View Classification] [CheXpert-Public] [{row_idx}]'
                study_id = f'[chexpert] [{row_idx}]'
                data_source = "CheXpert-Public"
                task_name = "View Classification"
                image_path = f"{self.data_dir}/{row.Path}"
                text = ""
                target_text = ""
                options = {"AP": row["AP/PA"] == "AP", "PA": row["AP/PA"] == "PA", "Lateral": row["AP/PA"] == "Lateral"}

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

    def create_image_classification(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image Classification] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            for group_idx, group in split_data.groupby("study_id"):
                unique_id = f'[Image Classification] [CheXpert-Public] [{group_idx}]'
                study_id = f'[chexpert] [{group_idx}]'
                data_source = "CheXpert-Public"
                task_name = "Image Classification"
                image_path = [f"{self.data_dir}/{path}" for path in group.Path.tolist()]
                text = ""
                target_text = ""
                options = group.iloc[0][label_set].to_dict()
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

    def create_findings_generation(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Findings Generation] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            all_study_ids = set(self.text_data.index.tolist())
            for group_idx, group in split_data.groupby("study_id"):
                if group_idx not in all_study_ids:
                    continue
                original_index = str(self.study2original[group_idx])
                if original_index not in self.findings:
                    continue
                unique_id = f'[Findings Generation] [CheXpert-Public] [{group_idx}]'
                study_id = f'[chexpert] [{group_idx}]'
                data_source = "CheXpert-Public"
                task_name = "Findings Generation"
                image_path = [f"{self.data_dir}/{path}" for path in group.Path.tolist()]
                text = ""
                target_text = format_report(self.findings[original_index])
                qa_pair = form_qa_func(self.instruct)(target_text)
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

    def create_findings_generation_with_indication(self):
        dataset = {"train": [], "val": [], "test": [],
                   "dataset_name": "[Findings Generation with Indication] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            all_study_ids = set(self.text_data.index.tolist())
            for group_idx, group in split_data.groupby("study_id"):
                if group_idx not in all_study_ids:
                    continue
                original_index = str(self.study2original[group_idx])
                if original_index not in self.findings:
                    continue
                unique_id = f'[Findings Generation] [CheXpert-Public] [{group_idx}]'
                study_id = f'[chexpert] [{group_idx}]'
                data_source = "CheXpert-Public"
                task_name = "Findings Generation"
                image_path = [f"{self.data_dir}/{path}" for path in group.Path.tolist()]

                clinical_history = self.text_data.loc[group_idx].section_clinical_history
                if isinstance(clinical_history, pd.Series):
                    clinical_history = clinical_history.iloc[0]
                if isinstance(clinical_history, float):
                    clinical_history = "None"
                clinical_history = re.sub("\s+", " ", clinical_history).strip()

                text = clinical_history
                target_text = format_report(self.findings[original_index])
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
                dataset[split].append(sample)
        return dataset

    def create_impression_generation(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Impression Generation] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            all_study_ids = set(self.text_data.index.tolist())
            for group_idx, group in split_data.groupby("study_id"):
                if group_idx not in all_study_ids:
                    continue
                original_index = str(self.study2original[group_idx])
                if original_index not in self.impression:
                    continue
                unique_id = f'[Impression Generation] [CheXpert-Public] [{group_idx}]'
                study_id = f'[chexpert] [{group_idx}]'
                data_source = "CheXpert-Public"
                task_name = "Impression Generation"
                image_path = [f"{self.data_dir}/{path}" for path in group.Path.tolist()]
                text = ""
                target_text = format_report(self.impression[original_index])
                qa_pair = form_qa_func(self.instruct)(target_text)
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

    def create_impression_generation_with_indication(self):
        dataset = {"train": [], "val": [], "test": [],
                   "dataset_name": "[Impression Generation with Indication] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            all_study_ids = set(self.text_data.index.tolist())
            for group_idx, group in split_data.groupby("study_id"):
                if group_idx not in all_study_ids:
                    continue
                original_index = str(self.study2original[group_idx])
                if original_index not in self.impression:
                    continue
                unique_id = f'[Impression Generation] [CheXpert-Public] [{group_idx}]'
                study_id = f'[chexpert] [{group_idx}]'
                data_source = "CheXpert-Public"
                task_name = "Impression Generation"
                image_path = [f"{self.data_dir}/{path}" for path in group.Path.tolist()]

                clinical_history = self.text_data.loc[group_idx].section_clinical_history
                if isinstance(clinical_history, pd.Series):
                    clinical_history = clinical_history.iloc[0]
                if isinstance(clinical_history, float):
                    clinical_history = "None"
                clinical_history = re.sub("\s+", " ", clinical_history).strip()

                text = clinical_history
                target_text = format_report(self.impression[original_index])
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
                dataset[split].append(sample)
        return dataset

    def create_local_findings_generation(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Local Findings Generation] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            all_study_ids = set(self.text_data.index.tolist())
            for group_idx, group in split_data.groupby("study_id"):
                if group_idx not in all_study_ids:
                    continue
                original_index = str(self.study2original[group_idx])
                if original_index not in self.findings:
                    continue
                unique_id = f'[Local Findings Generation] [CheXpert-Public] [{group_idx}]'
                study_id = f'[chexpert] [{group_idx}]'
                data_source = "CheXpert-Public"
                task_name = "Local Findings Generation"
                image_path = [f"{self.data_dir}/{path}" for path in group.Path.tolist()]
                for anatomy, description in parse_report(self.findings[original_index]).items():
                    text = anatomy
                    target_text = " ".join(description)
                    target_text = format_report(target_text)
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
                    dataset[split].append(sample)
        return dataset

    def create_local_impression_generation(self):
        dataset = {"train": [], "val": [], "test": [],
                   "dataset_name": "[Local Impression Generation] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            all_study_ids = set(self.text_data.index.tolist())
            for group_idx, group in split_data.groupby("study_id"):
                if group_idx not in all_study_ids:
                    continue
                original_index = str(self.study2original[group_idx])
                if original_index not in self.impression:
                    continue
                unique_id = f'[Local Impression Generation] [CheXpert-Public] [{group_idx}]'
                study_id = f'[chexpert] [{group_idx}]'
                data_source = "CheXpert-Public"
                task_name = "Local Impression Generation"
                image_path = [f"{self.data_dir}/{path}" for path in group.Path.tolist()]
                for anatomy, description in parse_report(self.impression[original_index]).items():
                    text = anatomy
                    target_text = " ".join(description)
                    target_text = format_report(target_text)
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
                    dataset[split].append(sample)
        return dataset

    def create_progression_findings_generation(self):
        dataset = {"train": [], "val": [], "test": [],
                   "dataset_name": "[Progression Findings Generation] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            all_study_ids = set(self.text_data.index.tolist())
            study2paths = {group_idx: group.Path.tolist() for group_idx, group in split_data.groupby("study_id")}
            for group_idx, group in split_data.groupby("study_id"):
                if group_idx not in all_study_ids:
                    continue
                original_index = str(self.study2original[group_idx])
                if original_index not in self.findings:
                    continue
                previous_study = f'{group_idx[:-1]}{int(group_idx[-1]) - 1}'
                if previous_study not in study2paths:
                    continue
                if not any([keyword in self.findings[original_index].lower() for keyword in keywords]):
                    continue
                unique_id = f'[Progression Findings Generation] [CheXpert-Public] [{group_idx}]'
                study_id = f'[chexpert] [{group_idx}]'
                data_source = "CheXpert-Public"
                task_name = "Progression Findings Generation"
                image_path = [
                    f"{self.data_dir}/{path}" for path in [study2paths[previous_study][0], study2paths[group_idx][0]]
                ]
                text = ""
                target_text = self.findings[original_index]
                qa_pair = form_qa_func(self.instruct)(target_text)
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

    def create_progression_impression_generation(self):
        dataset = {"train": [], "val": [], "test": [],
                   "dataset_name": "[Progression Impression Generation] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            all_study_ids = set(self.text_data.index.tolist())
            study2paths = {group_idx: group.Path.tolist() for group_idx, group in split_data.groupby("study_id")}
            for group_idx, group in split_data.groupby("study_id"):
                if group_idx not in all_study_ids:
                    continue
                original_index = str(self.study2original[group_idx])
                if original_index not in self.impression:
                    continue
                previous_study = f'{group_idx[:-1]}{int(group_idx[-1]) - 1}'
                if previous_study not in study2paths:
                    continue
                if not any([keyword in self.impression[original_index].lower() for keyword in keywords]):
                    continue
                unique_id = f'[Progression Impression Generation] [CheXpert-Public] [{group_idx}]'
                study_id = f'[chexpert] [{group_idx}]'
                data_source = "CheXpert-Public"
                task_name = "Progression Impression Generation"
                image_path = [
                    f"{self.data_dir}/{path}" for path in [study2paths[previous_study][0], study2paths[group_idx][0]]
                ]
                text = ""
                target_text = self.impression[original_index]
                qa_pair = form_qa_func(self.instruct)(target_text)
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

    def create_local_progression_findings_generation(self):
        dataset = {"train": [], "val": [], "test": [],
                   "dataset_name": "[Local Progression Findings Generation] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            all_study_ids = set(self.text_data.index.tolist())
            study2paths = {group_idx: group.Path.tolist() for group_idx, group in split_data.groupby("study_id")}
            for group_idx, group in split_data.groupby("study_id"):
                if group_idx not in all_study_ids:
                    continue
                original_index = str(self.study2original[group_idx])
                if original_index not in self.findings:
                    continue
                previous_study = f'{group_idx[:-1]}{int(group_idx[-1]) - 1}'
                if previous_study not in study2paths:
                    continue
                if not any([keyword in self.findings[original_index].lower() for keyword in keywords]):
                    continue
                unique_id = f'[Local Progression Findings Generation] [CheXpert-Public] [{group_idx}]'
                study_id = f'[chexpert] [{group_idx}]'
                data_source = "CheXpert-Public"
                task_name = "Local Progression Findings Generation"
                image_path = [
                    f"{self.data_dir}/{path}" for path in [study2paths[previous_study][0], study2paths[group_idx][0]]
                ]
                for anatomy, description in parse_report(self.findings[original_index]).items():
                    text = anatomy
                    target_text = " ".join(description)
                    target_text = target_text
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
                    dataset[split].append(sample)
        return dataset

    def create_local_progression_impression_generation(self):
        dataset = {"train": [], "val": [], "test": [],
                   "dataset_name": "[Local Progression Impression Generation] [CheXpert-Public]"}
        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.val_data, self.test_data]):
            split_data["study_id"] = split_data.Path.map(lambda x: "_".join(x.split("/")[1:3]))
            all_study_ids = set(self.text_data.index.tolist())
            study2paths = {group_idx: group.Path.tolist() for group_idx, group in split_data.groupby("study_id")}
            for group_idx, group in split_data.groupby("study_id"):
                if group_idx not in all_study_ids:
                    continue
                original_index = str(self.study2original[group_idx])
                if original_index not in self.impression:
                    continue
                previous_study = f'{group_idx[:-1]}{int(group_idx[-1]) - 1}'
                if previous_study not in study2paths:
                    continue
                if not any([keyword in self.impression[original_index].lower() for keyword in keywords]):
                    continue
                unique_id = f'[Local Progression Impression Generation] [CheXpert-Public] [{group_idx}]'
                study_id = f'[chexpert] [{group_idx}]'
                data_source = "CheXpert-Public"
                task_name = "Local Progression Impression Generation"
                image_path = [
                    f"{self.data_dir}/{path}" for path in [study2paths[previous_study][0], study2paths[group_idx][0]]
                ]
                for anatomy, description in parse_report(self.impression[original_index]).items():
                    text = anatomy
                    target_text = " ".join(description)
                    target_text = target_text
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
                    dataset[split].append(sample)
        return dataset


def prepro_chexpert():
    random.seed(42)

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

    CHEXPERT_VIEW_COL = "Frontal/Lateral"

    data_root = "data/chexpert-public/"
    chexpert_train_path = f"{data_root}/train.csv"
    chexpert_test_path = f"{data_root}/valid.csv"

    df = pd.read_csv(chexpert_train_path)
    df = df.fillna(0)
    df = df[df["Frontal/Lateral"] == "Frontal"]
    df["Path"] = df["Path"].map(lambda x: x.replace("CheXpert-v1.0-small/", ""))

    task_dfs = []
    task_dfs_20 = []
    for i, t in enumerate(CHEXPERT_COMPETITION_TASKS):
        index = np.zeros(14)
        index[i] = 1
        df_task = df[
            (df["Atelectasis"] == index[0])
            & (df["Cardiomegaly"] == index[1])
            & (df["Consolidation"] == index[2])
            & (df["Edema"] == index[3])
            & (df["Pleural Effusion"] == index[4])
            & (df["Enlarged Cardiomediastinum"] == index[5])
            & (df["Lung Lesion"] == index[7])
            & (df["Lung Opacity"] == index[8])
            & (df["Pneumonia"] == index[9])
            & (df["Pneumothorax"] == index[10])
            & (df["Pleural Other"] == index[11])
            & (df["Fracture"] == index[12])
            & (df["Support Devices"] == index[13])
            ]
        df_task = df_task.sample(n=200)
        task_dfs.append(df_task)
        task_dfs_20.append(df_task.iloc[:20])
    df_200 = pd.concat(task_dfs)
    df_20 = pd.concat(task_dfs_20)

    df = pd.read_csv(chexpert_train_path)
    test_df = pd.read_csv(chexpert_test_path)
    df["Path"] = df["Path"].map(lambda x: x.replace("CheXpert-v1.0-small/", ""))
    test_df["Path"] = test_df["Path"].map(lambda x: x.replace("CheXpert-v1.0-small/", ""))

    df = df[~df["Path"].isin(df_200["Path"])]
    valid_ids = np.random.choice(len(df), size=5000, replace=False)
    valid_df = df.iloc[valid_ids]
    train_df = df.drop(valid_ids, errors="ignore")

    train_df = train_df[train_df[CHEXPERT_VIEW_COL] == "Frontal"]
    valid_df = valid_df[valid_df[CHEXPERT_VIEW_COL] == "Frontal"]
    test_df = test_df[test_df[CHEXPERT_VIEW_COL] == "Frontal"]
    df_200 = df_200[df_200[CHEXPERT_VIEW_COL] == "Frontal"]
    df_20 = df_20[df_20[CHEXPERT_VIEW_COL] == "Frontal"]

    train_df["Path"] = train_df["Path"].map(lambda x: os.path.join(data_root, x))
    valid_df["Path"] = valid_df["Path"].map(lambda x: os.path.join(data_root, x))
    test_df["Path"] = test_df["Path"].map(lambda x: os.path.join(data_root, x))
    df_200["Path"] = df_200["Path"].map(lambda x: os.path.join(data_root, x))
    df_20["Path"] = df_20["Path"].map(lambda x: os.path.join(data_root, x))

    train_df = train_df.fillna(0)
    valid_df = valid_df.fillna(0)
    test_df = test_df.fillna(0)
    df_200 = df_200.fillna(0)
    df_20 = df_20.fillna(0)

    uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
    train_df = train_df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
    valid_df = valid_df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
    test_df = test_df.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
    df_200 = df_200.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
    df_20 = df_20.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

    train_df_001 = train_df.sample(frac=0.01)
    train_df_01 = train_df.sample(frac=0.1)

    print(f"Number of train samples: {len(train_df)}")
    print(f"Number of valid samples: {len(valid_df)}")
    print(f"Number of test samples: {len(test_df)}")
    print(f"Number of chexpert5x200 samples: {len(df_200)}")
    print(f"Number of chexpert5x20 samples: {len(df_20)}")

    train_df_001.to_csv(f"{data_root}/train_gloria_001.csv")
    train_df_01.to_csv(f"{data_root}/train_gloria_01.csv")
    train_df.to_csv(f"{data_root}/train_gloria.csv")
    valid_df.to_csv(f"{data_root}/val_gloria.csv")
    test_df.to_csv(f"{data_root}/test_gloria.csv")
    df_200.to_csv(f"{data_root}/chexpert5x200.csv")
    df_20.to_csv(f"{data_root}/chexpert5x20.csv")

    # CheXpert 5*20
    save_dir = f"{data_root}/chexpert5x20"
    image_dir = f"{save_dir}/images"
    os.makedirs(image_dir, exist_ok=True)
    simple_df_20 = []
    for idx, (row_idx, row) in enumerate(df_20.iterrows()):
        src_path = row.Path
        tgt_path = f'{image_dir}/{idx:03d}.jpg'
        shutil.copy(src_path, tgt_path)
        simple_df_20.append([f"{idx:03d}.jpg", row[5:].index[row[5:] == 1].item()])
    simple_df_20 = pd.DataFrame(simple_df_20)
    simple_df_20.to_csv(f"{save_dir}/chexpert5x20.csv", index=False)

    # CheXpert 5*20
    save_dir = f"{data_root}/chexpert200"
    image_dir = f"{save_dir}/images"
    os.makedirs(image_dir, exist_ok=True)
    simple_test_df = []
    for idx, (row_idx, row) in enumerate(test_df.iterrows()):
        src_path = row.Path
        tgt_path = f'{image_dir}/{idx:03d}.jpg'
        shutil.copy(src_path, tgt_path)
        simple_test_df.append([f"{idx:03d}.jpg", row[5:].index[row[5:] == 1].tolist()])
    simple_test_df = pd.DataFrame(simple_test_df)
    simple_test_df.to_csv(f"{save_dir}/chexpert200.csv", index=False)


def extract_chexpert_reports():
    save_dir = "data/chexpert-public/texts"
    os.makedirs(save_dir, exist_ok=True)

    path = "data/chexpert-public/df_master_77_for_release_with_dcm.csv"
    data = pd.read_csv(path)
    data = data.sort_values("patient_report_date_order")
    # step 1: extract findings
    data_findings = data.dropna(subset=["section_findings"])
    dict_findings = {
        row["original_index"]: re.sub("\s+", " ", row["section_findings"]).strip()
        for row_idx, row in data_findings.iterrows()
        if not isinstance(row["section_findings"], float) and len(row["section_findings"]) > 0
    }
    list_findings = list(set([
        re.sub("\s+", " ", row["section_findings"]).strip()
        for row_idx, row in data_findings.iterrows()
        if not isinstance(row["section_findings"], float) and len(row["section_findings"]) > 0
    ]))

    # step 2: extract impression
    data_impression = data.dropna(subset=["section_impression"])
    dict_impression = {
        row["original_index"]: re.sub("\s+", " ", row["section_impression"]).strip()
        for row_idx, row in data_impression.iterrows()
        if not isinstance(row["section_impression"], float) and len(row["section_impression"]) > 0
    }
    list_impression = list(set([
        re.sub("\s+", " ", row["section_impression"]).strip()
        for row_idx, row in data_impression.iterrows()
        if not isinstance(row["section_impression"], float) and len(row["section_impression"]) > 0
    ]))

    for filename, item in zip(
            ["dict_findings", "list_findings", "dict_impression", "list_impression"],
            [dict_findings, list_findings, dict_impression, list_impression]
    ):
        json.dump(item, open(f"{save_dir}/{filename}.json", "wt"), ensure_ascii=False, indent=2)


def format_chexpert_reports():
    save_dir = "data/chexpert-public/texts"
    os.makedirs(save_dir, exist_ok=True)

    path = "data/chexpert-public/df_master_77_for_release_with_dcm.csv"
    data = pd.read_csv(path)
    data = data.sort_values("patient_report_date_order")
    # step 1: extract findings
    fingdings_mappings = json.load(open("data/chexpert-public/texts/dict_findings_formatted.json"))
    data_findings = data.dropna(subset=["section_findings"])
    dict_findings = {
        row["original_index"]: re.sub("\s+", " ", row["section_findings"]).strip()
        for row_idx, row in data_findings.iterrows()
        if not isinstance(row["section_findings"], float) and len(row["section_findings"]) > 0
    }
    print(f"[Findings] Original Length: {len(dict_findings)}")
    dict_findings = {
        k: fingdings_mappings[v]
        for k, v in dict_findings.items() if v in fingdings_mappings
    }
    print(f"[Findings] New Length: {len(dict_findings)}")

    # step 2: extract impression
    impression_mappings_1 = json.load(open("data/chexpert-public/texts/dict_impression_formatted.json"))
    impression_mappings_2 = json.load(open("data/chexpert-public/texts/dict_impression_remaining_formatted.json"))
    impression_mappings_1.update(impression_mappings_2)
    impression_mappings = impression_mappings_1
    data_impression = data.dropna(subset=["section_impression"])
    dict_impression = {
        row["original_index"]: re.sub("\s+", " ", row["section_impression"]).strip()
        for row_idx, row in data_impression.iterrows()
        if not isinstance(row["section_impression"], float) and len(row["section_impression"]) > 0
    }
    print(f"[Impression] Original Length: {len(dict_impression)}")
    dict_impression = {
        k: impression_mappings[v]
        for k, v in dict_impression.items() if v in impression_mappings
    }
    print(f"[Impression] New Length: {len(dict_impression)}")

    for filename, item in zip(
            ["findings_clean", "impression_clean"],
            [dict_findings, dict_impression]
    ):
        json.dump(item, open(f"{save_dir}/{filename}.json", "wt"), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    # prepro_chexpert()
    # extract_chexpert_reports()
    # format_chexpert_reports()
    processor = CheXpertPublicProcessor()
    processor.create_view_classification()
    processor.create_image_classification()
    processor.create_findings_generation()
    processor.create_impression_generation()
    processor.create_local_findings_generation()
    processor.create_local_impression_generation()
    processor.create_progression_findings_generation()
    processor.create_progression_impression_generation()
    processor.create_local_progression_findings_generation()
    processor.create_local_progression_impression_generation()
    processor.create_findings_generation_with_indication()
    processor.create_impression_generation_with_indication()
