import json
import re
from collections import defaultdict

import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template
from .utils import create_dicom_to_meta_mapping, create_dicom_to_path_mapping
from .utils import create_study_to_paths_mapping, create_study_to_split_mapping
from .utils import create_study_to_texts_mapping

keywords = [
    "new", "change", "unchanged", "prior", "stable", "interval", "previous", "again", "increased", "improve",
    "remain", "worse", "persistent", "remov", "decrease", "similar", "earlier", "recurrence", "redemonstrate"
]


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


class MIMICCXRStructProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/mimic-cxr-struct"
        self.ann_path = f"{self.data_dir}/mimic_cxr_struct.json"
        self.ann_data = json.load(open(self.ann_path))
        self.study2split = create_study_to_split_mapping()
        self.study2paths = create_study_to_paths_mapping()
        self.dicom2meta = create_dicom_to_meta_mapping()
        self.dicom2path, self.path2dicom = create_dicom_to_path_mapping()
        self.study2texts = create_study_to_texts_mapping()

        self.chexpert_path = f"data/mimic-cxr/mimic-cxr-2.0.0-chexpert.csv"
        self.chexpert_data = pd.read_csv(self.chexpert_path)
        self.study2labels = {}
        for idx, row in self.chexpert_data.iterrows():
            labels = set([k for k, v in row[2:].to_dict().items() if v > 0])
            if len(labels) == 0:
                labels = set(['No Finding'])
            self.study2labels[str(int(row['study_id']))] = labels

    def create_findings_generation(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Findings Generation] [MIMIC-CXR-Struct]"}

        records = set()
        form_qa_func = create_template(dataset["dataset_name"])
        for k, v in self.ann_data["findings"].items():
            if k not in records:
                records.add(k)
            else:
                continue
            unique_id = f'[Findings Generation] [MIMIC-CXR-Struct] [{k}]'
            study_id = f'[mimic-cxr] [{int(k[1:])}]'
            data_source = "MIMIC-CXR-Struct"
            task_name = "Findings Generation"
            image_path = self.study2paths[int(k[1:])]
            text = ""
            target_text = format_report(v)

            if len(target_text) == 0:
                continue
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
            split = self.study2split[int(k[1:])]
            dataset[split].append(sample)
        return dataset

    def create_findings_generation_with_indication(self):
        dataset = {"train": [], "val": [], "test": [],
                   "dataset_name": "[Findings Generation with Indication] [MIMIC-CXR-Struct]"}
        records = set()
        form_qa_func = create_template(dataset["dataset_name"])
        for k, v in self.ann_data["findings"].items():
            if k not in records:
                records.add(k)
            else:
                continue
            unique_id = f'[Findings Generation with Indication] [MIMIC-CXR-Struct] [{k}]'
            study_id = f'[mimic-cxr] [{int(k[1:])}]'
            data_source = "MIMIC-CXR-Struct"
            task_name = "Findings Generation with Indication"
            image_path = self.study2paths[int(k[1:])]

            indication = self.study2texts.indication[k]
            if isinstance(indication, float):
                indication = "None"
            indication = re.sub("\s+", " ", indication).strip()

            text = indication
            target_text = format_report(v)
            if len(target_text) == 0:
                continue
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
            split = self.study2split[int(k[1:])]
            dataset[split].append(sample)
        return dataset

    def create_local_findings_generation(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Local Findings Generation] [MIMIC-CXR-Struct]"}
        records = set()

        anatomies = []
        form_qa_func = create_template(dataset["dataset_name"])
        for k, v in self.ann_data["findings"].items():
            if k not in records:
                records.add(k)
            else:
                continue
            unique_id = f'[Local Findings Generation] [MIMIC-CXR-Struct] [{k}]'
            study_id = f'[mimic-cxr] [{int(k[1:])}]'
            data_source = "MIMIC-CXR-Struct"
            task_name = "Local Findings Generation"
            image_path = self.study2paths[int(k[1:])]
            for anatomy, description in parse_report(v).items():
                text = anatomy
                target_text = " ".join(description)
                target_text = format_report(target_text)
                if len(target_text) == 0 or target_text[-1] not in [".", "*"]:
                    continue
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
                split = self.study2split[int(k[1:])]
                dataset[split].append(sample)
                anatomies.append(anatomy)
        return dataset

    def create_impression_generation(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Impression Generation] [MIMIC-CXR-Struct]"}

        records = set()
        form_qa_func = create_template(dataset["dataset_name"])

        for k, v in self.ann_data["impression"].items():
            if k not in records:
                records.add(k)
            else:
                continue
            unique_id = f'[Impression Generation] [MIMIC-CXR-Struct] [{k}]'
            study_id = f'[mimic-cxr] [{int(k[1:])}]'
            data_source = "MIMIC-CXR-Struct"
            task_name = "Impression Generation"
            image_path = self.study2paths[int(k[1:])]
            text = ""
            target_text = format_report(v)
            if len(target_text) == 0:
                continue
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
            split = self.study2split[int(k[1:])]
            dataset[split].append(sample)
        return dataset

    def create_impression_generation_with_indication(self):
        dataset = {"train": [], "val": [], "test": [],
                   "dataset_name": "[Impression Generation with Indication] [MIMIC-CXR-Struct]"}

        records = set()
        form_qa_func = create_template(dataset["dataset_name"])

        for k, v in self.ann_data["impression"].items():
            if k not in records:
                records.add(k)
            else:
                continue
            unique_id = f'[Impression Generation with Indication] [MIMIC-CXR-Struct] [{k}]'
            study_id = f'[mimic-cxr] [{int(k[1:])}]'
            data_source = "MIMIC-CXR-Struct"
            task_name = "Impression Generation with Indication"
            image_path = self.study2paths[int(k[1:])]

            indication = self.study2texts.indication[k]
            if isinstance(indication, float):
                indication = "None"
            indication = re.sub("\s+", " ", indication).strip()

            text = indication
            target_text = format_report(v)
            if len(target_text) == 0:
                continue
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
            split = self.study2split[int(k[1:])]
            dataset[split].append(sample)
        return dataset

    def create_local_impression_generation(self):
        dataset = {"train": [], "val": [], "test": [],
                   "dataset_name": "[Local Impression Generation] [MIMIC-CXR-Struct]"}
        records = set()

        form_qa_func = create_template(dataset["dataset_name"])
        for k, v in self.ann_data["impression"].items():
            if k not in records:
                records.add(k)
            else:
                continue
            unique_id = f'[Local Impression Generation] [MIMIC-CXR-Struct] [{k}]'
            study_id = f'[mimic-cxr] [{int(k[1:])}]'
            data_source = "MIMIC-CXR-Struct"
            task_name = "Local Impression Generation"
            image_path = self.study2paths[int(k[1:])]
            for anatomy, description in parse_report(v).items():
                text = anatomy
                target_text = " ".join(description)
                target_text = format_report(target_text)
                if len(target_text) == 0 or target_text[-1] not in [".", "*"]:
                    continue
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
                split = self.study2split[int(k[1:])]
                dataset[split].append(sample)
        return dataset

    def create_progression_findings_generation(self):
        self.dicom2meta['cxrtime'] = pd.to_datetime(self.dicom2meta['cxrtime'])
        dicom2meta = self.dicom2meta.sort_values(by='cxrtime')
        dicom2meta.study_id = dicom2meta.study_id.astype(int)
        prev_dict = {}
        for subject_id, subject in dicom2meta.groupby("subject_id"):
            study_ids = subject.study_id.unique().tolist()
            assert study_ids[0] == subject.iloc[0].study_id
            for idx in range(1, len(study_ids)):
                prev_dict[study_ids[idx]] = study_ids[idx - 1]

        dataset = {
            "train": [], "val": [], "test": [], "dataset_name": "[Progression Findings Generation] [MIMIC-CXR-Struct]"
        }
        form_qa_func = create_template(dataset["dataset_name"])
        study2findings = self.study2texts.findings.dropna()
        study2findings = study2findings[study2findings.map(lambda x: len(x) != 0)]
        for study_id, study in study2findings.items():
            unique_id = f'[Progression Findings Generation] [MIMIC-CXR-Struct] [{study_id}]'
            data_source = "MIMIC-CXR-Struct"
            task_name = "Progression Findings Generation"
            if study_id not in self.ann_data["findings"]:
                continue
            if not any([keyword in self.ann_data["findings"][study_id].lower() for keyword in keywords]):
                continue
            if int(study_id[1:]) not in prev_dict:
                continue
            image_path = [self.study2paths[prev_dict[int(study_id[1:])]][0], self.study2paths[int(study_id[1:])][0]]
            text = ""
            target_text = self.ann_data["findings"][study_id]
            qa_pair = form_qa_func(self.instruct)(target_text)
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

    def create_progression_impression_generation(self):
        self.dicom2meta['cxrtime'] = pd.to_datetime(self.dicom2meta['cxrtime'])
        dicom2meta = self.dicom2meta.sort_values(by='cxrtime')
        dicom2meta.study_id = dicom2meta.study_id.astype(int)
        prev_dict = {}
        for subject_id, subject in dicom2meta.groupby("subject_id"):
            study_ids = subject.study_id.unique().tolist()
            for idx in range(1, len(study_ids)):
                prev_dict[study_ids[idx]] = study_ids[idx - 1]

        dataset = {
            "train": [], "val": [], "test": [],
            "dataset_name": "[Progression Impression Generation] [MIMIC-CXR-Struct]"
        }
        form_qa_func = create_template(dataset["dataset_name"])
        study2impression = self.study2texts.impression.dropna()
        study2impression = study2impression[study2impression.map(lambda x: len(x) != 0)]
        for study_id, study in study2impression.items():
            unique_id = f'[Progression Impression Generation] [MIMIC-CXR-Struct] [{study_id}]'
            data_source = "MIMIC-CXR-Struct"
            task_name = "Progression Impression Generation"
            if study_id not in self.ann_data["impression"]:
                continue
            if not any([keyword in self.ann_data["impression"][study_id].lower() for keyword in keywords]):
                continue
            if int(study_id[1:]) not in prev_dict:
                continue
            image_path = [self.study2paths[prev_dict[int(study_id[1:])]][0], self.study2paths[int(study_id[1:])][0]]
            text = ""
            target_text = self.ann_data["impression"][study_id]
            qa_pair = form_qa_func(self.instruct)(target_text)
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

    def create_local_progression_findings_generation(self):
        self.dicom2meta['cxrtime'] = pd.to_datetime(self.dicom2meta['cxrtime'])
        dicom2meta = self.dicom2meta.sort_values(by='cxrtime')
        dicom2meta.study_id = dicom2meta.study_id.astype(int)
        prev_dict = {}
        for subject_id, subject in dicom2meta.groupby("subject_id"):
            study_ids = subject.study_id.unique().tolist()
            for idx in range(1, len(study_ids)):
                prev_dict[study_ids[idx]] = study_ids[idx - 1]
        study2findings = self.study2texts.findings.dropna()
        study2findings = study2findings[study2findings.map(lambda x: len(x) != 0)]

        dataset = {
            "train": [], "val": [], "test": [],
            "dataset_name": "[Local Progression Findings Generation] [MIMIC-CXR-Struct]"
        }
        form_qa_func = create_template(dataset["dataset_name"])
        for study_id, study in study2findings.items():
            unique_id = f'[Local Progression Findings Generation] [MIMIC-CXR-Struct] [{study_id}]'
            data_source = "MIMIC-CXR-Struct"
            task_name = "Local Progression Findings Generation"
            if study_id not in self.ann_data["findings"]:
                continue
            if int(study_id[1:]) not in prev_dict:
                continue
            image_path = [self.study2paths[prev_dict[int(study_id[1:])]][0], self.study2paths[int(study_id[1:])][0]]
            for anatomy, description in parse_report(self.ann_data["findings"][study_id]).items():
                text = anatomy
                target_text = " ".join(description)
                if len(target_text) == 0 or target_text[-1] not in [".", "*"]:
                    continue
                if not any([keyword in target_text.lower() for keyword in keywords]):
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

    def create_local_progression_impression_generation(self):
        self.dicom2meta['cxrtime'] = pd.to_datetime(self.dicom2meta['cxrtime'])
        dicom2meta = self.dicom2meta.sort_values(by='cxrtime')
        dicom2meta.study_id = dicom2meta.study_id.astype(int)
        prev_dict = {}
        for subject_id, subject in dicom2meta.groupby("subject_id"):
            study_ids = subject.study_id.unique().tolist()
            for idx in range(1, len(study_ids)):
                prev_dict[study_ids[idx]] = study_ids[idx - 1]
        study2impression = self.study2texts.impression.dropna()
        study2impression = study2impression[study2impression.map(lambda x: len(x) != 0)]

        dataset = {
            "train": [], "val": [], "test": [],
            "dataset_name": "[Local Progression Impression Generation] [MIMIC-CXR-Struct]"
        }
        form_qa_func = create_template(dataset["dataset_name"])
        for study_id, study in study2impression.items():
            unique_id = f'[Local Progression Impression Generation] [MIMIC-CXR-Struct] [{study_id}]'
            data_source = "MIMIC-CXR-Struct"
            task_name = "Local Progression Impression Generation"
            if study_id not in self.ann_data["impression"]:
                continue
            if int(study_id[1:]) not in prev_dict:
                continue
            image_path = [self.study2paths[prev_dict[int(study_id[1:])]][0], self.study2paths[int(study_id[1:])][0]]
            for anatomy, description in parse_report(self.ann_data["impression"][study_id]).items():
                text = anatomy
                target_text = " ".join(description)
                if len(target_text) == 0 or target_text[-1] not in [".", "*"]:
                    continue
                if not any([keyword in target_text.lower() for keyword in keywords]):
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


if __name__ == '__main__':
    processor = MIMICCXRStructProcessor()
    processor.create_findings_generation()
    processor.create_local_findings_generation()
    processor.create_impression_generation()
    processor.create_local_impression_generation()
    processor.create_progression_findings_generation()
    processor.create_progression_impression_generation()
    processor.create_local_progression_findings_generation()
    processor.create_local_progression_impression_generation()
    processor.create_findings_generation_with_indication()
    processor.create_impression_generation_with_indication()
