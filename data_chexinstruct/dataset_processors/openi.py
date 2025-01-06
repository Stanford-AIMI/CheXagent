import os
import xml.etree.ElementTree as etree

from .base_processor import BaseProcessor
from .templates import create_template


class OpenIProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/openi"
        self.image_dir = f'{self.data_dir}/images'
        self.report_dir = f'{self.data_dir}/ecgen-radiology'
        self.uid2findings, self.uid2impression, self.uid2report = self.preprocess()

    def create_findings_generation(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Findings Generation] [OpenI]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for study_id, study in self.uid2findings.items():
            unique_id = f'[Findings Generation] [OpenI] [{study_id}]'
            study_id = f'[openi] [{study[0][0].split("/")[-1].split("_")[0].replace("CXR", "")}]'
            data_source = "OpenI"
            task_name = "Findings Generation"
            image_path = study[0]
            text = ""
            target_text = study[1]
            qa_pair = form_qa_func(self.instruct)(target_text)
            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'data_source': data_source,
                'task_name': task_name,
                'image_path': image_path,
                "text": text,
                'target_text': target_text,
                "qa_pair": qa_pair,
            }
            dataset["test"].append(sample)
        return dataset

    def create_impression_generation(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Impression Generation] [OpenI]"}
        form_qa_func = create_template(dataset["dataset_name"])

        for study_id, study in self.uid2impression.items():
            unique_id = f'[Impression Generation] [OpenI] [{study_id}]'
            study_id = f'[openi] [{study[0][0].split("/")[-1].split("_")[0].replace("CXR", "")}]'
            data_source = "OpenI"
            task_name = "Impression Generation"
            image_path = study[0]
            text = ""
            target_text = study[1]
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
            dataset["test"].append(sample)
        return dataset

    def create_findings_summarization(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Findings Summarization] [OpenI]"}
        form_qa_func = create_template(dataset["dataset_name"])

        for study_id, study in self.uid2report.items():
            unique_id = f'[Findings Summarization] [OpenI] [{study_id}]'
            study_id = f'[openi] [{study[0][0].split("/")[-1].split("_")[0].replace("CXR", "")}]'
            data_source = "OpenI"
            task_name = "Findings Summarization"
            image_path = None
            text = study[0]
            target_text = study[1]
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
            dataset["test"].append(sample)
        return dataset

    def preprocess(self):
        uid2findings = {}
        uid2impression = {}
        uid2report = {}
        total = 3999
        for i in range(1, total + 1):
            path = os.path.join(self.report_dir, f'{i}.xml')
            if os.path.exists(path):
                tree = etree.parse(path)
                rt = tree.getroot()
                image_ids = []
                uid = None
                for ele in rt.findall(".//uId"):
                    uid = ele.attrib['id']
                for ele in rt.findall(".//parentImage"):
                    image_ids.append(ele.attrib['id'])
                findings = ''
                for ele in rt.findall(".//AbstractText[@Label='FINDINGS']"):
                    findings = ele.text
                    findings = findings.strip() if findings is not None else ''
                impression = ''
                for ele in rt.findall(".//AbstractText[@Label='IMPRESSION']"):
                    impression = ele.text
                    impression = impression.strip() if impression is not None else ''

                if len(findings) > 0 and len(image_ids) > 0:
                    images = []
                    for image_id in image_ids:
                        images.append(os.path.join(self.image_dir, f'{image_id}.png'))
                    images = [image for image in images if os.path.exists(image)]
                    if images:
                        uid2findings[uid] = (images, findings)

                if len(impression) > 0 and len(image_ids) > 0:
                    images = []
                    for image_id in image_ids:
                        images.append(os.path.join(self.image_dir, f'{image_id}.png'))
                    images = [image for image in images if os.path.exists(image)]
                    if images:
                        uid2impression[uid] = (images, impression)

                if len(findings) > 0 and len(impression) > 0:
                    uid2report[uid] = (findings, impression)
        return uid2findings, uid2impression, uid2report


if __name__ == '__main__':
    processor = OpenIProcessor()
    processor.create_findings_generation()
    processor.create_impression_generation()
    processor.create_findings_summarization()
