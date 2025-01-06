import os

from .base_processor import BaseProcessor
from .templates import create_template


class MIMICIIIProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/mimic-iii"

    def create_findings_summarization(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Findings Summarization] [MIMIC-III]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for modality_anatomy in os.listdir(self.data_dir):
            if not os.path.isdir(os.path.join(self.data_dir, modality_anatomy)):
                continue
            files = os.listdir(os.path.join(self.data_dir, modality_anatomy))
            splits = [split for split in ["train", "validate", "test"] if f'{split}.findings.tok' in files]
            for split in splits:
                for idx, (findings, impression) in enumerate(zip(
                        open(os.path.join(self.data_dir, modality_anatomy, f"{split}.findings.tok")),
                        open(os.path.join(self.data_dir, modality_anatomy, f"{split}.impression.tok"))
                )):
                    unique_id = f'[Findings Summarization] [MIMIC-III] [{modality_anatomy}-{split}-{idx}]'
                    study_id = f'[mimic-iii] [{split}-{int(idx)}]'
                    data_source = "MIMIC-III"
                    task_name = "Findings Summarization"
                    image_path = None
                    text = findings
                    target_text = impression
                    if any([ignore in target_text for ignore in ["___", "a.m.", "p.m."]]):
                        continue
                    qa_pair = form_qa_func(self.instruct)(text, target_text)
                    modality, anatomy = modality_anatomy.split("_")
                    sample = {
                        'unique_id': unique_id,
                        'study_id': study_id,
                        'data_source': data_source,
                        'task_name': task_name,
                        'image_path': image_path,
                        'modality': modality,
                        'anatomy': anatomy,
                        "text": text,
                        'target_text': target_text,
                        "qa_pair": qa_pair,
                    }
                    dataset[split.replace("validate", "val")].append(sample)
        return dataset


if __name__ == '__main__':
    processor = MIMICIIIProcessor()
    processor.create_findings_summarization()
