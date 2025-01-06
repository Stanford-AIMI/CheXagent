from collections import defaultdict

import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template


class ReXValProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/rexval/"
        self.report_path = f"{self.data_dir}/50_samples_gt_and_candidates.csv"
        self.ann_path = f"{self.data_dir}/6_valid_raters_per_rater_error_categories.csv"
        self.report_data = pd.read_csv(self.report_path)
        self.ann_data = pd.read_csv(self.ann_path)

    def create_report_evaluation(self):
        categories = {
            1: "False prediction of finding",
            2: "Omission of finding",
            3: "Incorrect location/position of finding",
            4: "Incorrect severity of finding",
            5: "Mention of comparison that is not present in the reference impression",
            6: "Omission of comparison describing a change from a previous study"
        }

        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Report Evaluation] [ReXVal]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for row_idx, row in self.report_data.iterrows():
            sample_sig = {cand_type: defaultdict(list) for cand_type in self.report_data.columns.tolist()[2:]}
            sample_insig = {cand_type: defaultdict(list) for cand_type in self.report_data.columns.tolist()[2:]}
            for ann_idx, ann in self.ann_data[self.ann_data.study_number == row_idx].iterrows():
                if ann.clinically_significant:
                    sample_sig[ann.candidate_type][categories[ann.error_category]].append(ann.num_errors)
                else:
                    sample_insig[ann.candidate_type][categories[ann.error_category]].append(ann.num_errors)

            for k in sample_sig.keys():
                unique_id = f'[Report Evaluation] [ReXVal] [{k}-{row_idx}]'
                study_id = f'[rexval] [{row_idx}]'
                data_source = "ReXVal"
                task_name = "Report Evaluation"
                image_path = None
                text = [row["gt_report"], row[k]]
                target_text = ""

                # major vote
                options = {
                    "clinically_significant": {k: max(set(v), key=v.count) for k, v in sample_sig[k].items()},
                    "clinically_insignificant": {k: max(set(v), key=v.count) for k, v in sample_insig[k].items()}
                }

                qa_pair = form_qa_func(self.instruct)(text, options["clinically_significant"])
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
                dataset["test"].append(sample)
        return dataset


if __name__ == '__main__':
    processor = ReXValProcessor()
    processor.create_report_evaluation()
