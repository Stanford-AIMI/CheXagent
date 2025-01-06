import json
import os

from .base_processor import BaseProcessor
from .templates import create_template


class RadRestructProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/rad-restruct/data/radrestruct/"
        self.image_dir = f"{self.data_dir}/images"
        self.train_path = f'{self.data_dir}/train_qa_pairs'
        self.val_path = f'{self.data_dir}/val_qa_pairs'
        self.test_path = f'{self.data_dir}/test_qa_pairs'
        self.id2image_path = f"{self.data_dir}/id_to_img_mapping_frontal_reports.json"
        self.id2image = json.load(open(self.id2image_path))

        self.train_data = [(self.id2image[file[:-5]], json.load(open(f'{self.train_path}/{file}')))
                           for file in os.listdir(self.train_path)]
        self.val_data = [(self.id2image[file[:-5]], json.load(open(f'{self.val_path}/{file}')))
                         for file in os.listdir(self.val_path)]
        self.test_data = [(self.id2image[file[:-5]], json.load(open(f'{self.test_path}/{file}')))
                          for file in os.listdir(self.test_path)]

    def create_open_ended_vqa(self):
        dataset = {"dataset_name": "[Open-Ended VQA] [Rad-Restruct]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.test_data, self.test_data]):
            samples = []
            for pairs_idx, pairs in enumerate(split_data):
                for pair_idx, pair in enumerate(pairs[1]):
                    unique_id = f'[Open-Ended VQA] [Rad-Restruct] [{pairs_idx}-{pair_idx}]'
                    study_id = f'[openi] [{pairs[0][0].split("_")[0].replace("CXR", "")}]'
                    data_source = "Rad-Restruct"
                    task_name = "Open-Ended VQA"
                    image_paths = [f'{self.image_dir}/{image_name}.png' for image_name in pairs[0]]
                    text = "\n".join(f'Q: {qa[0]} A: {", ".join(qa[1])}' for qa in pair[2]) + \
                           ("\n" if len(pair[2]) > 0 else "") + "Q: " + pair[0]
                    target_text = ", ".join(pair[1]).capitalize()
                    qa_pair = form_qa_func(self.instruct)(text, target_text)
                    sample = {
                        'unique_id': unique_id,
                        'study_id': study_id,
                        'data_source': data_source,
                        'task_name': task_name,
                        'image_path': image_paths,
                        "text": text,
                        'target_text': target_text,
                        "qa_pair": qa_pair
                    }
                    samples.append(sample)
            dataset[split] = samples
        return dataset

    def create_close_ended_vqa(self):
        dataset = {"dataset_name": "[Close-Ended VQA] [Rad-Restruct]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val", "test"], [self.train_data, self.test_data, self.test_data]):
            samples = []
            for pairs_idx, pairs in enumerate(split_data):
                for pair_idx, pair in enumerate(pairs[1]):
                    unique_id = f'[Close-Ended VQA] [Rad-Restruct] [{pairs_idx}-{pair_idx}]'
                    study_id = f'[openi] [{pairs[0][0].split("_")[0].replace("CXR", "")}]'
                    data_source = "Rad-Restruct"
                    task_name = "Close-Ended VQA"
                    image_paths = [f'{self.image_dir}/{image_name}.png' for image_name in pairs[0]]
                    text = "; ".join(f'Q: {qa[0]} A: {", ".join(qa[1])}' for qa in pair[2]) + \
                           ("\n" if len(pair[2]) > 0 else "") + "Q: " + pair[0]
                    target_text = ""
                    options = {option: option in pair[1] for option in pair[3]["options"]}
                    qa_pair = form_qa_func(self.instruct)(text, options)

                    sample = {
                        'unique_id': unique_id,
                        'study_id': study_id,
                        'data_source': data_source,
                        'task_name': task_name,
                        'image_path': image_paths,
                        "text": text,
                        'target_text': target_text,
                        "options": options,
                        "qa_pair": qa_pair
                    }
                    samples.append(sample)
            dataset[split] = samples
        return dataset


if __name__ == '__main__':
    processor = RadRestructProcessor()
    processor.create_open_ended_vqa()
    processor.create_close_ended_vqa()
