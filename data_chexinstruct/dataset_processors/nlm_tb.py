from .base_processor import BaseProcessor
from .templates import create_template


class NLMTBProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/nlm-tb/"
        self.image_dir = f"{self.data_dir}/images"
        self.images = open(f'{self.data_dir}/image.tok').read().strip().split("\n")
        self.labels = open(f'{self.data_dir}/label.tok').read().strip().split("\n")
        self.labels = [{"Tuberculosis": int(label)} for label in self.labels]

    def create_image_classification(self):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Image Classification] [NLM-TB]"}
        split = "train"

        form_qa_func = create_template(dataset["dataset_name"])
        for idx, (image, label) in enumerate(zip(self.images, self.labels)):
            unique_id = f'[Image Classification] [NLMTB] [{idx}]'
            study_id = f'[nlm-tb] [{split}-{idx}]'
            data_source = "NLM-TB"
            task_name = "Image Classification"
            image_path = f"{self.image_dir}/{image}"
            text = ""
            target_text = ""
            options = label
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


if __name__ == '__main__':
    processor = NLMTBProcessor()
    processor.create_image_classification()
