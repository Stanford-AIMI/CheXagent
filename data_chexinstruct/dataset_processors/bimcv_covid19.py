from collections import defaultdict

from .base_processor import BaseProcessor
from .templates import create_template


class BIMCVCOVID19Processor(BaseProcessor):
    def __init__(self):
        super().__init__()
        reader = lambda x: open(x).read().strip().split("\n")
        self.data_dir = "data/bimcv-covid19/"
        self.image_dir = f'{self.data_dir}/images'
        self.rrg_dir = f'{self.data_dir}/RRG'
        self.all_reports = defaultdict(dict)
        reports = reader(f"{self.rrg_dir}/bimcv-covid19/test.report.en.tok")
        images = reader(f"{self.rrg_dir}/bimcv-covid19/test.image.en.tok")
        for i, r in zip(images, reports):
            self.all_reports["train"][i] = r

    def create_impression_generation(self):
        format_report = lambda x: ". ".join([sent.capitalize() for sent in x.split(" . ")]).replace(" .", ".")
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Impression Generation] [BIMCV-COVID19]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split in ["train"]:
            for idx, (images, reports) in enumerate(self.all_reports[split].items()):
                study_id = "_".join(images.split(",")[0].split("/")[-3:-1])
                unique_id = f'[Report Generation] [BIMCV-COVID19] [{idx}]'
                study_id = f'[BIMCV-COVID19] [{study_id}]'
                data_source = "BIMCV-COVID19"
                task_name = "Report Generation"
                image_path = [f"{self.image_dir}/{path}" for path in images.split(",")]
                text = ""
                target_text = format_report(reports)
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


if __name__ == '__main__':
    processor = BIMCVCOVID19Processor()
    processor.create_impression_generation()
