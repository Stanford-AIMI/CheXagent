import pandas as pd

from .base_processor import BaseProcessor
from .templates import create_template
from .utils import create_dicom_to_path_mapping, create_dicom_to_meta_mapping, create_study_to_texts_mapping
from .utils import create_study_to_split_mapping


def cvt(coord):
    coord = min(coord, 999)
    coord = round(coord / 10)
    return str(coord)


class MSCXRProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        data_dir = "data/ms-cxr/"
        ann_path = f"{data_dir}/MS_CXR_Local_Alignment_v1.0.0.csv"
        self.ann_data = pd.read_csv(ann_path)
        self.dicom2path, self.path2dicom = create_dicom_to_path_mapping()
        self.dicom2meta = create_dicom_to_meta_mapping()
        self.study2texts = create_study_to_texts_mapping()
        self.study2split = create_study_to_split_mapping()

    def create_grounded_captioning(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Grounded Captioning] [MS-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for group_idx, group in enumerate(self.ann_data.groupby(["dicom_id", "label_text"])):
            df = group[-1]
            unique_id = f"[Grounded Captioning] [MS-CXR] [{group_idx}]"
            study_id = f'[mimic-cxr] [{int(df.iloc[0].path.split("/")[3][1:])}]'
            data_source = "MS-CXR"
            task_name = "Grounded Captioning"
            image_path = self.dicom2path[df.dicom_id.iloc[0]]
            image_path = image_path.replace("mimic-cxr", "ms-cxr")

            # tgt_path = image_path.replace("mimic-cxr", "ms-cxr")
            # os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            # shutil.copy(image_path, tgt_path)

            ratio = min(df.iloc[0].image_width, df.iloc[0].image_height) / 512
            df.iloc[:, 4:] = (df.iloc[:, 4:] // ratio).astype(int)

            regions = [[row.x, row.y, row.x + row.w, row.y + row.h] for row_idx, row in df.iterrows()]

            quantized_boxes = [[
                int(row.x / row.image_width * qbins), int(row.y / row.image_height * qbins),
                int((row.x + row.w) / row.image_width * qbins), int((row.y + row.h) / row.image_height * qbins)]
                for row_idx, row in df.iterrows()
            ]
            quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
            text = boxes = "".join([
                f"<|box|>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})<|/box|>" for b in quantized_boxes
            ])
            target_text = df.iloc[0].label_text
            qa_pair = form_qa_func(self.instruct)(boxes, target_text)

            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'data_source': data_source,
                'task_name': task_name,
                'image_path': image_path,
                'region': regions,
                'text': text,
                'target_text': target_text,
                'qa_pair': qa_pair
            }
            dataset[self.study2split[int(df.iloc[0].path.split("/")[3][1:])]].append(sample)
        return dataset

    def create_grounded_diagnosis(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Grounded Diagnosis] [MS-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for group_idx, group in enumerate(self.ann_data.groupby(["dicom_id", "label_text"])):
            df = group[-1]
            unique_id = f"[Grounded Diagnosis] [MS-CXR] [{group_idx}]"
            study_id = f'[mimic-cxr] [{int(df.iloc[0].path.split("/")[3][1:])}]'
            data_source = "MS-CXR"
            task_name = "Grounded Diagnosis"
            image_path = self.dicom2path[df.dicom_id.iloc[0]]
            image_path = image_path.replace("mimic-cxr", "ms-cxr")

            # tgt_path = image_path.replace("mimic-cxr", "ms-cxr")
            # os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            # shutil.copy(image_path, tgt_path)

            ratio = min(df.iloc[0].image_width, df.iloc[0].image_height) / 512
            df.iloc[:, 4:] = (df.iloc[:, 4:] // ratio).astype(int)

            regions = [[row.x, row.y, row.x + row.w, row.y + row.h] for row_idx, row in df.iterrows()]

            quantized_boxes = [[
                int(row.x / row.image_width * qbins), int(row.y / row.image_height * qbins),
                int((row.x + row.w) / row.image_width * qbins), int((row.y + row.h) / row.image_height * qbins)]
                for row_idx, row in df.iterrows()
            ]
            quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
            text = boxes = "".join([
                f"<|box|>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})<|/box|>" for b in quantized_boxes
            ])
            target_txt = df.iloc[0].category_name
            qa_pair = form_qa_func(self.instruct)(boxes, target_txt)

            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'data_source': data_source,
                'task_name': task_name,
                'image_path': image_path,
                'region': regions,
                'text': text,
                'target_text': target_txt,
                'qa_pair': qa_pair,
            }
            dataset[self.study2split[int(df.iloc[0].path.split("/")[3][1:])]].append(sample)
        return dataset

    def create_grounded_phrase_extraction(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Grounded Phrase Extraction] [MS-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for group_idx, group in enumerate(self.ann_data.groupby(["dicom_id", "label_text"])):
            df = group[-1]
            unique_id = f"[Grounded Phrase Extraction] [MS-CXR] [{group_idx}]"
            study_id = f'[mimic-cxr] [{int(df.iloc[0].path.split("/")[3][1:])}]'
            data_source = "MS-CXR"
            task_name = "Grounded Phrase Extraction"
            image_path = self.dicom2path[df.dicom_id.iloc[0]]
            image_path = image_path.replace("mimic-cxr", "ms-cxr")

            # tgt_path = image_path.replace("mimic-cxr", "ms-cxr")
            # os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            # shutil.copy(image_path, tgt_path)

            ratio = min(df.iloc[0].image_width, df.iloc[0].image_height) / 512
            df.iloc[:, 4:] = (df.iloc[:, 4:] // ratio).astype(int)
            regions = [[row.x, row.y, row.x + row.w, row.y + row.h] for row_idx, row in df.iterrows()]
            target_text = df.iloc[0].label_text
            texts = self.study2texts.loc[f's{int(self.dicom2meta.loc[df.dicom_id.iloc[0]].study_id)}']
            findings, impression = texts.findings, texts.impression
            last_paragraph, comparison = texts.last_paragraph, texts.comparison
            if not isinstance(findings, float) and target_text in findings:
                text = findings
            elif not isinstance(impression, float) and target_text in impression:
                text = impression
            elif not isinstance(last_paragraph, float) and target_text in last_paragraph:
                text = last_paragraph
            elif not isinstance(comparison, float) and target_text in comparison:
                text = comparison
            else:
                continue

            quantized_boxes = [[
                int(row.x / row.image_width * qbins), int(row.y / row.image_height * qbins),
                int((row.x + row.w) / row.image_width * qbins), int((row.y + row.h) / row.image_height * qbins)]
                for row_idx, row in df.iterrows()
            ]
            quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
            boxes = "".join([
                f"<|box|>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})<|/box|>" for b in quantized_boxes
            ])
            qa_pair = form_qa_func(self.instruct)(boxes, text, target_text)

            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'data_source': data_source,
                'task_name': task_name,
                'image_path': image_path,
                'region': regions,
                "text": text,
                'target_text': target_text,
                'qa_pair': qa_pair,
            }
            dataset[self.study2split[int(df.iloc[0].path.split("/")[3][1:])]].append(sample)
        return dataset

    def create_phrase_grounding(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Phrase Grounding] [MS-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for group_idx, group in enumerate(self.ann_data.groupby(["dicom_id", "label_text"])):
            df = group[-1]
            unique_id = f"[Phrase Grounding] [MS-CXR] [{group_idx}]"
            study_id = f'[mimic-cxr] [{int(df.iloc[0].path.split("/")[3][1:])}]'
            data_source = "MS-CXR"
            task_name = "Phrase Grounding"
            image_path = self.dicom2path[df.dicom_id.iloc[0]]
            image_path = image_path.replace("mimic-cxr", "ms-cxr")

            # tgt_path = image_path.replace("mimic-cxr", "ms-cxr")
            # os.makedirs(os.path.dirname(tgt_path), exist_ok=True)
            # shutil.copy(image_path, tgt_path)

            ratio = min(df.iloc[0].image_width, df.iloc[0].image_height) / 512
            df.iloc[:, 4:] = (df.iloc[:, 4:] // ratio).astype(int)

            regions = [[row.x, row.y, row.x + row.w, row.y + row.h] for row_idx, row in df.iterrows()]

            text = df.iloc[0].label_text
            quantized_boxes = [[
                int(row.x / row.image_width * qbins), int(row.y / row.image_height * qbins),
                int((row.x + row.w) / row.image_width * qbins), int((row.y + row.h) / row.image_height * qbins)]
                for row_idx, row in df.iterrows()
            ]
            quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
            target_text = boxes = "".join([
                f"<|ref|>{text}<|/ref|><|box|>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})<|/box|>"
                for b in quantized_boxes
            ])

            qa_pair = form_qa_func(self.instruct)(text, boxes)

            sample = {
                'unique_id': unique_id,
                'study_id': study_id,
                'data_source': data_source,
                'task_name': task_name,
                'image_path': image_path,
                'region': regions,
                'text': text,
                'target_text': target_text,
                'qa_pair': qa_pair,
            }
            dataset[self.study2split[int(df.iloc[0].path.split("/")[3][1:])]].append(sample)
        return dataset


if __name__ == '__main__':
    processor = MSCXRProcessor()
    processor.create_grounded_captioning()
    processor.create_grounded_diagnosis()
    processor.create_grounded_phrase_extraction()
    processor.create_phrase_grounding()
