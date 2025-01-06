import os.path
import random

import cv2
import numpy as np
import pandas as pd
import torch
from torchvision.ops import masks_to_boxes

from .base_processor import BaseProcessor
from .templates import create_template


def cvt(coord):
    coord = min(coord, 999)
    coord = round(coord / 10)
    return str(coord)


class CandidProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/candid"
        self.image_dir = f'{self.data_dir}/candid-ptx-images-512'
        self.chest_tubes = pd.read_csv(f'{self.data_dir}/chest_tube.csv')
        self.reports = pd.read_csv(f'{self.data_dir}/Pneumothorax_reports.csv')
        self.rib_fractures = pd.read_csv(f'{self.data_dir}/Rib_fracture_mask_rle.csv')
        self.images = [line.strip() for line in open(f'{self.data_dir}/RRG/candid-ptx/image.tok')]
        self.impressions = [line.strip() for line in open(f'{self.data_dir}/RRG/candid-ptx/impression.tok')]
        self.image2study = {row["SOPInstanceUID"]: row["StudyInstanceUID"] for row_id, row in self.reports.iterrows()}
        assert len(self.images) == len(self.impressions)

    def create_impression_generation(self):
        format_report = lambda x: ". ".join([sent.capitalize() for sent in x.split(" . ")]).replace(" .", ".")
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Impression Generation] [Candid-PTX]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for pair_idx, (images, impression) in enumerate(zip(self.images, self.impressions)):
            unique_id = f'[Impression Generation] [Candid-PTX] [{pair_idx}]'
            study_id = f'[Candid-PTX] [{self.image2study[images.split(",")[0].split("/")[-1].replace(".jpg", "")]}]'
            data_source = "Candid"
            task_name = "Impression Generation"
            image_path = [f'{self.data_dir}/{image}' for image in images.split(",")]
            text = ""
            target_text = format_report(impression)
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
            dataset["train"].append(sample)
        return dataset

    def create_pneumothorax_segmentation(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Pneumothorax Segmentation] [Candid-PTX]"}

        form_qa_func = create_template(dataset["dataset_name"])
        neg_count = 0
        for split, split_data in zip(["train"], [self.reports]):
            for group_idx, group in split_data.groupby("SOPInstanceUID"):
                unique_id = f"[Pneumothorax Segmentation] [Candid-PTX] [{group_idx}]"
                study_id = f'[Candid-PTX] [{group.StudyInstanceUID.iloc[0]}]'
                image_source = "Candid"
                task_name = "Pneumothorax Segmentation"
                image_path = f'{self.image_dir}/{group_idx}.jpg'

                if "-1" in group["EncodedPixels"].tolist():
                    regions = []
                    text = "pneumothorax"
                    target_text = "No pneumothorax detected."
                    neg_count += 1
                    if neg_count >= 5000:
                        continue
                else:
                    regions = [
                        ("pneumothorax", mask2polygen(rle2mask(rle, 1024, 1024, 1, 512, 512)))
                        for rle in group["EncodedPixels"].tolist()
                    ]
                    boxes = masks_to_boxes(torch.Tensor(np.array([
                        rle2mask(rle, 1024, 1024, 1, 512, 512)
                        for rle in group["EncodedPixels"].tolist()
                    ])))
                    quantized_boxes = (boxes / 512 * qbins).int().tolist()
                    quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
                    text = "pneumothorax"
                    target_text = "".join([
                        f"<|ref|>pneumothorax<|/ref|><|box|>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})<|/box|>"
                        for b in quantized_boxes
                    ])
                qa_pair = form_qa_func(self.instruct)(text, target_text)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'image_source': image_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'region': regions,
                    'text': text,
                    'target_text': target_text,
                    'qa_pair': qa_pair,
                }
                dataset[split].append(sample)
        return dataset

    def create_rib_fracture_segmentation(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Rib Fracture Segmentation] [Candid-PTX]"}

        neg_images = [image for images, impression in zip(self.images, self.impressions)
                      for image in images.split(",") if "rib" not in impression.lower()]
        pos_images = set(self.rib_fractures.anon_SOPUID.tolist())
        neg_images = [image for image in neg_images if image.split("/")[1][:-4] not in pos_images]

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train"], [self.rib_fractures]):
            for group_idx, group in split_data.groupby("anon_SOPUID"):
                unique_id = f"[Rib Fracture Segmentation] [Candid-PTX] [{group_idx}]"
                study_id = f'[Candid-PTX] [{self.image2study[group_idx]}]'
                image_source = "Candid"
                task_name = "Rib Fracture Segmentation"
                image_path = f'{self.image_dir}/{group_idx}.jpg'
                regions = [
                    ("Rib Fracture", mask2polygen(rle2mask(rle, 1024, 1024, 1, 512, 512)))
                    for rle in group.mask_rle.tolist()
                ]

                boxes = masks_to_boxes(torch.Tensor(np.array([
                    rle2mask(rle, 1024, 1024, 1, 512, 512)
                    for rle in group.mask_rle.tolist()
                ])))
                quantized_boxes = (boxes / 512 * qbins).int().tolist()
                quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
                text = "rib fracture"
                target_text = "".join([
                    f"<|ref|>rib fracture<|/ref|><|box|>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})<|/box|>" for
                    b in quantized_boxes
                ])

                qa_pair = form_qa_func(self.instruct)(text, target_text)

                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'image_source': image_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'region': regions,
                    'text': text,
                    'target_text': target_text,
                    'qa_pair': qa_pair
                }
                dataset[split].append(sample)

                # Negative Samples
                unique_id = f"[Rib Fracture Segmentation] [Candid-PTX] [{group_idx} - negative]"
                study_id = f'[Candid-PTX] [{self.image2study[group_idx]}]'
                image_source = "Candid"
                task_name = "Rib Fracture Segmentation"
                image_path = f'{self.data_dir}/{random.choice(neg_images)}'
                regions = []
                text = "rib fracture"
                target_text = "No rib fracture detected."
                qa_pair = form_qa_func(self.instruct)(text, target_text)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'image_source': image_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'region': regions,
                    'text': text,
                    'target_text': target_text,
                    'qa_pair': qa_pair
                }
                dataset[split].append(sample)
        return dataset

    def create_chest_tube_segmentation(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Chest Tube Segmentation] [Candid-PTX]"}

        neg_images = [image for images, impression in zip(self.images, self.impressions)
                      for image in images.split(",") if "tube" not in impression.lower()]
        pos_images = set(self.chest_tubes.anon_SOPUID.tolist())
        neg_images = [image for image in neg_images if image.split("/")[1][:-4] not in pos_images]

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train"], [self.chest_tubes]):
            split_data = split_data[split_data.mask_rle.map(lambda x: "-1" not in x)]
            for group_idx, group in split_data.groupby("anon_SOPUID"):
                unique_id = f"[Chest Tube Segmentation] [Candid-PTX] [{group_idx}]"
                if group_idx not in self.image2study:
                    continue
                study_id = f'[Candid-PTX] [{self.image2study[group_idx]}]'
                image_source = "Candid"
                task_name = "Chest Tube Segmentation"
                image_path = f'{self.image_dir}/{group_idx}.jpg'
                if not os.path.exists(image_path):
                    continue
                regions = [("Chest Tube", mask2polygen(rle2mask(rle, 1024, 1024, 1, 512, 512)))
                           for rle in group.mask_rle.tolist()]

                boxes = masks_to_boxes(torch.Tensor(np.array([
                    rle2mask(rle, 1024, 1024, 1, 512, 512)
                    for rle in group.mask_rle.tolist()
                ])))
                quantized_boxes = (boxes / 512 * qbins).int().tolist()
                quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
                text = "chest tube"
                target_text = "".join([
                    f"<|ref|>chest tube<|/ref|><|box|>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})<|/box|>" for b
                    in quantized_boxes
                ])

                qa_pair = form_qa_func(self.instruct)(text, target_text)

                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'image_source': image_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'region': regions,
                    'text': text,
                    'target_text': target_text,
                    'qa_pair': qa_pair
                }
                dataset[split].append(sample)

                # Negative Samples
                unique_id = f"[Chest Tube Segmentation] [Candid-PTX] [{group_idx} - negative]"
                study_id = f'[Candid-PTX] [{self.image2study[group_idx]}]'
                image_source = "Candid"
                task_name = "Chest Tube Segmentation"
                image_path = f'{self.data_dir}/{random.choice(neg_images)}'
                regions = []
                text = "chest tube"
                target_text = "No chest tubes detected."
                qa_pair = form_qa_func(self.instruct)(text, target_text)
                sample = {
                    'unique_id': unique_id,
                    'study_id': study_id,
                    'image_source': image_source,
                    'task_name': task_name,
                    'image_path': image_path,
                    'region': regions,
                    'text': text,
                    'target_text': target_text,
                    'qa_pair': qa_pair
                }
                dataset[split].append(sample)
        return dataset


def mask2polygen(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    segmentation = []
    for contour in contours:
        if contour.size >= 6:
            segmentation.append(contour.flatten().tolist())
    return segmentation[0]


def rle2mask(rle, height=1024, width=1024, fill_value=1, out_height=1024, out_width=1024):
    component = np.zeros((height, width), np.float32)
    component = component.reshape(-1)
    rle = np.array([int(s) for s in rle.strip().split(' ')])
    rle = rle.reshape(-1, 2)
    start = 0
    for index, length in rle:
        start = start + index
        end = start + length
        component[start: end] = fill_value
        start = end
    component = component.reshape(width, height).T
    if height != out_height or width != out_width:
        component = cv2.resize(component, (out_height, out_width)).astype(bool)
    return component.astype(np.uint8)


def mask2rle(component):
    component = component.T.flatten()
    start = np.where(component[1:] > component[:-1])[0] + 1
    end = np.where(component[:-1] > component[1:])[0] + 1
    length = end - start
    rle = []
    for i in range(len(length)):
        if i == 0:
            rle.extend([start[0], length[0]])
        else:
            rle.extend([start[i] - end[i - 1], length[i]])
    rle = ' '.join([str(r) for r in rle])
    return rle


def rle2bbox(rle, shape):
    a = np.fromiter(rle.split(), dtype=np.uint)
    a = a.reshape((-1, 2))  # an array of (start, length) pairs
    a[:, 0] -= 1  # `start` is 1-indexed

    y0 = a[:, 0] % shape[0]
    y1 = y0 + a[:, 1]
    if np.any(y1 > shape[0]):
        y0 = 0
        y1 = shape[0]
    else:
        y0 = np.min(y0)
        y1 = np.max(y1)

    x0 = a[:, 0] // shape[0]
    x1 = (a[:, 0] + a[:, 1]) // shape[0]
    x0 = np.min(x0)
    x1 = np.max(x1)

    if x1 > shape[1]:
        raise ValueError("invalid RLE or image dimensions: x1=%d > shape[1]=%d" % (x1, shape[1]))
    return x0, y0, x1, y1


if __name__ == '__main__':
    processor = CandidProcessor()
    processor.create_impression_generation()
    processor.create_pneumothorax_segmentation()
    processor.create_rib_fracture_segmentation()
    processor.create_chest_tube_segmentation()
