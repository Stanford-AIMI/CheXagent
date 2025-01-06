import json
import random

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.io import read_image
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks


def load_data():
    with open("data_chexinstruct/data_chexinstruct.json", "rt") as f:
        datasets = json.load(f)
    name2datasets = dict()
    for dataset in datasets:
        name2datasets[dataset["dataset_name"]] = dataset
    return name2datasets


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


def fetch_instance(instance):
    # Fetch Images
    images = instance.get("image_path", [])
    if not isinstance(images, list):
        images = [images]
    image_1, image_2, *_ = images + [None, None]
    if "detection" in instance["task_name"].lower() and instance["region"]:
        image_2 = read_image(image_1)
        boxes, labels = [region[1] for region in instance["region"]], [str(region[0]) for region in instance["region"]]
        image_2 = draw_bounding_boxes(image_2, torch.tensor(boxes), labels, colors="red", width=4)
        image_2 = to_pil_image(image_2)
    if "ground" in instance["task_name"].lower() and instance["region"]:
        image_2 = read_image(image_1)
        boxes = instance["region"]
        image_2 = draw_bounding_boxes(image_2, torch.tensor(boxes), colors="red", width=4)
        image_2 = to_pil_image(image_2)
    if "segment" in instance["task_name"].lower() and instance["region"]:
        image_2 = read_image(image_1)
        h, w = image_2.size(1), image_2.size(2)
        masks = []
        for region in instance["region"]:
            mask = Image.new('L', (w, h), 0)
            ImageDraw.Draw(mask).polygon(region[1], outline=1, fill=1)
            mask = np.array(mask)
            masks.append(mask)
        if masks:
            image_2 = draw_segmentation_masks(image_2, torch.BoolTensor(masks), alpha=0.5, colors=["red"] * len(masks))
        image_2 = to_pil_image(image_2)

    text = instance.get("text", "")
    target_text = instance.get("target_text", "")
    options = instance.get("options", {})
    qa_pair = instance.get("qa_pair", "")

    return image_1, image_2, text, target_text, options, qa_pair


def manual_select(task_dataset_name):
    print(task_dataset_name)
    task_name, dataset_name = task_dataset_name[1:-1].split("] [")
    dataset = data[task_dataset_name]
    split = random.choice([split for split in ["train", "val", "test"] if len(dataset[split]) > 0])
    instance = random.choice(dataset[split])
    image_1, image_2, text, target_text, options, qa_pair = fetch_instance(instance)
    return task_name, dataset_name, split, instance, image_1, image_2, text, target_text, options, qa_pair


def main():
    with gr.Blocks() as selection_demo:
        with gr.Row():
            choice = gr.Dropdown(task_dataset_names, label="[Task] [Dataset]")
            btn = gr.Button(value="Sample")
        with gr.Row():
            task_name = gr.Textbox(value="", label="Task")
            dataset_name = gr.Textbox(value="", label="Dataset")
            split = gr.Textbox(value="", label="Split")
        qa_pair = gr.JSON(value="{}", label="QA Pair")
        with gr.Row():
            image_1 = gr.Image(type="pil", height=640)
            image_2 = gr.Image(type="pil", height=640)
        with gr.Row():
            text = gr.Textbox(value="", label="Source Text")
            target_text = gr.Textbox(value="", label="Target Text")
            options = gr.JSON(value="{}", label="Options")
        instance = gr.JSON(value="{}", label="Instance Information")
        btn.click(manual_select, inputs=[choice], outputs=[task_name, dataset_name, split, instance,
                                                           image_1, image_2, text, target_text, options, qa_pair])

    demo = gr.TabbedInterface([selection_demo], ["Selection"])
    demo.launch(server_name="0.0.0.0", server_port=8888)


if __name__ == "__main__":
    data = load_data()
    task_dataset_names = sorted(list(data.keys()))
    main()
