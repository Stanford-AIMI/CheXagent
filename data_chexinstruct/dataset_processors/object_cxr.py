import pandas as pd
from PIL import Image

from .base_processor import BaseProcessor
from .templates import create_template


def cvt(coord):
    coord = min(coord, 999)
    coord = round(coord / 10)
    return str(coord)


class ObjectCXRProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()
        self.data_dir = "data/object-cxr"
        train_path, val_path = f'{self.data_dir}/train.csv', f'{self.data_dir}/dev.csv'
        self.train_data, self.val_data = pd.read_csv(train_path), pd.read_csv(val_path)

    def create_foreign_objects_detection(self, qbins=1000):
        dataset = {"train": [], "val": [], "test": [], "dataset_name": "[Foreign Object Detection] [Object-CXR]"}

        form_qa_func = create_template(dataset["dataset_name"])
        for split, split_data in zip(["train", "val"], [self.train_data, self.val_data]):
            for row_idx, row in split_data.iterrows():
                unique_id = f"[Foreign Object Detection] [Object-CXR] [{row_idx}]"
                study_id = f'[object-cxr] [{split}-{row.image_name.split(".")[0]}]'
                image_source = "Object-CXR"
                task_name = "Foreign Object Detection"
                image_path = f'{self.data_dir}/{split.replace("val", "dev")}/{row.image_name}'
                if isinstance(row.annotation, float):
                    regions = []
                    text = "foreign object"
                    target_text = "No foreign objects detected."
                else:
                    # 0, 1 for rectangular box; 2 for segmentation polygen
                    regions = [[int(box.split(" ")[0]), [int(x) for x in box.split(" ")[1:]]]
                               for box in row.annotation.split(";")]
                    for idx in range(len(regions)):
                        if regions[idx][0] == 2:
                            x_1, x_2 = max(min(regions[idx][1][::2]), 0), max(regions[idx][1][::2])
                            y_1, y_2 = max(min(regions[idx][1][1::2]), 0), max(regions[idx][1][1::2])
                            regions[idx][1] = [x_1, y_1, x_2, y_2]
                    text = "foreign object"
                    image = Image.open(image_path)
                    quantized_boxes = [
                        [int((r[1][0] / image.width) * qbins), int((r[1][1] / image.height) * qbins),
                         int((r[1][2] / image.width) * qbins), int((r[1][3] / image.height) * qbins)]
                        for r in regions
                    ]
                    quantized_boxes = sorted(quantized_boxes, key=lambda x: (x[0], x[1]))
                    target_text = "".join(
                        [
                            f"<|ref|>foreign object<|/ref|><|box|>({cvt(b[0])},{cvt(b[1])}),({cvt(b[2])},{cvt(b[3])})<|/box|>"
                            for b in quantized_boxes
                        ]
                    )

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


if __name__ == '__main__':
    processor = ObjectCXRProcessor()
    processor.create_foreign_objects_detection()
