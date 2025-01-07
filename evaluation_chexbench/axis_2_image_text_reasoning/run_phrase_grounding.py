import copy
import json
import os

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def bbox_iou(box1, box2, x1y1x2y2=True):
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1, 0) * torch.clamp(inter_rect_y2 - inter_rect_y1, 0)

    b1_area = (b1_x2 - b1_x1) * (b1_y2 - b1_y1)
    b2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1)

    return inter_area / (b1_area + b2_area - inter_area + 1e-16)


@torch.no_grad()
def main():
    # Step 1: load tokenizer and model
    model_name = "StanfordAIMI/CheXagent-2-3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True).eval()

    # Step 2: Fetch inputs
    data_path = "evaluation_chexbench/data.json"
    bench = json.load(open(data_path))
    task_samples = bench["Phrase Grounding"]

    # Step 3: Inference
    reference_boxes, candidate_boxes, last = [], [], []
    for sample in task_samples:
        paths = sample["image_path"]
        prompt = sample["question"]
        query = tokenizer.from_list_format([*[{'image': path} for path in paths], {'text': prompt}])
        conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
        input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
        generate_kwargs = dict(
            input_ids=input_ids.to(model.device),
            do_sample=False,
            num_beams=1,
            temperature=1.,
            top_p=1.,
            use_cache=True,
            max_new_tokens=512
        )
        output = model.generate(**generate_kwargs)[0]
        response = tokenizer.decode(output[input_ids.size(1):-1])
        reference = sample["answer"]
        reference_box = [_dict["box"] for _dict in tokenizer.to_list_format(reference) if "box" in _dict]
        candidate_box = [_dict["box"] for _dict in tokenizer.to_list_format(response) if "box" in _dict]
        if len(candidate_box) == 0:
            candidate_box = last
        else:
            last = copy.deepcopy(candidate_box)
        reference_box = [int(cord) for cord in reference_box[0].replace("(", "").replace(")", "").split(",")]
        candidate_box = [int(cord) for cord in candidate_box[0].replace("(", "").replace(")", "").split(",")]
        reference_boxes.append(reference_box)
        candidate_boxes.append(candidate_box)

        reference_box = torch.tensor([reference_box])
        candidate_box = torch.tensor([candidate_box])
        iou = bbox_iou(reference_box, candidate_box).item()

        sample["reference_box"] = reference_box.tolist()
        sample["candidate_box"] = candidate_box.tolist()
        sample["slice"] = iou

        print(f'Reference: {reference}')
        print(f'Candidate: {response}')
        print()

    save_dir = "evaluation_chexbench/results/axis_2_image_text_reasoning"
    save_path = f"{save_dir}/predictions/Phrase Grounding/predictions_CheXagent.json"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    json.dump(task_samples, open(save_path, "wt"), indent=2, ensure_ascii=False)


if __name__ == '__main__':
    assert os.path.exists(
        "evaluation_chexbench/data.json"
    ), "Please download the evaluation_chexbench/data.json file from [https://huggingface.co/datasets/StanfordAIMI/chexbench]."
    main()
