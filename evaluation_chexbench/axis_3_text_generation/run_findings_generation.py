import json
import os
import re

import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object
from f1chexbert import F1CheXbert

from evaluation_chexbench.models import CheXagent


def main(model_name="CheXagent", num_beams=1):
    """Findings Generation for CheXagent supporting multi-gpu inference"""
    # constant
    data_path = "evaluation_chexbench/data.json"
    save_dir = f"evaluation_chexbench/results/axis_3/axis_3_text_generation"

    # load benchmark
    bench = json.load(open(data_path))
    dataset = bench["Findings Generation"]
    accelerator = Accelerator()

    # load the model
    model = CheXagent(device=f"cuda:{accelerator.process_index}")
    accelerator.wait_for_everyone()

    # inference
    results = []
    for sample_idx, sample in tqdm.tqdm(enumerate(dataset), total=len(dataset)):
        if sample_idx % accelerator.num_processes != accelerator.process_index:
            continue

        text = model.generate(
            sample["image_path"][:2],
            f'Given the indication: "{sample["section_indication"]}", write a structured findings section for the CXR.',
            num_beams=num_beams
        )
        results.append({
            "data_source": sample["data_source"],
            "sample_idx": sample_idx,
            "image_path": sample["image_path"],
            "section_indication": sample["section_indication"],
            "section_findings": sample["section_findings"],
            "candidate_findings": text,
        })

    # gather results from multiple processes
    results = [results]
    results = gather_object(results)
    if accelerator.is_main_process:
        to_save = [sample for result in results for sample in result]
        to_save = sorted(to_save, key=lambda x: x["sample_idx"])
        save_path = f'{save_dir}/predictions/Findings Generation/{model_name}.json'
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        json.dump(to_save, open(save_path, "wt"), ensure_ascii=False, indent=2)


def compute_scores():
    clean_text = lambda x: re.sub("\s+", " ", re.sub("\[.*?\]", "", x).replace("**", "")).strip().lower()
    root_dir = f"evaluation_chexbench/results/axis_3/axis_3_text_generation/predictions/Findings Generation/"
    result_path = f"{root_dir}/CheXagent.json"
    data = json.load(open(result_path))

    candidates = [clean_text(sample["candidate_findings"]) for sample in data]
    references = [clean_text(sample["section_findings"]) for sample in data]
    text_pairs = [(cand, refer) for cand, refer in zip(candidates, references) if refer]
    candidates, references = [pair[0] for pair in text_pairs], [pair[1] for pair in text_pairs]
    assert len(candidates) == len(references)
    scores = F1CheXbert()(references, candidates)
    print(scores)


if __name__ == '__main__':
    assert os.path.exists(
        "evaluation_chexbench/data.json"
    ), "Please download the evaluation_chexbench/data.json file from [https://huggingface.co/datasets/StanfordAIMI/chexbench]."
    main()
    compute_scores()
