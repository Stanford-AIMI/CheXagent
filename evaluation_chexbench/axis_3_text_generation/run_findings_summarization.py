import json
import os
from collections import defaultdict

import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object

from evaluation_chexbench.models import CheXagent


def main(model_name="CheXagent", num_beams=1):
    """Findings Summarization for CheXagent supporting multi-gpu inference"""
    # constant
    data_path = "evaluation_chexbench/data.json"
    save_dir = f"evaluation_chexbench/results/axis_3/axis_3_text_generation"
    os.makedirs(save_dir, exist_ok=True)

    # load benchmark
    bench = json.load(open(data_path))
    dataset = bench["Findings Summarization"]
    accelerator = Accelerator()

    # load the model
    model = CheXagent(device=f"cuda:{accelerator.process_index}")
    accelerator.wait_for_everyone()

    # inference
    results = []
    for sample_idx, sample in tqdm.tqdm(enumerate(dataset)):
        if sample_idx % accelerator.num_processes != accelerator.process_index:
            continue
        text = model.generate(
            [], f'Summarize the following Findings: {sample["section_findings"]}',
            num_beams=num_beams
        )
        results.append({
            "sample_idx": sample_idx,
            "image_path": sample["image_path"],
            "section_findings": sample["section_findings"],
            "section_impression": sample["section_impression"],
            "candidate_impression": text,
        })

    # gather results from multiple processes
    results = [results]
    results = gather_object(results)
    to_save = defaultdict()
    if accelerator.is_main_process:
        for dataset_name in results[0]:
            to_save[dataset_name] = [sample for result in results for sample in result[dataset_name]]
            to_save[dataset_name] = sorted(to_save[dataset_name], key=lambda x: x["sample_idx"])
        save_path = f'{save_dir}/predictions/Findings Summarization/{model_name}.json'
        json.dump(to_save, open(save_path, "wt"), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    assert os.path.exists(
        "evaluation_chexbench/data.json"
    ), "Please download the evaluation_chexbench/data.json file from [https://huggingface.co/datasets/StanfordAIMI/chexbench]."
    main()
