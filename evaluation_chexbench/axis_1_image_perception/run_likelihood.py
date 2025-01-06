import json
import os
from collections import defaultdict

import numpy as np
from rich import print
from tqdm import tqdm

from evaluation_chexbench.models import Random, BLIP2, InstructBLIP, MedFlamingo, LlavaMed, RadFM, XrayGPT


def load_models():
    baselines = [
        ("Random", Random),
        ("BLIP2", BLIP2),
        ("InstructBLIP", InstructBLIP),
        ("MedFlamingo", MedFlamingo),
        ("LlavaMed", LlavaMed),
        ("RadFM", RadFM),
        ("XrayGPT", XrayGPT),
    ]

    return baselines


def create_tasks():
    all_tasks = [
        "View Classification",
        "Binary Disease Classification",
        "Single Disease Identification",
        "Multiple Disease Identification",
        "Temporal Classification",
    ]
    return all_tasks


def main():
    # Constant
    data_path = "evaluation_chexbench/data.json"
    save_dir = "evaluation_chexbench/results/axis_1_image_perception"
    os.makedirs(save_dir, exist_ok=True)

    # Load data
    bench = json.load(open(data_path))

    # Create tasks
    all_tasks = create_tasks()

    # Load models
    models = load_models()

    # Evaluate baselines on each task
    for model_name, model in models:
        model = model()
        results, results_slice = defaultdict(list), defaultdict(list)
        for task_name in all_tasks:
            # Create task samples
            print(f"Evaluating {task_name}:")
            task_samples = bench[task_name]

            # Print two task samples
            options = [task_samples[0][f'option_{k}'] for k in [0, 1, 2, 3]]
            print("Example Question/Options:", task_samples[0]['question'], [x for x in options if isinstance(x, str)])

            # Inference
            correct, total = defaultdict(int), defaultdict(int)
            correct_slice, total_slice = defaultdict(int), defaultdict(int)

            for i in tqdm(range(len(task_samples))):
                data_source = task_samples[i]["data_source"]
                img_path = task_samples[i]['image_path']
                question = task_samples[i]['question']
                options = [task_samples[i][f'option_{k}'] for k in [0, 1, 2, 3]]
                options = [x for x in options if isinstance(x, str)]
                target = task_samples[i]["answer"]

                img = model.process_img(img_path)
                prompts = model.get_likelihood_prompt(question, options)
                likelihood, ans_indices = model.get_logits(img, prompts, options)
                scores = model.compute_scores(likelihood, ans_indices, True)
                response = np.argmax(scores)
                is_correct = (response == target).item()

                task_samples[i]["reference"] = options[target]
                task_samples[i]["candidate"] = options[response]
                task_samples[i]["correct"] = is_correct

                correct[data_source] += is_correct
                correct_slice[f'{data_source}__{task_samples[i]["slice"]}'] += is_correct
                total[data_source] += 1
                total_slice[f'{data_source}__{task_samples[i]["slice"]}'] += 1

            for k in total.keys():
                print(f"[{k}] {correct[k] / total[k] * 100:.4f}")
                results[task_name].append({k: {"accuracy": correct[k] / total[k] * 100}})

            for k in total_slice.keys():
                print(f"[{k}] {correct_slice[k] / total_slice[k] * 100:.4f}")
                results_slice[task_name].append({k: {"accuracy": correct_slice[k] / total_slice[k] * 100}})

            # Save the results
            save_path = os.path.join(save_dir, "predictions", task_name, f"predictions_{model_name}.json")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            json.dump(task_samples, open(save_path, "wt"), ensure_ascii=False, indent=2)

        metric_dir = os.path.join(save_dir, "metrics")
        os.makedirs(metric_dir, exist_ok=True)
        print("By source:")
        print(results)
        save_path = os.path.join(metric_dir, f"metrics_{model_name}.json")
        json.dump(results, open(save_path, "wt"), ensure_ascii=False, indent=2)

        print("By slice:")
        print(results_slice)
        save_path = os.path.join(metric_dir, f"metrics_slice_{model_name}.json")
        json.dump(results_slice, open(save_path, "wt"), ensure_ascii=False, indent=2)


if __name__ == '__main__':
    main()
