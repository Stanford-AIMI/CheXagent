import io

import requests
import torch
from PIL import Image
from rich import print
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig


def download_image(url):
    resp = requests.get(url)
    resp.raise_for_status()
    return Image.open(io.BytesIO(resp.content)).convert("RGB")


def generate(images, prompt, processor, model, device, dtype, generation_config):
    inputs = processor(
        images=images[:2], text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt"
    ).to(device=device, dtype=dtype)
    output = model.generate(**inputs, generation_config=generation_config)[0]
    response = processor.tokenizer.decode(output, skip_special_tokens=True)
    return response


def main():
    # step 1: Setup constant
    device = "cuda"
    dtype = torch.float16

    # step 2: Load Processor and Model
    processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
    generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
    model = AutoModelForCausalLM.from_pretrained(
        "StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True
    ).to(device)

    # step 3: Fetch the images
    image_path = "https://upload.wikimedia.org/wikipedia/commons/3/3b/Pleural_effusion-Metastatic_breast_carcinoma_Case_166_%285477628658%29.jpg"
    images = [download_image(image_path)]

    # step 4: Generate the Findings section
    for anatomy in anatomies:
        prompt = f'Describe "{anatomy}"'
        response = generate(images, prompt, processor, model, device, dtype, generation_config)
        print(f"Generating the Findings for [{anatomy}]:")
        print(response)


if __name__ == '__main__':
    anatomies = [
        "Airway", "Breathing", "Cardiac", "Diaphragm",
        "Everything else (e.g., mediastinal contours, bones, soft tissues, tubes, valves, and pacemakers)"
    ]
    main()
