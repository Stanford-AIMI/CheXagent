import os

import requests
import torch
from PIL import Image, ImageDraw
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

class CheXagent(object):
    def __init__(self):
        # step 1: Setup constant
        self.model_name = "StanfordAIMI/CheXagent-2-3b"
        self.dtype = torch.bfloat16
        self.device = "cuda"

        # step 2: Load Processor and Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name, device_map="auto", trust_remote_code=True)
        self.model = self.model.to(self.dtype)
        self.model.eval()

    def generate(self, paths, prompt):
        # step 3: Inference
        query = self.tokenizer.from_list_format([*[{'image': path} for path in paths], {'text': prompt}])
        conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
        input_ids = self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
        output = self.model.generate(
            input_ids.to(self.device), do_sample=False, num_beams=1, temperature=1., top_p=1., use_cache=True,
            max_new_tokens=512
        )[0]
        response = self.tokenizer.decode(output[input_ids.size(1):-1])
        return response

    def view_classification(self, path):
        assert isinstance(path, str)
        prompt = "What is the view of this chest X-ray? Options: (a) PA, (b) AP, (c) LATERAL"
        response = self.generate([path], prompt)
        return response

    def view_matching(self, paths):
        assert isinstance(paths, list)
        prompt = "Do these images belong to the same study?"
        response = self.generate(paths, prompt)
        return response

    def binary_disease_classification(self, paths, disease_name):
        assert isinstance(paths, list)
        assert isinstance(disease_name, str)
        prompt = f'Does this chest X-ray contain a {disease_name}?'
        response = self.generate(paths, prompt)
        return response

    def disease_identification(self, paths, disease_names):
        assert isinstance(paths, list)
        assert isinstance(disease_names, list)
        prompt = f'Given the CXR, identify any diseases. Options:\n{", ".join(disease_names)}'
        response = self.generate(paths, prompt)
        return response

    def findings_generation(self, paths, indication):
        assert isinstance(paths, list)
        assert isinstance(indication, str)
        prompt = f'Given the indication: "{indication}", write a structured findings section for the CXR.'
        response = self.generate(paths, prompt)
        return response

    def findings_generation_section_by_section(self, paths):
        assert isinstance(paths, list)
        anatomies = [
            "Airway", "Breathing", "Cardiac",
            "Diaphragm",
            "Everything else (e.g., mediastinal contours, bones, soft tissues, tubes, valves, and pacemakers)"
        ]
        prompts = [f'Please provide a detailed description of "{anatomy}" in the chest X-ray' for anatomy in anatomies]
        responses = []
        for anatomy, prompt in zip(anatomies, prompts):
            response = self.generate(paths, prompt)
            responses.append((anatomy, response))
        return responses

    def image_text_matching(self, paths, text):
        assert isinstance(paths, list)
        assert isinstance(text, str)
        prompt = f'Decide if the provided image matches the following text: {text}'
        response = self.generate(paths, prompt)
        return response

    def plot_image(self, path, response, save_path):
        if path.startswith("http://") or path.startswith("https://"):
            pil_image = Image.open(requests.get(path, stream=True).raw)
        else:
            pil_image = Image.open(path)
        pil_image = pil_image.convert("RGB")

        boxes = [_dict["box"] for _dict in self.tokenizer.to_list_format(response) if "box" in _dict]
        boxes = [[int(cord) / 100 for cord in box.replace("(", "").replace(")", "").split(",")] for box in boxes]
        w = pil_image.width
        h = pil_image.height
        draw = ImageDraw.Draw(pil_image)
        for box in boxes:
            draw.rectangle((box[0] * w, box[1] * h, box[2] * w, box[3] * h), width=10, outline="#FF6969")
        pil_image = pil_image.convert("RGB")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        pil_image.save(save_path)
        print(f'Saving the result at {save_path}')

        return response

    def phrase_grounding(self, path, phrase, save_path=f'results_visualization/phrase_grounding.png'):
        assert isinstance(path, str)
        assert isinstance(phrase, str)
        prompt = f'Please locate the following phrase: {phrase}'
        response = self.generate([path], prompt)

        self.plot_image(path, response, save_path)

        return response

    def abnormality_detection(self, path, disease_name, save_path="results_visualization/abnormality_detection.png"):
        assert isinstance(path, str)
        assert isinstance(disease_name, str)
        prompt = f'Locate areas in the chest X-ray where {disease_name} are present, using bounding box coordinates'
        response = self.generate([path], prompt)

        self.plot_image(path, response, save_path)

        return response

    def chest_tube_detection(self, path, save_path="results_visualization/chest_tube_detection.png"):
        assert isinstance(path, str)
        prompt = f'Locate chest tubes and specify their positions with bounding box coordinates'
        response = self.generate([path], prompt)

        self.plot_image(path, response, save_path)

        return response

    def rib_fracture_detection(self, path, save_path="results_visualization/rib_fracture_detection.png"):
        assert isinstance(path, str)
        prompt = f'Locate rib fractures and specify their positions with bounding box coordinates'
        response = self.generate([path], prompt)

        self.plot_image(path, response, save_path)

        return response

    def foreign_objects_detection(self, path, save_path="results_visualization/foreign_objects_detection.png"):
        assert isinstance(path, str)
        prompt = ("Examine the chest X-ray for the presence of foreign objects, such as tubes, clips, or hardware, "
                  "and provide their locations with bounding box coordinates.")
        response = self.generate([path], prompt)

        self.plot_image(path, response, save_path)

        return response

    def temporal_image_classification(self, paths, disease_name):
        assert isinstance(paths, list)
        assert isinstance(disease_name, str)
        prompt = (f"You are given two images: one reference image and one new image. "
                  f"Please identify the progression of {paths}:\n(a) worsening\n(b) stable\n(c) improving")
        response = self.generate(paths, prompt)
        return response

    def findings_summarization(self, findings):
        assert isinstance(findings, str)
        prompt = f'Summarize the following Findings: {findings}'
        response = self.generate([], prompt)
        return response

    def named_entity_recognition(self, text):
        assert isinstance(text, str)
        prompt = (f'Given the list of entity types '
                  f'[Observation (Definitely Absent), '
                  f'Observation (Definitely Present), '
                  f'Observation (Uncertain), Anatomy], '
                  f'find out all words/phrases that indicate the above types of named entities. '
                  f'Sentence: {text}')

        response = self.generate([], prompt)
        return response
