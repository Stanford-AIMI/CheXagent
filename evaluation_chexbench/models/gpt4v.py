import base64
import json
import time

import requests
import torch
from rich import print

from . import BaseLM


class GPT4V(BaseLM):
    def __init__(self, name=f"GPT-4V", device="cuda", dtype=torch.float16):
        super().__init__(name=name, device=device, dtype=dtype)
        print(f"\n=> Loading GPT-4V model")
        api_key = ""
        self.url = f"https://api.openai.com/v1/chat/completions"
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}

    def get_prompt(self, question, options):
        choice_style = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        option_str = ", ".join([f"({choice_style[option_idx]}) {option}" for option_idx, option in enumerate(options)])
        option_notations = ", ".join([f"({choice_style[option_idx]})" for option_idx, option in enumerate(options)])
        prompt = f'{question}\nOptions: {option_str}\n' \
                 f'Directly answer the question with the corresponding option: {option_notations}.'
        return prompt

    def process_img(self, paths):
        if isinstance(paths, str):
            paths = paths.split("|")
        return paths

    def generate(self, paths, prompt, **kwargs):
        base64_images = []
        for path in paths.split("|"):
            with open(path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode('utf-8')
            base64_images.append(base64_image)
        body = {
            "model": "gpt-4-turbo",
            "messages": [
                {
                    "role": "system", "content": "You are taking the medical examination. "
                                                 "Please answer it as accurate as possible."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        },
                        *[{
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        } for base64_image in base64_images]
                    ]
                }
            ],
            "max_tokens": 512,
            "temperature": 0,
            "stream": False
        }
        response = requests.post(self.url, headers=self.headers, json=body)

        try:
            response = json.loads(response.content)["choices"][0]["message"]["content"]
        except:
            print(response.content)
            response = ""

        time.sleep(5)
        return response
