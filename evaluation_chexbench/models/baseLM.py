import abc
import random
import re


class BaseLM(abc.ABC):
    def __init__(self, name, device, dtype):
        self.device = device
        self.dtype = dtype
        self.name = name

    def process_img(self, paths):
        raise NotImplementedError()

    def get_likelihood_prompt(self, question, options):
        raise NotImplementedError()

    def get_logits(self, pixel_values, prompt_ids, ans_ids):
        raise NotImplementedError()

    def compute_scores(self, likelihood, ans_indices, length_norm):
        raise NotImplementedError()

    def get_prompt(self, question, options):
        raise NotImplementedError()

    def parse_response(self, response, target, options):
        print(f'Response: {response}; Target: {options[target]}')
        choice_style = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        prediction = re.findall("\(([A-Z])\)", response)
        prediction = choice_style[random.choice(list(range(len(options))))] if len(prediction) == 0 else prediction[0]
        target = choice_style[target]
        return prediction.lower() == target.lower()
