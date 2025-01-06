import numpy as np
import torch

from . import BaseLM


class Random(BaseLM):
    def __init__(self, name=f"Random", device="cuda", dtype=torch.float16):
        super().__init__(name=name, device=device, dtype=dtype)

    def get_likelihood_prompt(self, question, options):
        return None

    def process_img(self, path):
        return None

    def get_logits(self, pixel_values, prompts, options):
        return None, options

    def compute_scores(self, logits, ans_indices, length_norm):
        return np.random.random(len(ans_indices))
