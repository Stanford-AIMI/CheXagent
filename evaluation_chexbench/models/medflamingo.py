import os
from abc import ABC, abstractmethod

import torch
from PIL import Image
from einops import repeat
from huggingface_hub import hf_hub_download
from open_flamingo import create_model_and_transforms
from rich import print

from . import BaseLM


class AbstractProcessor(ABC):
    """
    Abstract class for processors to show what methods they need to implement.
    Processors handle text encoding and image preprocessing.
    """

    @abstractmethod
    def encode_text(self, prompt):
        pass

    @abstractmethod
    def preprocess_images(self, images: list):
        pass


class FlamingoProcessor(AbstractProcessor):
    """
    Processor class for Flamingo.
    """

    def __init__(self, tokenizer, vision_processor):
        """
        OF does not use same vision processor, image_processor only transforms single image
        """
        self.tokenizer = tokenizer
        self.vision_processor = vision_processor

    def encode_text(self, prompt):
        self.tokenizer.padding_side = "left"
        # For generation padding tokens should be on the left
        return self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    def preprocess_images(self, images: list):
        vision_x = [self.vision_processor(im).unsqueeze(0) for im in images]
        vision_x = torch.cat(vision_x, dim=0)
        return vision_x


def clean_generation(response):
    """
    for some reason, the open-flamingo based model slightly changes the input prompt (e.g. prepends <unk>, an adds some spaces)
    """
    return response.replace('<unk> ', '').strip()


def create_medflamingo():
    llama_path = "pretrained_weights/med-flaimgo/llama-7b/"
    if not os.path.exists(llama_path):
        raise ValueError('Llama model not yet set up, please check README for instructions!')
    model, image_processor, tokenizer = create_model_and_transforms(
        clip_vision_encoder_path="ViT-L-14",
        clip_vision_encoder_pretrained="openai",
        lang_encoder_path=llama_path,
        tokenizer_path=llama_path,
        cross_attn_every_n_layers=4
    )
    checkpoint_path = hf_hub_download(
        "med-flamingo/med-flamingo", "model.pt", cache_dir="pretrained_weights/med-flaimgo/"
    )
    print(f'Downloaded Med-Flamingo checkpoint to {checkpoint_path}')
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=False)
    processor = FlamingoProcessor(tokenizer, image_processor)
    model = model
    model.eval()
    return model, processor


class MedFlamingo(BaseLM):
    def __init__(self, name=f"MedFlamingo", device="cuda", dtype=torch.float16):
        super().__init__(name=name, device=device, dtype=dtype)
        checkpoint = "med-flamingo/med-flamingo"
        print(f"=> Loading MedFlamingo model ({checkpoint})")
        self.model, self.processor = create_medflamingo()
        self.model = self.model.to(device).to(self.dtype)
        self.model.eval()

    def get_likelihood_prompt(self, question, options):
        prompts = [
            f"You are a helpful medical assistant. You are being provided with images, "
            f"a question about the image and an answer. Answer the question. "
            f"<image>Question: {question} Answer: {c}" for c in options
        ]
        return prompts

    def process_img(self, paths):
        if isinstance(paths, str):
            paths = paths.split("|")
        assert len(paths) == 1
        paths = [Image.open(path) for path in paths]
        image = self.processor.preprocess_images(paths)
        image = repeat(image, 'N c h w -> b N T c h w', b=1, T=1)
        return image

    @torch.no_grad()
    def get_logits(self, image, prompts, options):
        image = image.repeat(len(prompts), 1, 1, 1, 1, 1).to(self.dtype).to(self.device)
        tokenized_data = self.processor.encode_text(prompts)
        num_options = len(options)
        outputs = self.model(
            vision_x=image.to(self.device).to(self.dtype),
            lang_x=tokenized_data["input_ids"].to(self.device),
            attention_mask=tokenized_data["attention_mask"].to(self.device)
        )
        output = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        output = output[:, :-1, :]  # Remove last token
        all_outputs = []
        ans_indices = []
        for i in range(num_options):
            start = tokenized_data[i].char_to_token(prompts[i].rfind(options[i]))
            ans_indices.append(tokenized_data.input_ids[i][start:])
            logprobs = output[i, start - 1:, :].unsqueeze(0)
            assert logprobs.size(1) == len(ans_indices[-1])
            all_outputs.append(logprobs)
        return all_outputs, ans_indices

    def compute_scores(self, likelihood, ans_indices, length_norm=False):
        scores = []
        for i in range(len(ans_indices)):
            a_ids = ans_indices[i].view(1, -1, 1).cuda()
            score = torch.gather(likelihood[i], 2, a_ids).squeeze().sum().item()
            if length_norm:
                score /= len(ans_indices[i])
            scores.append(score)
        return scores

    def generate(self, paths, prompts):
        image = self.process_img(paths)
        if not isinstance(prompts, list):
            prompts = [prompts]

        prefix = ("You are a helpful medical assistant. "
                  "You are being provided with images and a question about the image. "
                  "<image>Question: {} Answer: ")

        prompts = [prefix.format(p) for p in prompts]

        image = image.repeat(len(prompts), 1, 1, 1, 1, 1).to(self.dtype).to(self.device)
        tokenized_data = self.processor.encode_text(prompts)

        generated_text = self.model.generate(
            vision_x=image.to(self.device),
            lang_x=tokenized_data["input_ids"].to(self.device),
            attention_mask=tokenized_data["attention_mask"].to(self.device),
            max_new_tokens=128,
        )
        processed_generated_text = []

        for inp, gen in zip(tokenized_data["input_ids"], generated_text):
            processed_generated_text.append(gen[len(inp):])

        response = self.processor.tokenizer.decode(processed_generated_text[0])
        response = clean_generation(response)
        return response
