import torch
from PIL import Image
from rich import print
from transformers import Blip2Processor, Blip2ForConditionalGeneration

from . import BaseLM


class BLIP2(BaseLM):
    def __init__(self, name=f"BLIP2", device="cuda", dtype=torch.float16):
        super().__init__(name=name, device=device, dtype=dtype)
        checkpoint = "Salesforce/blip2-opt-2.7b"  # Salesforce/blip2-opt-6.7b
        print(f"=> Loading BLIP2 model ({checkpoint})")
        self.checkpoint = checkpoint
        self.processor = Blip2Processor.from_pretrained(checkpoint)
        self.processor.tokenizer.padding_side = "left"
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            checkpoint, load_in_8bit=True, torch_dtype=torch.float16
        )
        self.model.eval()

    def get_likelihood_prompt(self, question, options):
        prompts = [f"{question} {c}" for c in options]
        return prompts

    def process_img(self, paths):
        if isinstance(paths, str):
            paths = paths.split("|")
        assert len(paths) == 1, "BLIP2 only support one image."
        image = Image.open(paths[0])
        image = self.processor(images=image, text=None, return_tensors="pt").to(device=self.device, dtype=torch.float16)
        return image.pixel_values

    @torch.no_grad()
    def get_logits(self, pixel_values, prompts, options):
        num_options = len(options)
        input = self.processor(images=None, text=prompts, padding=True, return_tensors="pt").to(device=self.device)
        input['pixel_values'] = torch.cat([pixel_values] * num_options)
        logits = self.model(**input).logits.detach()
        output = torch.nn.functional.log_softmax(logits, dim=-1)
        output = output[:, :-1, :]
        all_outputs = []
        ans_indices = [torch.tensor(self.processor.tokenizer(f" {ans}").input_ids[1:]) for ans in options]
        for i in range(num_options):
            num_ans_tokens = len(ans_indices[i])
            assert (input.input_ids[i, :].flatten()[-num_ans_tokens:].cpu() == ans_indices[i]).all().item()
            all_outputs.append(output[i, -num_ans_tokens:, :].unsqueeze(0))
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
