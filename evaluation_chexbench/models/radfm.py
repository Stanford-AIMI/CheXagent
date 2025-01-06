import torch
from rich import print

from . import BaseLM
from .src.radfm import create_model_and_transforms, combine_and_preprocess


class RadFM(BaseLM):
    def __init__(self, name=f"RadFM", device="cuda", dtype=torch.float16):
        super().__init__(name=name, device=device, dtype=dtype)
        checkpoint = "pretrained_weights/radfm_weights/pytorch_model.bin"
        print(f"=> Loading RadFM model ({checkpoint})")
        self.model, self.text_tokenizer, self.image_padding_tokens = create_model_and_transforms()
        self.model = self.model.to(self.dtype).to(device)
        self.model.eval()

    def get_likelihood_prompt(self, question, options):
        prompts = [f"{question} {c}" for c in options]
        return prompts

    def process_img(self, paths, question="None"):

        image = [{'img_path': paths, 'position': 0}]
        text, image = combine_and_preprocess(question, image, self.image_padding_tokens)
        return image, text

    @torch.no_grad()
    def get_logits(self, image, prompts, options):
        image, _ = image
        image = image.repeat(len(prompts), 1, 1, 1, 1, 1).to(self.dtype).to(self.device)

        texts = [
            "<image>" + self.image_padding_tokens[0] + "</image>" + prompt for prompt in prompts
        ]

        tokenized_data = self.text_tokenizer(texts, max_length=2048, padding=True, truncation=True, return_tensors="pt")

        num_options = len(options)
        outputs = self.model(
            vision_x=image.to(self.device).to(self.dtype), lang_x=tokenized_data["input_ids"].to(self.device),
            attention_mask=tokenized_data["attention_mask"].to(self.device),
            labels=None, loss_reweight=None, key_words_query=None
        )

        output = torch.nn.functional.log_softmax(outputs["logits"], dim=-1)
        output = output[:, :-1, :]  # Remove last token

        all_outputs = []
        ans_ids = []
        for i in range(num_options):
            start = tokenized_data[i].char_to_token(texts[i].rfind(options[i]))
            ans_ids.append(tokenized_data.input_ids[i][start:])
            logprobs = output[i, start - 1:, :].unsqueeze(0)
            assert logprobs.size(1) == len(ans_ids[-1])
            all_outputs.append(logprobs)
        return all_outputs, ans_ids

    def compute_scores(self, likelihood, ans_indices, length_norm=False):
        scores = []
        for i in range(len(ans_indices)):
            a_ids = ans_indices[i].view(1, -1, 1).cuda()
            score = torch.gather(likelihood[i], 2, a_ids).squeeze().sum().item()
            if length_norm:
                score /= len(ans_indices[i])
            scores.append(score)
        return scores

    def generate(self, paths, prompt):
        image, _ = self.process_img(paths)
        text = ["<image>" + self.image_padding_tokens[0] + "</image>" + prompt]
        lang_x = self.text_tokenizer(
            text, max_length=2048, truncation=True, return_tensors="pt"
        )['input_ids']
        image = image.to(self.dtype)
        generation = self.model.generate(lang_x.to(self.device), image.to(self.device))
        generated_text = self.text_tokenizer.batch_decode(generation, skip_special_tokens=True)[0]
        return generated_text
