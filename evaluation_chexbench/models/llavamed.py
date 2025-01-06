import torch
from PIL import Image
from rich import print
from transformers import AutoTokenizer, AutoConfig, CLIPImageProcessor

from . import BaseLM
from .src.llava import LlavaLlamaForCausalLM

# See this script for example implementation: https://github.com/microsoft/LLaVA-Med/blob/main/llava/eval/model_vqa.py
# Official Llava-Med prompt format: 
# prompt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions.###"
# prompt += "Human: Hi!###"
# prompt += "Assistant: Hi there!  How can I help you today?\n###"
# prompt = f"Human: {qs}###"
# prompt += f"Assistant: {options[i]}"

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


class LlavaMed(BaseLM):
    def __init__(self, name=f"LLaVAMed", device="cuda", dtype=torch.float16):
        super().__init__(name=name, device=device, dtype=dtype)
        checkpoint = "pretrained_weights/llava_med_weights"
        print(f"\n=> Loading LlavaMed model ({checkpoint})")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.tokenizer.padding_side = "left"
        patch_config(checkpoint)
        self.model = LlavaLlamaForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.float16).to(device)
        self.processor = CLIPImageProcessor.from_pretrained(
            self.model.config.mm_vision_tower,
            torch_dtype=torch.float16
        )
        self.mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if self.mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        vision_tower = self.model.model.vision_tower[0]
        vision_tower.to(device=self.device, dtype=self.dtype)
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = self.mm_use_im_start_end
        if self.mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids(
                [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        self.image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2
        self.model.eval()

    def get_likelihood_prompt(self, question, options):
        prompts = [f"{question} {c}" for c in options]
        return prompts

    def process_img(self, paths):
        if isinstance(paths, str):
            paths = paths.split("|")
        assert len(paths) == 1
        path = paths[0]
        image = self.processor.preprocess(Image.open(path), return_tensors='pt')['pixel_values'][0]
        return image

    def get_logits(self, pixel_values, prompts, options):
        all_prompts = []
        for i in range(len(prompts)):
            qs = prompts[i][0:-len(options[i]) - 1]
            if self.mm_use_im_start_end:
                qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN
            else:
                qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
            prompt = f"{qs} {options[i]}"
            all_prompts.append(prompt)
        tokens = self.tokenizer(all_prompts, return_tensors="pt", padding=True)
        input_ids = tokens.input_ids.cuda()
        attn_mask = tokens.attention_mask.cuda()
        with torch.no_grad():
            pixel_values = torch.cat([pixel_values.unsqueeze(0)] * len(options)).half().cuda()
            logits = self.model(input_ids, images=pixel_values, attention_mask=attn_mask).logits.detach()
        output = torch.nn.functional.log_softmax(logits, dim=-1)
        output = output[:, :-1, :]  # Remove last token

        all_outputs = []
        ans_indices = [torch.tensor(self.tokenizer(f"{ans}").input_ids[1:]) for ans in options]
        for i in range(len(options)):
            num_ans_tokens = len(ans_indices[i])
            assert (input_ids[i, :].flatten()[-num_ans_tokens:].cpu() == ans_indices[i]).all().item()
            all_outputs.append(output[i, -num_ans_tokens:, :].unsqueeze(0))
        return all_outputs, ans_indices

    def compute_scores(self, likelihood, ans_indices, length_norm=False):
        scores = []
        for i in range(len(likelihood)):
            a_ids = ans_indices[i].view(1, -1, 1).cuda()
            score = torch.gather(likelihood[i], 2, a_ids).squeeze().sum().item()
            if length_norm:
                score /= len(ans_indices[i])
            scores.append(score)
        return scores

    def generate(self, paths, prompt, **kwargs):
        from transformers import StoppingCriteria
        class KeywordsStoppingCriteria(StoppingCriteria):
            def __init__(self, keywords, tokenizer, input_ids):
                self.keywords = keywords
                self.tokenizer = tokenizer
                self.start_len = None
                self.input_ids = input_ids

            def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
                if self.start_len is None:
                    self.start_len = self.input_ids.shape[1]
                else:
                    outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
                    for keyword in self.keywords:
                        if keyword in outputs:
                            return True
                return False

        prefix = "A chat between a curious human and an artificial intelligence assistant. " \
                 "The assistant gives helpful, detailed, and polite answers to the human's questions. " \
                 "###Human: "

        images = self.process_img(paths).unsqueeze(0).to(torch.float16)
        if self.mm_use_im_start_end:
            qs = prefix + prompt + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN + "### Assistant: "
        else:
            qs = prefix + prompt + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
        input_ids = torch.as_tensor(self.tokenizer([qs], padding=True).input_ids).cuda()
        keywords = ['###']
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)
        generation = self.model.generate(
            input_ids.to(self.device), images=images.to(self.device), stopping_criteria=[stopping_criteria],
            do_sample=True, temperature=0.7, max_new_tokens=1024
        )
        generated_text = self.tokenizer.decode(generation[0], skip_special_tokens=True)
        generated_text = generated_text.split(" ### Assistant: ")[-1].replace("###", "").strip()
        return generated_text
