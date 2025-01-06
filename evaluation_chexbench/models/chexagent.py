import torch
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import BaseLM


class CheXagent(BaseLM):
    def __init__(self, name=f"CheXagent", device="cuda", dtype=torch.bfloat16):
        super().__init__(name=name, device=device, dtype=dtype)
        checkpoint = "StanfordAIMI/CheXagent-2-3b"
        print(f"=> Loading CheXagent3 model ({checkpoint})")
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.checkpoint, device_map=device, trust_remote_code=True
        )
        self.model = self.model.to(self.dtype)
        self.model.eval()

    def get_prompt(self, question, options):
        choice_style = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
        option_str = ", ".join([f"({choice_style[option_idx]}) {option}" for option_idx, option in enumerate(options)])
        option_notations = ", ".join([f"({choice_style[option_idx]})" for option_idx, option in enumerate(options)])
        prompt = f'{question}\nOptions: {option_str}\n' \
                 f'Directly answer the question by choosing one option: {option_notations}.'
        return prompt

    def process_img(self, paths):
        if isinstance(paths, str):
            paths = paths.split("|")
        return paths

    def generate(self, paths, prompt, **kwargs):
        paths = self.process_img(paths)
        query = self.tokenizer.from_list_format([*[{'image': path} for path in paths], {'text': prompt}])
        conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
        input_ids = self.tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
        output = self.model.generate(
            input_ids.to(self.device),
            pad_token_id=self.tokenizer.pad_token_id,
            do_sample=kwargs.get("do_sample", False),
            num_beams=kwargs.get("num_beams", 1),
            temperature=kwargs.get("temperature", 1.),
            top_p=kwargs.get("top_p", 1.),
            use_cache=True,
            max_new_tokens=512
        )[0]
        response = self.tokenizer.decode(output[input_ids.size(1):-1])
        return response


if __name__ == '__main__':
    model = CheXagent()
