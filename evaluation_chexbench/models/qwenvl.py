import torch
from rich import print
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import BaseLM


def make_context(
        tokenizer,
        query,
        history=None,
        system="",
        max_window_size=6144,
        chat_format="chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set(tokenizer.IMAGE_ST)
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set(tokenizer.IMAGE_ST))

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            if turn_response is not None:
                response_text, response_tokens_part = _tokenize_str("assistant", turn_response)
                response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

                next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            else:
                next_context_tokens = nl_tokens + query_tokens + nl_tokens
                prev_chat = f"\n{im_start}{query_text}{im_end}\n"

            current_context_size = (
                    len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        if query is not None:
            context_tokens = system_tokens + context_tokens
            raw_text = f"{im_start}{system_text}{im_end}" + raw_text
            context_tokens += (
                    nl_tokens
                    + im_start_tokens
                    + _tokenize_str("user", query)[1]
                    + im_end_tokens
                    + nl_tokens
                    + im_start_tokens
                    + tokenizer.encode("assistant")
                    + nl_tokens
            )
            raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


class QwenVL(BaseLM):
    def __init__(self, name=f"CheXagent2", device="cuda", dtype=torch.float16):
        super().__init__(name=name, device=device, dtype=dtype)
        checkpoint = "Qwen/Qwen-VL-Chat"
        print(f"\n=> Loading QwenVL model ({checkpoint})")
        self.checkpoint = checkpoint
        self.tokenizer = AutoTokenizer.from_pretrained(self.checkpoint, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(self.checkpoint, device_map="cuda", trust_remote_code=True)
        self.model = self.model.to(self.dtype).to(device)
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
        response, history = self.model.chat(self.tokenizer, query=query, history=None, **kwargs)
        return response
