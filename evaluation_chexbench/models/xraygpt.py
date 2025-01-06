import re

import torch
from PIL import Image
from rich import print
from transformers import StoppingCriteria, StoppingCriteriaList

from . import BaseLM
from .src.xraygpt.models.mini_gpt4 import MiniGPT4
from .src.xraygpt.processors.blip_processors import Blip2ImageEvalProcessor

MODEL_CFG = {
    'arch': 'mini_gpt4',
    'image_size': 224,
    'drop_path_rate': 0,
    'use_grad_checkpoint': False,
    'vit_precision': 'fp16',
    'freeze_vit': True,
    'freeze_qformer': True,
    'num_query_token': 32,
    'llama_model': 'pretrained_weights/xraygpt_weights/Vicuna_Radiology_fp16/',
    'model_type': 'pretrain_vicuna',
    'max_txt_len': 160,
    'end_sym': '###',
    'low_resource': True,
    'ckpt': 'pretrained_weights/xraygpt_weights/xraygpt_pretrained1.pth',
    'device_8bit': 0}
VIS_CFG = {'name': 'blip2_image_eval', 'image_size': 224}


class XrayGPT(BaseLM):
    def __init__(self, name=f"XrayGPT", device="cuda", dtype=torch.float16):
        super().__init__(name=name, device=device, dtype=dtype)
        checkpoint = "pretrained_weights/xraygpt_weights/Vicuna_Radiology_fp16/"
        print(f"=> Loading XrayGPT model ({checkpoint})")
        self.model = MiniGPT4.from_config(MODEL_CFG).to(device)
        self.vis_processor = Blip2ImageEvalProcessor.from_config(VIS_CFG)

    def get_likelihood_prompt(self, question, options):
        prompts = [f"{question} {c}" for c in options]
        return prompts

    def process_img(self, paths):
        if isinstance(paths, str):
            paths = paths.split("|")
        assert len(paths) == 1, "BLIP2 only support one image."
        image = Image.open(paths[0]).convert('RGB')
        image = self.vis_processor(image).unsqueeze(0).to("cuda:0")
        image_emb, _ = self.model.encode_img(image)
        return image_emb

    def get_logits(self, pixel_values, prompts, options):
        # Format prompt for Xray GPT
        seg_tokens = []
        for prompt in prompts:
            prompt = '<Img><ImageHere></Img> ' + prompt
            prompt_segs = prompt.split('<ImageHere>')
            # only add bos to the first seg
            seg_tokens.append(
                [
                    self.model.llama_tokenizer(
                        seg, return_tensors="pt", add_special_tokens=(i == 0)).to("cuda:0").input_ids
                    for i, seg in enumerate(prompt_segs)
                ]
            )

        # Apply padding and incorporate image embeddings
        num_tokens_per_prompt = [sum([a.shape[1] for a in x]) for x in seg_tokens]
        num_padding_tokens = []
        for i in range(len(prompts)):
            padding_tokens = max(num_tokens_per_prompt) - num_tokens_per_prompt[i]
            num_padding_tokens.append(padding_tokens)
            if padding_tokens == 0:
                continue
            seg_tokens[i][0] = torch.cat((torch.tensor([[2] * padding_tokens]).cuda(), seg_tokens[i][0]), axis=1)
        embs = []
        attn_mask = []
        for i in range(len(prompts)):
            seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens[i]]
            mixed_embs = [emb for pair in zip(seg_embs[:-1], [pixel_values]) for emb in pair] + [seg_embs[-1]]
            mixed_embs = torch.cat(mixed_embs, dim=1)
            attn = torch.ones(mixed_embs.shape[1])
            attn[0:num_padding_tokens[i]] = 0
            embs.append(mixed_embs)
            attn_mask.append(attn)
        embs = torch.cat(embs)
        attn_mask = torch.stack(attn_mask)

        # Compute logits
        with torch.no_grad():
            logits = self.model.llama_model(inputs_embeds=embs, attention_mask=attn_mask).logits.detach()
        output = torch.nn.functional.log_softmax(logits, dim=-1)
        output = output[:, :-1, :]  # Remove last token

        all_outputs = []
        ans_indices = [torch.tensor(self.model.llama_tokenizer(f"{ans}").input_ids[1:]) for ans in options]
        for i in range(len(options)):
            num_ans_tokens = len(ans_indices[i])
            assert (seg_tokens[i][-1].flatten()[-num_ans_tokens:].cpu() == ans_indices[i]).all().item()
            all_outputs.append(output[i, -num_ans_tokens:, :].unsqueeze(0))
        return all_outputs, ans_indices

    def compute_scores(self, likelihood, ans_indices, length_norm=False):
        scores = []
        for i in range(len(ans_indices)):
            a_ids = ans_indices[i].view(1, -1, 1).cuda()
            score = torch.gather(likelihood[i], 2, a_ids).squeeze().sum().item()
            if length_norm: score /= len(ans_indices[i])
            scores.append(score)
        return scores

    def generate(
            self, paths, prompt, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9, repetition_penalty=1.0,
            length_penalty=1, temperature=1.0, max_length=2000
    ):
        class StoppingCriteriaSub(StoppingCriteria):
            def __init__(self, stops=[], encounters=1):
                super().__init__()
                self.stops = stops

            def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
                for stop in self.stops:
                    if torch.all((stop == input_ids[0][-len(stop):])).item():
                        return True

                return False

        prefix = "You are an experienced Doctor, give the following medical scan: <Img>ImageContent</Img>. " \
                 "You will be able to see the medical scan once I provide it to you. Please answer my questions. "

        stop_words_ids = [torch.tensor([835]).to(self.device), torch.tensor([2277, 29937]).to(self.device)]
        stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

        image_emb = self.process_img(paths)
        prompt = prefix + "### Doctor: " + '<Img><ImageHere></Img> ' + prompt + " " + "### Assistant: "
        prompt_segs = prompt.split('<ImageHere>')

        # only add bos to the first seg
        seg_tokens = [
            self.model.llama_tokenizer(seg, return_tensors="pt", add_special_tokens=(i == 0)).to("cuda:0").input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [self.model.llama_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
        img_list = [image_emb]

        embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
        embs = torch.cat(embs, dim=1)

        current_max_len = embs.shape[1] + max_new_tokens
        if current_max_len - max_length > 0:
            print(
                'Warning: The number of tokens in current conversation exceeds the max length. '
                'The model will not see the contexts outside the range.'
            )
        begin_idx = max(0, current_max_len - max_length)
        embs = embs[:, begin_idx:]
        outputs = self.model.llama_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=True,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
        output_token = outputs[0]
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.model.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Doctor:')[-1].strip()
        findings = []
        for sentence in re.sub("\s+", " ", output_text).split(". "):
            if "impression" in sentence.lower():
                break
            else:
                findings.append(sentence)
        output_text = ". ".join(findings)
        if len(output_text) > 0 and output_text[-1] != ".":
            output_text += "."
        return output_text
