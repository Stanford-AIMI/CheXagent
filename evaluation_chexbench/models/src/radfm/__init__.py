import torch
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile
from torchvision import transforms
from transformers import LlamaTokenizerFast

from .Model.RadFM.multimodality_model import MultiLLaMAForCausalLM
from .Model.RadFM.multimodality_model import MultiLLaMAForCausalLM


def get_tokenizer(tokenizer_path, max_img_size=100, image_num=32):
    if isinstance(tokenizer_path, str):
        image_padding_tokens = []
        text_tokenizer = LlamaTokenizerFast.from_pretrained(tokenizer_path)
        text_tokenizer.padding_side = "left"
        special_token = {"additional_special_tokens": ["<image>", "</image>"]}
        for i in range(max_img_size):
            image_padding_token = ""
            for j in range(image_num):
                image_token = "<image" + str(i * image_num + j) + ">"
                image_padding_token = image_padding_token + image_token
                special_token["additional_special_tokens"].append("<image" + str(i * image_num + j) + ">")
            image_padding_tokens.append(image_padding_token)
            text_tokenizer.add_special_tokens(special_token)
            ## make sure the bos eos pad tokens are correct for LLaMA-like models
            text_tokenizer.pad_token_id = 0
            text_tokenizer.bos_token_id = 1
            text_tokenizer.eos_token_id = 2

    return text_tokenizer, image_padding_tokens


def combine_and_preprocess(question, image_list, image_padding_tokens):
    transform = transforms.Compose([
        transforms.RandomResizedCrop([512, 512], scale=(0.8, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
    ])
    images = []
    new_qestions = [_ for _ in question]
    padding_index = 0
    for img in image_list:
        img_path = img['img_path']
        position = img['position']

        if not isinstance(img_path, JpegImageFile):
            image = Image.open(img_path).convert('RGB')
        else:
            image = img_path.convert('RGB')

        image = transform(image)
        image = image.unsqueeze(0).unsqueeze(-1)  # c,w,h,d

        target_H = 512
        target_W = 512
        target_D = 4
        images.append(torch.nn.functional.interpolate(image, size=(target_H, target_W, target_D)))
        new_qestions[position] = "<image>" + image_padding_tokens[padding_index] + "</image>" + new_qestions[position]
        padding_index += 1

    vision_x = torch.cat(images, dim=1).unsqueeze(0)
    text = ''.join(new_qestions)
    return text, vision_x


def create_model_and_transforms(path="pretrained_weights/radfm_weights/pytorch_model.bin"):
    text_tokenizer, image_padding_tokens = get_tokenizer('evaluation_chexbench/models/src/radfm/Language_files/')
    model = MultiLLaMAForCausalLM(lang_model_path='evaluation_chexbench/models/src/radfm/Language_files/')
    ckpt = torch.load(path, map_location='cpu')
    incompatible_keys = model.load_state_dict(ckpt, strict=False)
    print(incompatible_keys)
    return model, text_tokenizer, image_padding_tokens
