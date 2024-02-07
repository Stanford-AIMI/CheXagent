<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
  CheXagent
</h1>
</div>

<p align="center">
📝 <a href="https://arxiv.org/abs/2401.12208" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/StanfordAIMI/CheXagent-8b/" target="_blank">Hugging Face</a> • 🧩 <a href="https://github.com/Stanford-AIMI/CheXagent" target="_blank">Github</a> • 🪄 <a href="https://stanford-aimi.github.io/chexagent.html" target="_blank">Project</a>
</p>

<div align="center">
</div>

## ✨ Latest News

- [12/15/2023]: Model released in [Hugging Face](https://huggingface.co/StanfordAIMI/CheXagent-8b/).

## 🎬 Get Started

```python
import io

import requests
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor, GenerationConfig

# step 1: Setup constant
device = "cuda"
dtype = torch.float16

# step 2: Load Processor and Model
processor = AutoProcessor.from_pretrained("StanfordAIMI/CheXagent-8b", trust_remote_code=True)
generation_config = GenerationConfig.from_pretrained("StanfordAIMI/CheXagent-8b")
model = AutoModelForCausalLM.from_pretrained("StanfordAIMI/CheXagent-8b", torch_dtype=dtype, trust_remote_code=True)

# step 3: Fetch the images
image_path = "https://upload.wikimedia.org/wikipedia/commons/3/3b/Pleural_effusion-Metastatic_breast_carcinoma_Case_166_%285477628658%29.jpg"
images = [Image.open(io.BytesIO(requests.get(image_path).content)).convert("RGB")]

# step 4: Generate the Findings section
prompt = f'Describe "Airway"'
inputs = processor(images=images, text=f" USER: <s>{prompt} ASSISTANT: <s>", return_tensors="pt").to(device=device, dtype=dtype)
output = model.generate(**inputs, generation_config=generation_config)[0]
response = processor.tokenizer.decode(output, skip_special_tokens=True)
```

## ✏️ Citation

```
@article{chexagent-2024,
  title={CheXagent: Towards a Foundation Model for Chest X-Ray Interpretation},
  author={Chen, Zhihong and Varma, Maya and Delbrouck, Jean-Benoit and Paschali, Magdalini and Blankemeier, Louis and Veen, Dave Van and Valanarasu, Jeya Maria Jose and Youssef, Alaa and Cohen, Joseph Paul and Reis, Eduardo Pontes and Tsai, Emily B. and Johnston, Andrew and Olsen, Cameron and Abraham, Tanishq Mathew and Gatidis, Sergios and Chaudhari, Akshay S and Langlotz, Curtis},
  journal={arXiv preprint arXiv:2401.12208},
  url={https://arxiv.org/abs/2401.12208},
  year={2024}
}
```
