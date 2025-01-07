<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->

<div align="center">
<h1>
  CheXagent
</h1>
</div>

<p align="center">
📝 <a href="https://arxiv.org/abs/2401.12208" target="_blank">Paper</a> • 🤗 <a href="https://huggingface.co/collections/StanfordAIMI/chexagent-and-its-byproducts-677bd19b15ed5fab582f288a/" target="_blank">Hugging Face</a> • 🧩 <a href="https://github.com/Stanford-AIMI/CheXagent" target="_blank">Github</a> • 🪄 <a href="https://stanford-aimi.github.io/chexagent.html" target="_blank">Project</a>
</p>

<div align="center">
</div>

> Note that the repository and models are only for research purposes and not for clinical use.

## 🤖 Model
CheXagent and its "byproducts" are available on Hugging Face:

| Model            | Link                                                                                                                 | Note                                                                       |
|------------------|----------------------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------|
| CheXagent        | [Huggingface](https://huggingface.co/StanfordAIMI/CheXagent-2-3b)                                                    | The CheXagent model                                                        |
| Vision Encoder   | [Huggingface](https://huggingface.co/collections/StanfordAIMI/chexagent-and-its-byproducts-677bd19b15ed5fab582f288a) | Eight SigLIP/CLIP models of various sizes adapted for CXR                  |
| Language Decoder | [Huggingface](https://huggingface.co/StanfordAIMI/RadPhi-2)                                                          | A language model adapted for clinical use cases (especially for radiology) |

Check out `model_chexagent/chexagent.py` for the following usages:

<details>
<summary>Expand to check the model overview.</summary>

```python
class CheXagent:
    def generate(self, paths, prompt): ...
    def view_classification(self, path): ...
    def view_matching(self, paths): ...
    def binary_disease_classification(self, paths, disease_name): ...
    def disease_identification(self, paths, disease_names): ...
    def findings_generation(self, paths, indication): ...
    def findings_generation_section_by_section(self, paths): ...
    def image_text_matching(self, paths, text): ...
    def plot_image(self, path, response, save_path): ...
    def phrase_grounding(self, path, phrase, save_path): ...
    def abnormality_detection(self, path, disease_name, save_path): ...
    def chest_tube_detection(self, path, save_path): ...
    def rib_fracture_detection(self, path, save_path): ...
    def foreign_objects_detection(self, path, save_path): ...
    def temporal_image_classification(self, paths, disease_name): ...
    def findings_summarization(self, findings): ...
    def named_entity_recognition(self, text): ...
```

</details>

## 🎬 Get started
![Get Started (CheXagent)](assets/chexagent_intro.gif)

Run the script to play with CheXagent:
```shell
python demos/demos/run_examples.py
```
or run the following command to interact with CheXagent through a web demo hosted by Gradio:

```shell
python demos/app_demos.py
```

## 📚 Data

Run the following command to compile the CheXinstruct dataset:

```shell
python data_chexinstruct/compile_chexinstruct.py
```

Run the following command to visualize the CheXinstruct dataset:

```shell
python data_chexinstruct/dataset_visualizer.py
```

## ✨ Evaluation

The scripts for evaluating FMs on CheXbench are in

```shell
+--evaluation_chexbench
|+--axis_1_image_perception
|+--axis_2_image_text_reasoning
|+--axis_3_text_generation
```

**To check whether your environment is set up in the same way as ours, run the following command:**

<details>
<summary>Expand to check the command.</summary>

Run
```shell
python evaluation_chexbench/axis_3_text_generation/run_findings_generation.py
```
and the result should be close to the following

| Macro F1 (14) | Micro F1 (14) | Macro F1 (5) | Micro F1 (5) |  Avg |
|:-------------:|:-------------:|:------------:|:------------:|:----:|
|      44.9     |      58.0     |     55.3     |     62.5     | 55.2 |

</details>

## 🩺 Clinical Reader Study

We provide the reader study interface implementation for future research in this area:

```shell
python reader_study/app.py
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
