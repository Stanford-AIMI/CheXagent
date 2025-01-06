import os
import tempfile
from threading import Thread

import gradio as gr
import torch
from PIL import ImageDraw
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TextIteratorStreamer

title_html = """
<div align="center">
    <h1>
        Demo (CheXagent) (for research purposes only)
    </h1>
</div>

<div align="center">
</div>
"""

separator_html = """
<hr/>
"""


def print_and_write(text):
    print(text)
    with open("log.txt", "at") as f:
        f.write(text)


def clean_text(text):
    text = text.replace("</s>", "")
    return text


@torch.no_grad()
def response_report_generation(pil_image_1, pil_image_2, request: gr.Request):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    paths = []
    if pil_image_1 is not None:
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_path = "tmp" + temp_file.name + ".png"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        pil_image_1.save(temp_path)
        path = temp_path
        paths.append(path)
    if pil_image_2 is not None:
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_path = "tmp" + temp_file.name + ".png"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        pil_image_2.save(temp_path)
        path = temp_path
        paths.append(path)

    # Step 1: Findings Generation
    anatomies = [
        "Airway", "Breathing", "Cardiac",
        "Diaphragm",
        "Everything else (e.g., mediastinal contours, bones, soft tissues, tubes, valves, and pacemakers)"
    ]
    prompts = [f'Please provide a detailed description of "{anatomy}" in the chest X-ray' for anatomy in anatomies]
    anatomies = ["View"] + anatomies
    prompts = ["Determine the view of this CXR"] + prompts

    findings = ""
    partial_message = "## Generating Findings (step-by-step):\n\n"
    for anatomy_idx, (anatomy, prompt) in enumerate(zip(anatomies, prompts)):
        query = tokenizer.from_list_format([*[{'image': path} for path in paths], {'text': prompt}])
        conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
        input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
        generate_kwargs = dict(
            input_ids=input_ids.to(model.device),
            do_sample=False,
            num_beams=1,
            temperature=1,
            top_p=1.,
            use_cache=True,
            max_new_tokens=512,
            streamer=streamer
        )
        t = Thread(target=model.generate, kwargs=generate_kwargs)
        t.start()
        partial_message += f'<span style="color:blue">**Step {anatomy_idx}**: Analyzing {anatomy}...</span> \n\n'
        for new_token in streamer:
            if anatomy_idx != 0:
                findings += new_token
            partial_message += new_token
            yield clean_text(partial_message)
        partial_message += "\n\n"
        findings += " "
    findings = findings.strip().replace("</s>", "")

    # Step 2: Impression Generation
    impression = ""
    partial_message += "## Generating Impression\n\n"
    prompt = f'Write the Impression section for the following Findings: {findings}'
    query = tokenizer.from_list_format([{'text': prompt}])
    conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
    input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
    generate_kwargs = dict(
        input_ids=input_ids.to(model.device),
        do_sample=False,
        num_beams=1,
        temperature=1.,
        top_p=1.,
        use_cache=True,
        repetition_penalty=1.,
        max_new_tokens=512,
        streamer=streamer
    )

    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    for new_token in streamer:
        impression += new_token
        partial_message += new_token
        yield clean_text(partial_message)
    partial_message += "\n\n"
    impression = impression.strip().replace("</s>", "")
    print_and_write(f"Findings: {findings}")
    print_and_write(f"Impression: {impression}")
    print_and_write("\n\n")


@torch.no_grad()
def response_phrase_grounding(pil_image, text, request: gr.Request):
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    if pil_image is not None:
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_path = "tmp" + temp_file.name + ".png"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        pil_image.save(temp_path)
        paths = [temp_path]
    else:
        paths = []

    query = tokenizer.from_list_format([*[{'image': path} for path in paths], {'text': text}])
    conv = [{"from": "system", "value": "You are a helpful assistant."}, {"from": "human", "value": query}]
    input_ids = tokenizer.apply_chat_template(conv, add_generation_prompt=True, return_tensors="pt")
    generate_kwargs = dict(
        input_ids=input_ids.to(model.device),
        do_sample=False,
        num_beams=1,
        temperature=1,
        top_p=1.,
        use_cache=True,
        max_new_tokens=512,
        streamer=streamer
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()
    response = ""
    partial_message = f"## User: \n{text}\n\n## CheXagent: \n"
    for new_token in streamer:
        partial_message += new_token
        response += new_token
        yield partial_message.replace("</s>", ""), None
    partial_message += "\n\n"

    print_and_write(f"Response: {partial_message}")
    print_and_write("\n\n")

    # draw bounding box
    boxes = [_dict["box"] for _dict in tokenizer.to_list_format(response) if "box" in _dict]
    boxes = [[int(cord) / 100 for cord in box.replace("(", "").replace(")", "").split(",")] for box in boxes]
    w = pil_image.width
    h = pil_image.height
    draw = ImageDraw.Draw(pil_image)
    for box in boxes:
        draw.rectangle((box[0] * w, box[1] * h, box[2] * w, box[3] * h), width=10, outline="#FF6969")
    yield partial_message, pil_image.convert("RGB")


def main():
    with gr.Blocks() as structured_report_generation_demo:
        gr.HTML(title_html)
        gr.HTML(separator_html)
        gr.Interface(
            fn=response_report_generation,
            inputs=[
                gr.Image(label="Input Image 1", image_mode="L", type="pil"),
                gr.Image(label="Input Image 2", image_mode="L", type="pil")
            ],
            outputs=gr.Markdown(label="Output"),
            analytics_enabled=True,
            concurrency_limit=1,
        )

    with gr.Blocks() as visual_grounding_demo:
        gr.HTML(title_html)
        gr.HTML(separator_html)
        gr.Interface(
            fn=response_phrase_grounding,
            inputs=[gr.Image(label="Input", image_mode="L", type="pil"),
                    gr.Textbox(value="Please locate the following phrase: ")],
            outputs=[gr.Markdown(label="Output"),
                     gr.Image(label="Output", image_mode="RGB", type="pil")],
            analytics_enabled=True,
            concurrency_limit=1,
        )

    demo = gr.TabbedInterface(
        [structured_report_generation_demo, visual_grounding_demo],
        ["Structured Report Generation", "Visual Grounding"]
    )
    demo.launch(server_name="0.0.0.0", server_port=8888)


if __name__ == '__main__':
    model_name = "StanfordAIMI/CheXagent-2-3b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    model = model.to(torch.float16).to("cuda")
    model = model.eval()
    main()
