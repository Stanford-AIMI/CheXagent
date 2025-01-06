import random

import numpy as np

MIMIC_DIAGNOSIS2LABEL = {
    'Atelectasis': 0,
    'Consolidation': 1,
    'Edema': 2,
    'Enlarged Cardiomediastinum': 3,
    'Lung Lesion': 4,
    'Lung Opacity': 5,
    'Pleural Effusion': 6,
    'Pleural Other': 7,
    'Pneumonia': 8,
    'Pneumothorax': 9
}
MIMIC_LABEL2DIAGNOSIS = {v: k for k, v in MIMIC_DIAGNOSIS2LABEL.items()}


def add_choice_styles(options):
    choice_styles = [
        [chr(i) for i in range(ord('A'), ord('Z') + 1)],
        [chr(i) for i in range(ord('a'), ord('z') + 1)],
        [i for i in range(1, 26 + 1)],
    ]
    choices = random.choice(choice_styles)
    random.shuffle(choices)
    options = {
        f'({choices[i]}) {k}': v for i, (k, v) in enumerate(options.items())
    }
    return options


def create_template(name):
    name2func = {
        "[Abnormality Detection] [VinDr-CXR]": abnormality_detection_vindr_cxr,
        "[Abnormality Detection] [VinDr-PCXR]": abnormality_detection_vindr_pcxr,
        "[Abnormality Grounding] [VinDr-CXR]": abnormality_grounding_vindr_cxr,
        "[Abnormality Grounding] [VinDr-PCXR]": abnormality_grounding_vindr_pcxr,
        "[Caption Generation] [ROCO]": caption_generation_roco,
        "[Chest Tube Segmentation] [Candid-PTX]": chest_tube_segmentation_candid,
        "[Close-Ended VQA] [MIMIC-CXR-VQA]": close_ended_vqa_mimic_cxr_vqa,
        "[Close-Ended VQA] [MedVQA-2019]": close_ended_vqa_medvqa2019,
        "[Close-Ended VQA] [PMC-VQA]": close_ended_vqa_pmc_vqa,
        "[Close-Ended VQA] [Rad-Restruct]": close_ended_vqa_rad_restruct,
        "[Close-Ended VQA] [SLAKE]": close_ended_vqa_slake,
        "[Close-Ended VQA] [VQA-RAD]": close_ended_vqa_vqarad,
        "[Difference VQA] [MIMIC-Diff-VQA]": difference_vqa_mimic_diff_vqa,
        "[Findings Generation] [MIMIC-CXR]": findings_generation_mimic_cxr,
        "[Findings Generation] [CheXpert-Public]": findings_generation_mimic_cxr_struct,
        "[Findings Generation with Indication] [CheXpert-Public]": findings_generation_with_indication_chexpert_struct,
        "[Findings Generation] [MIMIC-CXR-Struct]": findings_generation_mimic_cxr_struct,
        "[Findings Generation with Indication] [MIMIC-CXR-Struct]": findings_generation_with_indication_mimic_cxr_struct,
        "[Findings Generation] [OpenI]": findings_generation_openi,
        "[Findings Summarization] [MIMIC-III]": findings_summarization_mimic_iii,
        "[Findings Summarization] [MIMIC-CXR]": findings_summarization_mimic_cxr,
        "[Findings Summarization] [MIMIC-CXR-Struct]": findings_summarization_mimic_cxr_struct,
        "[Findings Summarization] [OpenI]": findings_summarization_openi,
        "[Foreign Object Detection] [Object-CXR]": foreign_objects_detection_objectcxr,
        "[Grounded Captioning] [MS-CXR]": grounded_captioning_ms_cxr,
        "[Grounded Diagnosis] [MS-CXR]": grounded_diagnosis_ms_cxr,
        "[Grounded Diagnosis] [VinDr-CXR]": grounded_diagnosis_vindr_cxr,
        "[Grounded Diagnosis] [VinDr-PCXR]": grounded_diagnosis_vindr_pcxr,
        "[Grounded Phrase Extraction] [MS-CXR]": grounded_phrase_extraction_ms_cxr,
        "[Image Classification] [Brax]": image_classification_brax,
        "[Image Classification] [COVIDX-CXR-3]": image_classification_covidcxr3,
        "[Image Classification] [CXR-LT]": image_classification_cxr_lt,
        "[Image Classification] [CheXpert-Public]": image_classification_chexpert_public,
        "[Image Classification] [ChestXray14]": image_classification_chestxray14,
        "[Image Classification] [MIMIC-CXR]": image_classification_mimic_cxr,
        "[Image Classification] [NLM-TB]": image_classification_nlm_tb,
        "[Image Classification] [PadChest]": image_classification_padchest,
        "[Image Classification] [RSNA]": image_classification_rsna,
        "[Image-Text Matching] [MIMIC-CXR]": image_text_matching_mimic_cxr,
        "[Image-Text Matching] [ROCO]": image_text_matching_roco,
        "[Image-Text Selection] [MIMIC-CXR]": image_text_selection_mimic_cxr,
        "[Image-Text Selection] [ROCO]": image_text_selection_roco,
        "[Impression Generation] [CXR-PRO]": impression_generation_cxrpro,
        "[Impression Generation] [Candid-PTX]": impression_generation_candid,
        "[Impression Generation] [InterMountain]": impression_generation_intermountain,
        "[Impression Generation] [CheXpert-Public]": impression_generation_mimic_cxr_struct,
        "[Impression Generation] [MIMIC-CXR-Struct]": impression_generation_mimic_cxr_struct,
        "[Impression Generation] [MIMIC-CXR]": impression_generation_mimic_cxr,
        "[Impression Generation] [OpenI]": impression_generation_openi,
        "[Impression Generation] [BIMCV-COVID19]": impression_generation_bimcv_covid19,
        "[Impression Generation] [PadChest]": impression_generation_padchest,
        "[Impression Generation with Indication] [CheXpert-Public]": impression_generation_with_indication_chexpert_struct,
        "[Impression Generation with Indication] [MIMIC-CXR-Struct]": impression_generation_with_indication_mimic_cxr_struct,
        "[Local Findings Generation] [CheXpert-Public]": local_findings_generation_mimic_cxr_struct,
        "[Local Findings Generation] [MIMIC-CXR-Struct]": local_findings_generation_mimic_cxr_struct,
        "[Local Impression Generation] [CheXpert-Public]": local_impression_generation_mimic_cxr_struct,
        "[Local Impression Generation] [MIMIC-CXR-Struct]": local_impression_generation_mimic_cxr_struct,
        "[Local Progression Findings Generation] [CheXpert-Public]": local_progression_findings_generation_mimic_cxr_struct,
        "[Local Progression Findings Generation] [MIMIC-CXR-Struct]": local_progression_findings_generation_mimic_cxr_struct,
        "[Local Progression Impression Generation] [CheXpert-Public]": local_progression_impression_generation_mimic_cxr_struct,
        "[Local Progression Impression Generation] [MIMIC-CXR-Struct]": local_progression_impression_generation_mimic_cxr_struct,
        "[Named Entity Recognition] [RadGraph]": named_entity_recognition_radgraph,
        "[Natural Language Explanation] [MIMIC-NLE]": natural_language_explanation_mimic_nle,
        "[Natural Language Inference] [RadNLI]": natural_language_inference_radnli,
        "[Open-Ended VQA] [MIMIC-CXR-VQA]": open_ended_vqa_mimic_cxr_vqa,
        "[Open-Ended VQA] [MedVQA-2019]": open_ended_vqa_medvqa2019,
        "[Open-Ended VQA] [PMC-VQA]": open_ended_vqa_pmc_vqa,
        "[Open-Ended VQA] [Rad-Restruct]": open_ended_vqa_rad_restruct,
        "[Open-Ended VQA] [SLAKE]": open_ended_vqa_slake,
        "[Open-Ended VQA] [VQA-RAD]": open_ended_vqa_vqarad,
        "[Phrase Extraction and Grounding] [MS-CXR]": phrase_extraction_and_grounding_ms_cxr,
        "[Phrase Grounding] [MS-CXR]": phrase_grounding_ms_cxr,
        "[Pneumothorax Segmentation] [Candid-PTX]": pneumothorax_segmentation_candid,
        "[Pneumothorax Segmentation] [SIIM]": pneumothorax_segmentation_siim,
        "[Progression Findings Generation] [MIMIC-CXR]": progression_findings_generation_mimic_cxr,
        "[Progression Findings Generation] [CheXpert-Public]": progression_findings_generation_mimic_cxr_struct,
        "[Progression Findings Generation] [MIMIC-CXR-Struct]": progression_findings_generation_mimic_cxr_struct,
        "[Progression Impression Generation] [MIMIC-CXR]": progression_impression_generation_mimic_cxr,
        "[Progression Impression Generation] [CheXpert-Public]": progression_impression_generation_mimic_cxr_struct,
        "[Progression Impression Generation] [MIMIC-CXR-Struct]": progression_impression_generation_mimic_cxr_struct,
        "[Report Evaluation] [ReXVal]": report_evaluation_rexval,
        "[Rib Fracture Segmentation] [Candid-PTX]": rib_fracture_segmentation_candid,
        "[Temporal Image Classification] [MS-CXR-T]": temporal_image_classification_ms_cxr_t,
        "[Temporal Sentence Similarity] [MS-CXR-T]": temporal_sentence_similarity_ms_cxr_t,
        "[Text QA] [RadQA]": text_qa_radqa,
        "[View Classification] [MIMIC-CXR]": view_classification_mimic_cxr,
        "[View Matching] [MIMIC-CXR]": view_matching_mimic_cxr,
        "[View Classification] [CheXpert-Public]": view_classification_mimic_cxr,
    }
    return name2func[name]


def abnormality_detection_vindr_cxr(instruct=True):
    prompt_templates = [
        "Detect {disease} in the given image.",
        "Locate areas in the chest X-ray where {disease} are present, using bounding box coordinates",
        "Perform abnormality detection (in the bounding box format) for the given image.",
        "Find the locations of {disease} in the bounding box format for the given image.",
        "Locate {disease} for the given image.",
        "Examine the chest X-ray and mark the regions affected by {disease} with bounding boxes",
        "Detect the following in the image: {disease}.",
        "Examine the image for regions affected by {disease}, and indicate their positions with bounding boxes.",
        "Perform detection for {disease}.",
        "Abnormality Detection (VinDr-CXR)",
    ]
    target_templates = ["{boxes}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(disease, boxes):
            return {
                "q": prompt_template.format(disease=disease),
                "a": target_template.format(boxes=boxes)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda disease, boxes: [func(disease, boxes) for func in form_qas[:-1]][:5]


def abnormality_detection_vindr_pcxr(instruct=True):
    prompt_templates = [
        "Detect {disease} in the given image.",
        "Locate areas in the chest X-ray where {disease} are present, using bounding box coordinates",
        "Perform abnormality detection (in the bounding box format) for the given image.",
        "Find the locations of {disease} in the bounding box format for the given image.",
        "Locate {disease} for the given image.",
        "Examine the chest X-ray and mark the regions affected by {disease} with bounding boxes",
        "Detect the following in the image: {disease}.",
        "Examine the image for regions affected by {disease}, and indicate their positions with bounding boxes.",
        "Perform detection for {disease}.",
        "Abnormality Detection (VinDr-PCXR).",
    ]
    target_templates = ["{boxes}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(disease, boxes):
            return {
                "q": prompt_template.format(disease=disease),
                "a": target_template.format(boxes=boxes)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda disease, boxes: [func(disease, boxes) for func in form_qas[:-1]][:5]


def abnormality_grounding_vindr_cxr(instruct=True):
    prompt_templates = [
        "Detect {disease} in the given image.",
        "Locate areas in the chest X-ray where {disease} is present, using bounding box coordinates",
        "Localize {disease} in the bounding box format for the given image.",
        "Find the locations of {disease} in the bounding box format for the given image.",
        "Locate {disease} for the given image.",
        "Examine the chest X-ray and mark the regions affected by {disease} with bounding boxes",
        "Detect the following in the image: {disease}.",
        "Examine the image for regions affected by {disease}, and indicate their positions with bounding boxes.",
        "Perform detection for {disease}.",
        "Abnormality Grounding (VinDr-CXR): {disease}.",
    ]
    target_templates = ["{boxes}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(disease, boxes):
            return {
                "q": prompt_template.format(disease=disease.lower()),
                "a": target_template.format(boxes=boxes)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda disease, boxes: [func(disease, boxes) for func in form_qas[:-1]][:5]


def abnormality_grounding_vindr_pcxr(instruct=True):
    prompt_templates = [
        "Detect {disease} in the given image.",
        "Locate areas in the chest X-ray where {disease} is present, using bounding box coordinates",
        "Localize {disease} in the bounding box format for the given image.",
        "Find the locations of {disease} in the bounding box format for the given image.",
        "Locate {disease} for the given image.",
        "Examine the chest X-ray and mark the regions affected by {disease} with bounding boxes",
        "Detect the following in the image: {disease}.",
        "Examine the image for regions affected by {disease}, and indicate their positions with bounding boxes.",
        "Perform detection for {disease}.",
        "Abnormality Grounding (VinDr-PCXR): {disease}",
    ]
    target_templates = ["{boxes}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(disease, boxes):
            return {
                "q": prompt_template.format(disease=disease.lower()),
                "a": target_template.format(boxes=boxes)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda disease, boxes: [func(disease, boxes) for func in form_qas[:-1]][:5]


def caption_generation_roco(instruct=True):
    prompt_templates = [
        "You are provided with an image from a scientific article. Please write a figure caption for it.",
        "Please write a figure caption for the given image.",
        "Write a caption for the given figure.",
        "Given a figure from PubMed Central, please write a caption for it.",
        "Write a PMC-style caption for the given figure.",
        "Write a academic-style caption for the given figure.",
        "Create a caption in academic tone for the figure.",
        "Provide a possible caption for this figure.",
        "Generate a caption for this figure.",
        "Caption Generation (ROCO)",
    ]
    target_templates = ["{report}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(report):
            return {
                "q": prompt_template,
                "a": target_template.format(report=report.strip())
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda report: [func(report) for func in form_qas[:-1]][:5]


def chest_tube_segmentation_candid(instruct=True):
    prompt_templates = [
        "Detect chest tubes in the given image.",
        "Perform chest tube detection for the given image.",
        "Locate chest tubes and specify their positions with bounding box coordinates",
        "Identify any chest tubes in the chest X-ray and provide the associated bounding box coordinates.",
        "Locate chest tubes in the given image.",
        "Find chest tubes in the given image.",
        "Detect the following in the image: chest tubes.",
        "Locate the following in the image: chest tubes.",
        "In the provided chest X-ray, identify bounding box coordinates for each chest tube.",
        "Chest Tube Detection (Candid)",
    ]
    target_templates = ["{boxes}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(disease, boxes):
            return {
                "q": prompt_template.format(disease=disease),
                "a": target_template.format(boxes=boxes)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda disease, boxes: [func(disease, boxes) for func in form_qas[:-1]][:5]


def close_ended_qa_task(instruct=True):
    prompt_template_1 = "{question} Options:\n{options}"
    target_template_1 = "{answer}"

    def create_qa(question, options):
        options = add_choice_styles(options)
        items = list(options.items())
        random.shuffle(items)
        options = dict(items)
        options_q = list(options.keys())
        options_a = [k for k, v in options.items() if v >= 1]
        suffix = ""
        if random.random() > 0.5:
            options_a = [option.split(" ")[0] for option in options_a]
            suffix = "\nAnswer with the option’s letter/number from the given choices directly."
        q = prompt_template_1.format(question=question, options="\n".join(options_q)) + suffix
        a = target_template_1.format(answer=", ".join(options_a))
        return {"q": q, "a": a}

    form_qas = [create_qa]
    return random.choice(form_qas) if instruct else form_qas[-1]


def close_ended_vqa_mimic_cxr_vqa(instruct=True):
    return close_ended_qa_task(instruct)


def close_ended_vqa_medvqa2019(instruct=True):
    return close_ended_qa_task(instruct)


def close_ended_vqa_pmc_vqa(instruct=True):
    return close_ended_qa_task(instruct)


def close_ended_vqa_rad_restruct(instruct=True):
    return close_ended_qa_task(instruct)


def close_ended_vqa_slake(instruct=True):
    return close_ended_qa_task(instruct)


def close_ended_vqa_vqarad(instruct=True):
    return close_ended_qa_task(instruct)


def difference_vqa_mimic_diff_vqa(instruct=True):
    return open_ended_qa_task(instruct)


def findings_generation_mimic_cxr(instruct=True):
    prompt_templates = [
        "Write an example findings section for the CXR",

        "You are provided with one or multiple chest X-ray image(s). "
        "Write an example findings section for the diagnostic report.",

        "You are provided with one or multiple chest X-ray image(s). "
        "Please provide an example findings section to detail the key findings.",

        "Examine the chest X-ray thoroughly and write an example findings section of the diagnostic report.",

        "Write an example Findings section in the clinical style for the provided CXR",

        "Assess the chest X-ray, identify key findings in the CXR and write an example findings section.",

        "For the given images, write an example Findings section that may exist in clinical practice.",

        "Write an example Findings section for the given images as if you are a radiologist.",

        "Write an example Findings section similar to the Findings sections in clinical practice.",

        "Findings Generation (MIMIC-CXR)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(findings):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=findings)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda findings: [func(findings) for func in form_qas[:-1]][:5]


def findings_generation_mimic_cxr_struct(instruct=True):
    prompt_templates = [
        "Write a findings section for the CXR.",

        "Write a structured findings section for the CXR.",

        "You are provided with one or multiple chest X-ray image(s). "
        "Write a structured findings section for the diagnostic report.",

        "You are provided with one or multiple chest X-ray image(s). "
        "Please provide a structured findings section to detail the key findings.",

        "Examine the chest X-ray thoroughly and write a structured findings section of the diagnostic report.",

        "Assess the chest X-ray, identify key findings in the CXR and write a structured findings section.",

        "Write a structured Findings section for the given images as if you are a radiologist.",

        "You are provided with one or multiple chest X-ray image(s). "
        "Write a structured Findings section with subsections "
        "for each anatomical feature and with the positive findings highlighted.",

        "Write a structured Findings section for the provided CXR with subsections for each anatomical feature. "
        "Highlight positive findings.",

        "Findings Generation (MIMIC-CXR-Struct)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(findings):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=findings),
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda findings: [func(findings) for func in form_qas[:-1]][:5]


def findings_generation_with_indication_chexpert_struct(instruct=True):
    prompt_templates = [
        'Given the clinical history: "{indication}", write a findings section for the CXR.',

        'Given the clinical history: "{indication}", write a structured findings section for the CXR.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the clinical history: "{indication}", write a structured findings section for the diagnostic report.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the clinical history: "{indication}", please provide a structured findings section to detail the key findings.',

        'Given the clinical history: "{indication}", '
        'examine the chest X-ray thoroughly and write a structured findings section of the diagnostic report.',

        'Given the clinical history: "{indication}", '
        'assess the chest X-ray, identify key findings in the CXR and write a structured findings section.',

        'Given the clinical history: "{indication}", '
        'write a structured Findings section for the given images as if you are a radiologist.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the clinical history: "{indication}", write a structured Findings section with subsections '
        'for each anatomical feature and with the positive findings highlighted.',

        'Given the clinical history: "{indication}", '
        'write a structured Findings section for the provided CXR with subsections for each anatomical feature. '
        'Highlight positive findings.',

        'Findings Generation with Indication (MIMIC-CXR-Struct): "{indication}"',
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(indication, findings):
            return {
                "q": prompt_template.format(indication=indication),
                "a": target_template.format(answer=findings),
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda indication, findings: [func(indication, findings) for func in form_qas[:-1]][:5]


def findings_generation_with_indication_mimic_cxr_struct(instruct=True):
    prompt_templates = [
        'Given the indication: "{indication}", write a findings section for the CXR.',

        'Given the indication: "{indication}", write a structured findings section for the CXR.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the indication: "{indication}", write a structured findings section for the diagnostic report.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the indication: "{indication}", please provide a structured findings section to detail the key findings.',

        'Given the indication: "{indication}", '
        'examine the chest X-ray thoroughly and write a structured findings section of the diagnostic report.',

        'Given the indication: "{indication}", '
        'assess the chest X-ray, identify key findings in the CXR and write a structured findings section.',

        'Given the indication: "{indication}", '
        'write a structured Findings section for the given images as if you are a radiologist.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the indication: "{indication}", write a structured Findings section with subsections '
        'for each anatomical feature and with the positive findings highlighted.',

        'Given the indication: "{indication}", '
        'write a structured Findings section for the provided CXR with subsections for each anatomical feature. '
        'Highlight positive findings.',

        'Findings Generation with Indication (MIMIC-CXR-Struct): "{indication}"',
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(indication, findings):
            return {
                "q": prompt_template.format(indication=indication),
                "a": target_template.format(answer=findings),
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda indication, findings: [func(indication, findings) for func in form_qas[:-1]][:5]


def findings_generation_openi(instruct=True):
    prompt_templates = [
        "Write an example findings section for the CXR",

        "You are provided with one or multiple chest X-ray image(s). "
        "Write an example findings section for the diagnostic report.",

        "You are provided with one or multiple chest X-ray image(s). "
        "Please provide an example findings section to detail the key findings.",

        "Examine the chest X-ray thoroughly and write an example findings section of the diagnostic report.",

        "Write an example Findings section in the clinical style for the provided CXR",

        "Assess the chest X-ray, identify key findings in the CXR and write an example findings section.",

        "For the given images, write an example Findings section that may exist in clinical practice.",

        "Write an example Findings section for the given images as if you are a radiologist.",

        "Write an example Findings section similar to the Findings sections in clinical practice.",

        "Findings Generation (OpenI)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(findings):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=findings)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda findings: [func(findings) for func in form_qas[:-1]][:5]


def summarization_task(prompt_templates, instruct=True):
    target_templates = ["{impression}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(findings, impression):
            return {
                "q": prompt_template.format(findings=findings),
                "a": target_template.format(impression=impression)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda findings, impression: [func(findings, impression) for func in form_qas[:-1]][:5]


def findings_summarization_mimic_iii(instruct=True):
    prompt_templates = [
        "You are provided with the findings from an imaging study. Please summarize it and write the impressions section of the diagnostic report: {findings}.",

        "Summarize the following findings: {findings}.",

        "Write the Impression section for the following Findings: {findings}.",

        "Summarize the following: {findings}.",

        "Craft a brief impressions section that captures the main insights from the provided findings in the diagnostic report: {findings}.",

        "Summarize it and write its Impression: {findings}.",

        "Generate the impressions section of a diagnostic report given the following findings: {findings}.",

        "Summarize the findings: {findings}.",

        "'{findings}.' Summarize the key takeaways from the above findings.",

        "Findings Summarization (MIMIC-III): {findings}",
    ]
    return summarization_task(prompt_templates, instruct)


def findings_summarization_mimic_cxr(instruct=True):
    prompt_templates = [
        "You are provided with the findings from an imaging study. Please summarize it and write the impressions " \
        "section of the diagnostic report: {findings}.",

        "Summarize the following Findings: {findings}.",

        "Write the Impression section for the following Findings: {findings}.",

        "Summarize the following: {findings}.",

        "Craft a brief impressions section that captures the main insights from the provided findings in the " \
        "diagnostic report: {findings}.",

        "Summarize it and write its Impression: {findings}.",

        "Generate the impressions section of a diagnostic report given the following findings: {findings}.",

        "Summarize the Findings: {findings}.",

        "'{findings}.' Summarize the key takeaways from the above findings.",

        "Findings Summarization (MIMIC-CXR): {findings}",
    ]
    return summarization_task(prompt_templates, instruct)


def findings_summarization_mimic_cxr_struct(instruct=True):
    prompt_templates = [
        "You are provided with the findings from an imaging study. Please summarize it and write the impressions " \
        "section of the diagnostic report: {findings}.",

        "Summarize the following Findings: {findings}.",

        "Write the Impression section for the following Findings: {findings}.",

        "Summarize the following: {findings}.",

        "Craft a brief impressions section that captures the main insights from the provided findings in the " \
        "diagnostic report: {findings}.",

        "Summarize it and write its Impression: {findings}.",

        "Generate the impressions section of a diagnostic report given the following findings: {findings}.",

        "Summarize the Findings: {findings}.",

        "'{findings}.' Summarize the key takeaways from the above findings.",

        "Findings Summarization (MIMIC-CXR): {findings}",
    ]
    return summarization_task(prompt_templates, instruct)


def findings_summarization_openi(instruct=True):
    prompt_templates = [
        "You are provided with the findings from an imaging study. Please summarize it and write the impressions section of the diagnostic report: {findings}.",
        "Summarize the following Findings: {findings}.",
        "Write the Impression section for the following Findings: {findings}.",
        "Summarize the following: {findings}.",
        "Craft a brief impressions section that captures the main insights from the provided findings in the diagnostic report: {findings}.",
        "Summarize it and write its Impression: {findings}.",
        "Generate the impressions section of a diagnostic report given the following findings: {findings}.",
        "Summarize the Findings: {findings}.",
        "'{findings}'. Summarize the key takeaways from the above findings.",
        "Findings Summarization (OpenI): {findings}",
    ]
    return summarization_task(prompt_templates, instruct)


def foreign_objects_detection_objectcxr(instruct=True):
    prompt_templates = [
        "Detect foreign objects in the given image.",

        "Examine the chest X-ray for the presence of foreign objects, such as tubes, clips, or hardware, " \
        "and provide their locations with bounding box coordinates.",

        "Identify any foreign objects, including tubes, clips, or hardware, in the chest X-ray, and specify " \
        "their positions using bounding boxes",

        "Find the locations of foreign objects in the bounding box format for the given image.",

        "Locate foreign objects in the given CXR.",

        "Identify foreign objects, such as tubes, clips, or hardware, within the chest X-ray, and provide " \
        "bounding box coordinates for each.",

        "Detect the following items in the image: foreign objects.",

        "Locate the following in the CXR image: {object}.",

        "Detect foreign objects.",

        "Foreign Object Detection (Object-CXR)",
    ]
    target_templates = ["{boxes}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(object, boxes):
            return {
                "q": prompt_template.format(object=object),
                "a": target_template.format(boxes=boxes),
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda object, boxes: [func(object, boxes) for func in form_qas[:-1]][:5]


def grounded_captioning_ms_cxr(instruct=True):
    prompt_templates = [
        "Please generate a caption for the following region(s): {boxes}",
        "Describe the content of the following region(s): {boxes}",
    ]
    target_templates = ["{phrase}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(boxes, phrase):
            return {
                "q": prompt_template.format(boxes=boxes),
                "a": target_template.format(phrase=phrase)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda boxes, phrase: [func(boxes, phrase) for func in form_qas][:5]


def grounded_diagnosis_ms_cxr(instruct=True):
    prompt_templates = [
        "Please give the corresponding diagnosis for the following region(s): {boxes}",
        "Provide a diagnosis based on the content of the following region(s): {boxes}",
    ]
    target_templates = ["{diagnosis}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(boxes, diagnosis):
            return {
                "q": prompt_template.format(boxes=boxes),
                "a": target_template.format(diagnosis=diagnosis)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda boxes, diagnosis: [func(boxes, diagnosis) for func in form_qas][:5]


def grounded_diagnosis_vindr_cxr(instruct=True):
    prompt_templates = [
        "Please give the corresponding diagnosis for the following region(s): {boxes}",
        "Provide a diagnosis based on the content of the following region(s): {boxes}",
    ]
    target_templates = ["{diagnosis}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(boxes, diagnosis):
            return {
                "q": prompt_template.format(boxes=boxes),
                "a": target_template.format(diagnosis=diagnosis)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda boxes, diagnosis: [func(boxes, diagnosis) for func in form_qas][:5]


def grounded_diagnosis_vindr_pcxr(instruct=True):
    prompt_templates = [
        "Please give the corresponding diagnosis for the following region(s): {boxes}",
        "Provide a diagnosis based on the content of the following region(s): {boxes}",
    ]
    target_templates = ["{diagnosis}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(boxes, diagnosis):
            return {
                "q": prompt_template.format(boxes=boxes),
                "a": target_template.format(diagnosis=diagnosis)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda boxes, diagnosis: [func(boxes, diagnosis) for func in form_qas][:5]


def grounded_phrase_extraction_ms_cxr(instruct=True):
    prompt_template_1 = "Please extract the phrase for the regions: {boxes} " \
                        "from the following text: {text}"
    target_template_1 = "{phrase}"
    form_qa_1 = lambda boxes, text, phrase: {
        "q": prompt_template_1.format(boxes=boxes, text=text),
        "a": target_template_1.format(phrase=phrase)
    }

    form_qas = [form_qa_1]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda boxes, text, phrase: [func(boxes, text, phrase) for func in form_qas][:5]


def image_classification_brax(instruct=True):
    prompt_templates = [
        "Please review the chest X-ray and classify any diseases from "
        "the following list that are present in the image. Options:\n{options}",

        "Given the CXR, identify any diseases. Options:\n{options}",

        "Identify any diseases visible in the given CXR. Options:\n{options}",

        "Perform disease classification on the given CXR. Options:\n{options}",

        "Which diseases are represented in this image:\n{options}",

        "{options}\nPerform disease classification by matching the observed "
        "conditions in the chest X-ray with the listed options",

        "Perform disease classification given the following label set:\n{options}",

        "Perform disease classification",

        "Identify abnormalities that are present in the CXRs.",

        "Image Classification (BRAX)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options):
            # attach choice styles
            if "options" in prompt_template:
                options = add_choice_styles(options)
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = [k for k, v in options.items() if v >= -1]
            options_a = [k for k, v in options.items() if v >= 1]
            if len(options_q) < 2 or len(options_q) == len(options_a) or random.random() < 0.8:
                options_q = list(options.keys())
            suffix = ""
            if "options" in prompt_template:
                suffix = "\nThere may be more than one answer."
                if random.random() > 0.5:
                    options_a = [option.split(" ")[0] for option in options_a]
                    suffix = "\nThere may be more than one answer. " \
                             "Answer with the options’ letter/number from the given choices directly."
            return {
                "q": prompt_template.format(options="\n".join(options_q)) + suffix,
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa

    prompt_templates_2 = [
        "Which findings are in this chest X-ray? Options:\n{options}",
        "Which diseases are represented in this image:\n{options}",
        "Perform disease classification given the following label set:\n{options}",
    ]
    target_templates_2 = ["{answer}"] * len(prompt_templates_2)

    def create_qa_2(prompt_template, target_template):
        def form_qa_2(options):
            all_labels = [x for x in options if x not in ["Pleural Other", "No Finding"]]
            pos_label = [x for x in options if options[x] == 1 and x in all_labels]
            if len(pos_label) < 1 or len(pos_label) > 5:
                return None
            # construct negative options
            neg_findings = []
            for i in range(3):
                neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                while neg_sample == set(pos_label):
                    neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                neg_findings.append(neg_sample)
            pos_findings = ", ".join(pos_label).lower()
            neg_findings = [", ".join(x).lower() for x in neg_findings]
            # construct options
            options = {pos_findings: 1}
            options.update({item: 0 for item in neg_findings})
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_2

    prompt_templates_3 = [
        "Does this chest X-ray show {disease}?",
        "Does this chest X-ray show {disease}? Options:\n{options}",
    ]
    target_templates_3 = ["{answer}"] * len(prompt_templates_3)

    def create_qa_3(prompt_template, target_template):
        def form_qa_3(options):
            all_labels = [x for x in options if x not in ["Pleural Other", "No Finding"]]
            pos_labels = [x for x in options if options[x] == 1 and x in all_labels]
            neg_labels = [x for x in options if options[x] == 0 and x in all_labels]
            if len(pos_labels) == 0 or len(neg_labels) == 0:
                return None
            if random.random() > 0.5:
                disease = random.choice(pos_labels).lower()
                options = {"Yes": 1, "No": 0}
            else:
                disease = random.choice(neg_labels).lower()
                options = {"Yes": 0, "No": 1}
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            if "options" in prompt_template:
                options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(disease=disease, options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_3

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    form_qas_2 = [create_qa_2(p, t) for p, t in zip(prompt_templates_2, target_templates_2)]
    form_qas_3 = [create_qa_3(p, t) for p, t in zip(prompt_templates_3, target_templates_3)]

    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: (
            [func(options) for func in form_qas[:-1]][:3]
            + [func(options) for func in form_qas_2 if func(options)]
            + [func(options) for func in form_qas_3 if func(options)]
    )


def image_classification_covidcxr3(instruct=True):
    prompt_templates = [
        "Review the provided chest X-ray and classify any diseases from "
        "the following list that are present in the image. Options: {options}",

        "Please determine if the chest X-ray shows signs of Covid-19",

        "Does the CXR show signs of Coronavirus disease 2019?",

        "Detect whether Covid-19 is present in the chest X-ray",

        "Is Covid-19 present in the CXR?",

        "{options}\nIdentify any diseases visible in the given CXRs from the above list. ",

        "Perform disease classification given the following label set: {options}",

        "Determine if the patient has Covid-19.",

        "Is this a CXR with Covid-19?",

        "Image Classification (COVIDX-CXR-3)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)
    answer_templates = [
        lambda options: ", ".join(options),
        lambda options: "Yes" if options[0] == "Covid (positive)" else "No",
        lambda options: "Yes" if options[0] == "Covid (positive)" else "No",
        lambda options: "Yes" if options[0] == "Covid (positive)" else "No",
        lambda options: "Yes" if options[0] == "Covid (positive)" else "No",
        lambda options: ", ".join(options),
        lambda options: ", ".join(options),
        lambda options: "Yes" if options[0] == "Covid (positive)" else "No",
        lambda options: "Yes" if options[0] == "Covid (positive)" else "No",
        lambda options: ", ".join(options),
    ]

    def create_qa(prompt_template, target_template, answer_template):
        def form_qa(options):
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = [k for k, v in options.items() if v >= -1]
            options_a = [k for k, v in options.items() if v >= 1]
            if len(options_q) < 2 or random.random() < 0.8:
                options_q = list(options.keys())
            return {
                "q": prompt_template.format(options=", ".join(options_q)),
                "a": target_template.format(answer=answer_template(options_a))
            }

        return form_qa

    form_qas = [create_qa(p, t, a) for p, t, a in zip(prompt_templates, target_templates, answer_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: [func(options) for func in form_qas[:-1]][:5]


def image_classification_cxr_lt(instruct=True):
    prompt_templates = [
        "Please review the chest X-ray and classify any diseases from "
        "the following list that are present in the image. Options:\n{options}",

        "Given the CXR, identify any diseases. Options:\n{options}",

        "Identify any diseases visible in the given CXR. Options:\n{options}",

        "Perform disease classification on the given CXR. Options:\n{options}",

        "Which diseases are represented in this image:\n{options}",

        "{options}\nPerform disease classification by matching the observed conditions "
        "in the chest X-ray with the diseases listed.",

        "Perform disease classification given the following label set:\n{options}",

        "Perform disease classification."

        "Identify abnormalities that are present in the CXRs.",

        "Image Classification (CXR-LT)"

    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options):
            # attach choice styles
            if "options" in prompt_template:
                options = add_choice_styles(options)
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = [k for k, v in options.items() if v >= -1]
            options_a = [k for k, v in options.items() if v >= 1]
            if len(options_q) < 2 or random.random() < 0.8:
                options_q = list(options.keys())
            suffix = ""
            if "options" in prompt_template:
                suffix = "\nThere may be more than one answer."
                if random.random() > 0.5:
                    options_a = [option.split(" ")[0] for option in options_a]
                    suffix = "\nThere may be more than one answer. " \
                             "Answer with the options’ letter/number from the given choices directly."
            return {
                "q": prompt_template.format(options="\n".join(options_q)) + suffix,
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa

    prompt_templates_2 = [
        "Which findings are in this chest X-ray? Options:\n{options}",
        "Which diseases are represented in this image:\n{options}",
        "Perform disease classification given the following label set:\n{options}",
    ]
    target_templates_2 = ["{answer}"] * len(prompt_templates_2)

    def create_qa_2(prompt_template, target_template):
        def form_qa_2(options):
            all_labels = [x for x in options if x not in ["Pleural Other", "No Finding"]]
            pos_label = [x for x in options if options[x] == 1 and x in all_labels]
            if len(pos_label) < 1 or len(pos_label) > 5:
                return None
            neg_findings = []
            for i in range(3):
                neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                while neg_sample == set(pos_label):
                    neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                neg_findings.append(neg_sample)
            pos_findings = ", ".join(pos_label).lower()
            neg_findings = [", ".join(x).lower() for x in neg_findings]
            # construct options
            options = {pos_findings: 1}
            options.update({item: 0 for item in neg_findings})
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_2

    prompt_templates_3 = [
        "Does this chest X-ray show {disease}?",
        "Does this chest X-ray show {disease}? Options:\n{options}",
    ]
    target_templates_3 = ["{answer}"] * len(prompt_templates_3)

    def create_qa_3(prompt_template, target_template):
        def form_qa_3(options):
            all_labels = [x for x in options if x not in ["Pleural Other", "No Finding"]]
            pos_labels = [x for x in options if options[x] == 1 and x in all_labels]
            neg_labels = [x for x in options if options[x] == 0 and x in all_labels]
            if len(pos_labels) == 0 or len(neg_labels) == 0:
                return None
            if random.random() > 0.5:
                disease = random.choice(pos_labels).lower()
                options = {"Yes": 1, "No": 0}
            else:
                disease = random.choice(neg_labels).lower()
                options = {"Yes": 0, "No": 1}
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            if "options" in prompt_template:
                options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(disease=disease, options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_3

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    form_qas_2 = [create_qa_2(p, t) for p, t in zip(prompt_templates_2, target_templates_2)]
    form_qas_3 = [create_qa_3(p, t) for p, t in zip(prompt_templates_3, target_templates_3)]

    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: (
            [func(options) for func in form_qas[:-1]][:3]
            + [func(options) for func in form_qas_2 if func(options)]
            + [func(options) for func in form_qas_3 if func(options)]
    )


def image_classification_chexpert_public(instruct=True):
    prompt_templates = [
        "Given the CXR, identify any diseases. Options:\n{options}",

        "Identify any diseases visible in the given CXR. Options:\n{options}",

        "Perform disease classification on the given CXR. Options:\n{options}",

        "Which diseases are represented in this image:\n{options}",

        "{options}\nPerform disease classification by matching the observed "
        "conditions in the chest X-ray with the diseases listed",

        "Perform disease classification given the following label set:\n{options}",

        "Perform disease classification",

        "Identify abnormalities that are present in the CXRs.",

        "Image Classification (CheXpert)"
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options):
            # attach choice styles
            if "options" in prompt_template:
                options = add_choice_styles(options)
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = [k for k, v in options.items() if v >= -1]
            options_a = [k for k, v in options.items() if v >= 1]
            if len(options_q) < 2 or random.random() < 0.8:
                options_q = list(options.keys())
            suffix = ""
            if "options" in prompt_template:
                suffix = "\nThere may be more than one answer."
                if random.random() > 0.5:
                    options_a = [option.split(" ")[0] for option in options_a]
                    suffix = "\nThere may be more than one answer. " \
                             "Answer with the options’ letter/number from the given choices directly."
            return {
                "q": prompt_template.format(options="\n".join(options_q)) + suffix,
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa

    prompt_templates_2 = [
        "Which findings are in this chest X-ray? Options:\n{options}",
        "Which diseases are represented in this image:\n{options}",
        "Perform disease classification given the following label set:\n{options}",
    ]
    target_templates_2 = ["{answer}"] * len(prompt_templates_2)

    def create_qa_2(prompt_template, target_template):
        def form_qa_2(options):
            all_labels = [x for x in options if x not in ["Pleural Other", "No Finding"]]
            pos_label = [x for x in options if options[x] == 1 and x in all_labels]
            if len(pos_label) < 1 or len(pos_label) > 5:
                return None
            # construct negative options
            neg_findings = []
            for i in range(3):
                neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                while neg_sample == set(pos_label):
                    neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                neg_findings.append(neg_sample)
            pos_findings = ", ".join(pos_label).lower()
            neg_findings = [", ".join(x).lower() for x in neg_findings]
            # construct options
            options = {pos_findings: 1}
            options.update({item: 0 for item in neg_findings})
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_2

    prompt_templates_3 = [
        "Does this chest X-ray show {disease}?",
        "Does this chest X-ray show {disease}? Options:\n{options}",
    ]
    target_templates_3 = ["{answer}"] * len(prompt_templates_3)

    def create_qa_3(prompt_template, target_template):
        def form_qa_3(options):
            all_labels = [x for x in options if x not in ["Pleural Other", "No Finding"]]
            pos_labels = [x for x in options if options[x] == 1 and x in all_labels]
            neg_labels = [x for x in options if options[x] == 0 and x in all_labels]
            if len(pos_labels) == 0 or len(neg_labels) == 0:
                return None
            if random.random() > 0.5:
                disease = random.choice(pos_labels).lower()
                options = {"Yes": 1, "No": 0}
            else:
                disease = random.choice(neg_labels).lower()
                options = {"Yes": 0, "No": 1}
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            if "options" in prompt_template:
                options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(disease=disease, options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_3

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    form_qas_2 = [create_qa_2(p, t) for p, t in zip(prompt_templates_2, target_templates_2)]
    form_qas_3 = [create_qa_3(p, t) for p, t in zip(prompt_templates_3, target_templates_3)]

    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: (
            [func(options) for func in form_qas[:-1]][:3]
            + [func(options) for func in form_qas_2 if func(options)]
            + [func(options) for func in form_qas_3 if func(options)]
    )


def image_classification_chestxray14(instruct=True):
    prompt_templates = [
        "Please review the chest X-ray and classify any diseases from the following "
        "list that are present in the image. Options:\n{options}",

        "Given the CXR, identify the diseases. Options:\n{options}",

        "Identify any diseases visible in the given CXR. Options:\n{options}",

        "Perform disease classification on the given CXR. Options:\n{options}",

        "Which diseases are represented in this image:\n{options}",

        "{options}\nPerform disease classification by matching the observed conditions "
        "in the chest X-ray with the diseases listed",

        "Perform disease classification given the following label set:\n{options}",

        "Perform disease classification",

        "Identify abnormalities that are present in the CXRs.",

        "Image Classification (ChestXray14)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options):
            # attach choice styles
            if "options" in prompt_template:
                options = add_choice_styles(options)
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = list(options.keys())
            options_a = [k for k, v in options.items() if v >= 1]
            suffix = ""
            if "options" in prompt_template:
                suffix = "\nThere may be more than one answer."
                if random.random() > 0.5:
                    options_a = [option.split(" ")[0] for option in options_a]
                    suffix = "\nThere may be more than one answer. " \
                             "Answer with the options’ letter/number from the given choices directly."
            return {
                "q": prompt_template.format(options="\n".join(options_q)) + suffix,
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa

    prompt_templates_2 = [
        "Which findings are in this chest X-ray? Options:\n{options}",
        "Which diseases are represented in this image:\n{options}",
        "Perform disease classification given the following label set:\n{options}",
    ]
    target_templates_2 = ["{answer}"] * len(prompt_templates_2)

    def create_qa_2(prompt_template, target_template):
        def form_qa_2(options):
            all_labels = [x for x in options if x not in ["no findings"]]
            pos_label = [x for x in options if options[x] == 1 and x in all_labels]
            if len(pos_label) < 1 or len(pos_label) > 5:
                return None
            # construct negative options
            neg_findings = []
            for i in range(3):
                neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                while neg_sample == set(pos_label):
                    neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                neg_findings.append(neg_sample)
            pos_findings = ", ".join(pos_label).lower()
            neg_findings = [", ".join(x).lower() for x in neg_findings]
            # construct options
            options = {pos_findings: 1}
            options.update({item: 0 for item in neg_findings})
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_2

    prompt_templates_3 = [
        "Does this chest X-ray show {disease}?",
        "Does this chest X-ray show {disease}? Options:\n{options}",
    ]
    target_templates_3 = ["{answer}"] * len(prompt_templates_3)

    def create_qa_3(prompt_template, target_template):
        def form_qa_3(options):
            all_labels = [x for x in options if x not in ["no findings"]]
            pos_labels = [x for x in options if options[x] == 1 and x in all_labels]
            neg_labels = [x for x in options if options[x] == 0 and x in all_labels]
            if len(pos_labels) == 0 or len(neg_labels) == 0:
                return None
            if random.random() > 0.5:
                disease = random.choice(pos_labels).lower()
                options = {"Yes": 1, "No": 0}
            else:
                disease = random.choice(neg_labels).lower()
                options = {"Yes": 0, "No": 1}
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            if "options" in prompt_template:
                options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(disease=disease, options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_3

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    form_qas_2 = [create_qa_2(p, t) for p, t in zip(prompt_templates_2, target_templates_2)]
    form_qas_3 = [create_qa_3(p, t) for p, t in zip(prompt_templates_3, target_templates_3)]

    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: (
            [func(options) for func in form_qas[:-1]][:3]
            + [func(options) for func in form_qas_2 if func(options)]
            + [func(options) for func in form_qas_3 if func(options)]
    )


def image_classification_mimic_cxr(instruct=True):
    prompt_templates = [
        "Please review the chest X-ray(s) and classify any diseases from the following list "
        "that are present in the image. Options:\n{options}",

        "Given the CXR, identify any diseases. Options:\n{options}",

        "Identify any diseases visible in the given CXR. Options:\n{options}",

        "Perform disease classification on the given CXR. Options:\n{options}",

        "Which diseases are represented in this image:\n{options}",

        "{options}\nPerform disease classification by matching the observed conditions "
        "in the chest X-ray with the diseases listed",

        "Perform disease classification given the following label set:\n{options}",

        "Perform disease classification",

        "Identify abnormalities that are present in the CXRs.",

        "Image Classification (MIMIC-CXR)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options):
            # attach choice styles
            if "options" in prompt_template:
                options = add_choice_styles(options)
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = [k for k, v in options.items() if v >= -1]
            options_a = [k for k, v in options.items() if v >= 1]
            if len(options_q) < 2 or random.random() < 0.8:
                options_q = list(options.keys())
            suffix = ""
            if "options" in prompt_template:
                suffix = "\nThere may be more than one answer."
                if random.random() > 0.5:
                    options_a = [option.split(" ")[0] for option in options_a]
                    suffix = "\nThere may be more than one answer. " \
                             "Answer with the options’ letter/number from the given choices directly."
            return {
                "q": prompt_template.format(options="\n".join(options_q)) + suffix,
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa

    prompt_templates_2 = [
        "Which findings are in this chest X-ray? Options:\n{options}",
        "Which diseases are represented in this image:\n{options}",
        "Perform disease classification given the following label set:\n{options}",
    ]
    target_templates_2 = ["{answer}"] * len(prompt_templates_2)

    def create_qa_2(prompt_template, target_template):
        def form_qa_2(options):
            all_labels = [x for x in options if x not in ["Pleural Other", "No Finding"]]
            pos_label = [x for x in options if options[x] == 1 and x in all_labels]
            if len(pos_label) < 1 or len(pos_label) > 5:
                return None
            # construct negative options
            neg_findings = []
            for i in range(3):
                neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                while neg_sample == set(pos_label):
                    neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                neg_findings.append(neg_sample)
            pos_findings = ", ".join(pos_label).lower()
            neg_findings = [", ".join(x).lower() for x in neg_findings]
            # construct options
            options = {pos_findings: 1}
            options.update({item: 0 for item in neg_findings})
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_2

    prompt_templates_3 = [
        "Does this chest X-ray show {disease}?",
        "Does this chest X-ray show {disease}? Options:\n{options}",
    ]
    target_templates_3 = ["{answer}"] * len(prompt_templates_3)

    def create_qa_3(prompt_template, target_template):
        def form_qa_3(options):
            all_labels = [x for x in options if x not in ["Pleural Other", "No Finding"]]
            pos_labels = [x for x in options if options[x] == 1 and x in all_labels]
            neg_labels = [x for x in options if options[x] == 0 and x in all_labels]
            if len(pos_labels) == 0 or len(neg_labels) == 0:
                return None
            if random.random() > 0.5:
                disease = random.choice(pos_labels).lower()
                options = {"Yes": 1, "No": 0}
            else:
                disease = random.choice(neg_labels).lower()
                options = {"Yes": 0, "No": 1}
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            if "options" in prompt_template:
                options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(disease=disease, options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_3

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    form_qas_2 = [create_qa_2(p, t) for p, t in zip(prompt_templates_2, target_templates_2)]
    form_qas_3 = [create_qa_3(p, t) for p, t in zip(prompt_templates_3, target_templates_3)]

    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: (
            [func(options) for func in form_qas[:-1]][:3]
            + [func(options) for func in form_qas_2 if func(options)]
            + [func(options) for func in form_qas_3 if func(options)]
    )


def image_classification_nlm_tb(instruct=True):
    prompt_templates = [
        "You will be given an image of chest X-ray. please identify if tuberculosis is present.",
        "Identify if tuberculosis is present in the chest X-ray.",
        "Has the patient been diagnosed with tuberculosis?",
        "Does the chest X-ray show tuberculosis?",
        "Is the given chest X-ray positive or negative for tuberculosis?",
        "Is the image positive or negative for tuberculosis?",
        "Tuberculosis\nIs it positive or negative?",
        "Can this patient be diagnosed with the following disease: Tuberculosis?",
        "Determine if tuberculosis is present in the X-ray.",
        "Image Classification (NLT-TB)",
    ]

    target_templates = ["{formatted_answer}"] * len(prompt_templates)
    answer_templates = [
        lambda options: "Yes" if options["Tuberculosis"] else "No",
        lambda options: "Yes" if options["Tuberculosis"] else "No",
        lambda options: "Yes" if options["Tuberculosis"] else "No",
        lambda options: "Yes" if options["Tuberculosis"] else "No",
        lambda options: "Positive" if options["Tuberculosis"] else "Negative",
        lambda options: "Positive" if options["Tuberculosis"] else "Negative",
        lambda options: "Positive" if options["Tuberculosis"] else "Negative",
        lambda options: "Yes" if options["Tuberculosis"] else "No",
        lambda options: "Yes" if options["Tuberculosis"] else "No",
        lambda options: "Positive" if options["Tuberculosis"] else "Negative",
    ]

    def create_qa(prompt_template, target_template, answer_template):
        def form_qa(options):
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            return {
                "q": prompt_template,
                "a": target_template.format(formatted_answer=answer_template(options))
            }

        return form_qa

    form_qas = [create_qa(p, t, a) for p, t, a in zip(prompt_templates, target_templates, answer_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: [func(options) for func in form_qas[:-1]][:5]


def image_classification_padchest(instruct=True):
    prompt_templates = [
        "You will be given an X-ray or multiple X-rays of the chest. The image may contain "
        "different diseases. Identify the diseases. Options:\n{options}",

        "Given the CXR, identify the diseases. Options:\n{options}",

        "Identify any diseases visible in the given CXR. Options:\n{options}",

        "Perform disease classification on the given CXR. Options:\n{options}",

        "Which diseases are represented in this image:\n{options}",

        "{options}\nPerform disease classification by matching the observed conditions "
        "in the chest X-ray with the diseases listed",

        "Perform disease classification given the following label set:\n{options}",

        "Perform disease classification",

        "Identify abnormalities that are present in the CXRs.",

        "Image Classification (PadChest)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options):
            # attach choice styles
            if "options" in prompt_template:
                options = add_choice_styles(options)
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = [k for k, v in options.items()]
            random.shuffle(options_q)
            options_a = [k for k, v in options.items() if v >= 1]
            suffix = ""
            if "options" in prompt_template:
                suffix = "\nThere may be more than one answer."
                if random.random() > 0.5:
                    options_a = [option.split(" ")[0] for option in options_a]
                    suffix = "\nThere may be more than one answer. " \
                             "Answer with the options’ letter/number from the given choices directly."
            return {
                "q": prompt_template.format(options="\n".join(options_q)) + suffix,
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa

    prompt_templates_2 = [
        "Which findings are in this chest X-ray? Options:\n{options}",
        "Which diseases are represented in this image:\n{options}",
        "Perform disease classification given the following label set:\n{options}",
    ]
    target_templates_2 = ["{answer}"] * len(prompt_templates_2)

    def create_qa_2(prompt_template, target_template):
        def form_qa_2(options):
            all_labels = [x for x in options if x not in ["normal"]]
            pos_label = [x for x in options if options[x] == 1 and x in all_labels]
            if len(pos_label) < 1 or len(pos_label) > 5:
                return None
            # construct negative options
            neg_findings = []
            for i in range(3):
                neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                while neg_sample == set(pos_label):
                    neg_sample = set(np.random.choice(all_labels, len(pos_label), replace=False))
                neg_findings.append(neg_sample)
            pos_findings = ", ".join(pos_label).lower()
            neg_findings = [", ".join(x).lower() for x in neg_findings]
            # construct options
            options = {pos_findings: 1}
            options.update({item: 0 for item in neg_findings})
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_2

    prompt_templates_3 = [
        "Does this chest X-ray show {disease}?",
        "Does this chest X-ray show {disease}? Options:\n{options}",
    ]
    target_templates_3 = ["{answer}"] * len(prompt_templates_3)

    def create_qa_3(prompt_template, target_template):
        def form_qa_3(options):
            all_labels = [x for x in options if x not in ["normal"]]
            pos_labels = [x for x in options if options[x] == 1 and x in all_labels]
            neg_labels = [x for x in options if options[x] == 0 and x in all_labels]
            if len(pos_labels) == 0 or len(neg_labels) == 0:
                return None
            if random.random() > 0.5:
                disease = random.choice(pos_labels).lower()
                options = {"Yes": 1, "No": 0}
            else:
                disease = random.choice(neg_labels).lower()
                options = {"Yes": 0, "No": 1}
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            # add option style
            if "options" in prompt_template:
                options = add_choice_styles(options)
            options_a = [k for k, v in options.items() if v]
            options = [k for k, v in options.items()]
            return {
                "q": prompt_template.format(disease=disease, options="\n".join(options)),
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa_3

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    form_qas_2 = [create_qa_2(p, t) for p, t in zip(prompt_templates_2, target_templates_2)]
    form_qas_3 = [create_qa_3(p, t) for p, t in zip(prompt_templates_3, target_templates_3)]

    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: (
            [func(options) for func in form_qas[:-1]][:3]
            + [func(options) for func in form_qas_2 if func(options)]
            + [func(options) for func in form_qas_3 if func(options)]
    )


def image_classification_rsna(instruct=True):
    prompt_templates = [
        "You will be given an image of chest X-ray. Please identify if pneumonia is present.",
        "From the given CXR, identify if the patient has pneumonia. Option: (A) Yes, (B) No",
        "Does the patient have pneumonia?",
        "Does the CXR show signs of pneumonia?",
        "Is the CXR positive or negative for pneumonia?",
        "Please determine if pneumonia is present in the chest X-ray. Option: (A) Yes, (B) No",
        "Pneumonia\nIs the CXR positive or negative? Option: (A) Positive, (B) Negative",
        "Has the patient been diagnosed with the following disease: pneumonia?",
        "Evaluate the chest X-ray for the presence of pneumonia.",
        "Image Classification (RNSA)",
    ]

    target_templates = ["{answer}"] * len(prompt_templates)
    answer_templates = [
        lambda options: "Yes" if options["Pneumonia"] else "No",
        lambda options: "(A) Yes" if options["Pneumonia"] else "(B) No",
        lambda options: "Yes" if options["Pneumonia"] else "No",
        lambda options: "Yes" if options["Pneumonia"] else "No",
        lambda options: "Positive" if options["Pneumonia"] else "Negative",
        lambda options: "(A) Yes" if options["Pneumonia"] else "(B) No",
        lambda options: "(A) Positive" if options["Pneumonia"] else "(B) Negative",
        lambda options: "Yes" if options["Pneumonia"] else "No",
        lambda options: "Yes" if options["Pneumonia"] else "No",
        lambda options: "Positive" if options["Pneumonia"] else "Negative",
    ]

    def create_qa(prompt_template, target_template, answer_template):
        def form_qa(options):
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            return {
                "q": prompt_template,
                "a": target_template.format(answer=answer_template(options))
            }

        return form_qa

    form_qas = [create_qa(p, t, a) for p, t, a in zip(prompt_templates, target_templates, answer_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: [func(options) for func in form_qas[:-1]][:5]


def image_text_matching_mimic_cxr(instruct=True):
    prompt_templates = [
        "In this task, you are given one/multiple chest X-ray(s). You need to decide "
        "if it matches the following text: {text}",

        "Decide if the provided image matches the following text: {text}",

        "Assess whether the following text caption accurately describes the content of the image: {text}",

        "In this task, you are given one/multiple chest X-rays. Decide if it matches the "
        "following text or not: {text}",

        "Please decide if the image matches the following text: {text}",

        "Check if the information in the text caption accurately represents the features in the image: {text}",

        "Decide whether the CXR matches the following text: {text}",

        "In this task, you are given a CXR. You need to decide if it matches the following text: {text}",

        "Verify if the following text caption matches the visual content of the chest X-ray: {text}",

        "Image-Text Matching (MIMIC-CXR): {text}",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(text, options):
            return {
                "q": prompt_template.format(text=text),
                "a": target_template.format(answer=", ".join([k for k, v in options.items() if v >= 1]))
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda text, options: [func(text, options) for func in form_qas[:-1]][:5]


def image_text_matching_roco(instruct=True):
    prompt_templates = [
        "In this task, you are given one/multiple chest X-ray(s). You need to decide if it matches the "
        "following text: {text}",

        "Decide if the provided image matches the following text: {text}",

        "Assess whether the following text caption accurately describes the content of the image: {text}",

        "In this task, you are given one/multiple chest X-ray. Decide if it matches the following text "
        "or not: {text}",

        "Please decide if the image matches the following text: {text}",

        "Check if the information in the text caption accurately represents the features in the image: {text}",

        "Decide whether the CXR matches the following text: {text}",

        "In this task, you are given a CXR. You need to decide if it matches the following text: {text}",

        "Verify if the following text caption matches the visual content of the chest X-ray: {text}",

        "Image-Text Matching (ROCO): {text}",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(text, options):
            return {
                "q": prompt_template.format(text=text),
                "a": target_template.format(answer=", ".join([k for k, v in options.items() if v >= 1]))
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda text, options: [func(text, options) for func in form_qas[:-1]][:5]


def image_text_selection_roco(instruct=True):
    prompt_templates = [
        "Please choose the text caption that accurately describes the content of the image. "
        "Options: (a) {a}, (b) {b}",

        "Which option best describes the provided image? Options: (a) {a}, (b) {b}",

        "Indicate the text caption that most closely matches the features observed in the chest "
        "X-ray. Options: (a) {a}, (b) {b}",

        "Choose the caption that provides the best description of the chest X-ray. Options: (a) {a}, (b) {b}",

        "Choose the text caption that best describes the image. Options: (a) {a}, (b) {b}",

        "Select a caption matching the image. Options: (a) {a}, (b) {b}",

        "Choose the caption that provides the most fitting interpretation of the chest "
        "X-ray image. Options: (a) {a}, (b) {b}",

        "From the given captions, select the one that aligns best with the content of the "
        "chest X-ray. Options: (a) {a}, (b) {b}",

        "Select the best caption for the image. Options: (a) {a}, (b) {b}",

        "Image-Text Selection (ROCO) Options: (a) {a}, (b) {b}",
    ]
    target_templates = ["{formatted_answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options):
            return {
                "q": prompt_template.format(a=options[0][0], b=options[1][0]),
                "a": target_template.format(formatted_answer=["(a)", "(b)"][[k[1] for k in options].index(True)])
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: [func(options) for func in form_qas[:-1]][:5]


def image_text_selection_mimic_cxr(instruct=True):
    prompt_templates = [
        "Please choose the report that accurately describes the content of the image. "
        "Options: (a) {a}, (b) {b}",

        "Which option best describes the provided image? Options: (a) {a}, (b) {b}",

        "Indicate the text caption that most closely matches the features observed in the "
        "chest X-ray. Options: (a) {a}, (b) {b}",

        "Choose the text that provides the best description of the chest X-ray. Options: (a) {a}, (b) {b}",

        "Choose the text that best describes the image. Options: (a) {a}, (b) {b}",

        "Select a report matching the image. Options: (a) {a}, (b) {b}",

        "Choose the report that provides the most fitting interpretation of the chest "
        "X-ray. Options: (a) {a}, (b) {b}",

        "From the given captions, select the one that aligns best with the content of the chest "
        "X-ray. Options: (a) {a}, (b) {b}",

        "Select the best caption for the image. Options: (a) {a}, (b) {b}",

        "Image-Text Selection (MIMIC-CXR) Options: (a) {a}, (b) {b}",
    ]
    target_templates = ["{formatted_answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options):
            return {
                "q": prompt_template.format(a=options[0][0], b=options[1][0]),
                "a": target_template.format(formatted_answer=["(a)", "(b)"][[k[1] for k in options].index(True)])
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: [func(options) for func in form_qas[:-1]][:5]


def impression_generation_cxrpro(instruct=True):
    prompt_templates = [
        "Summarize your overall assessment of the chest X-ray, following the style of ChatGPT "
        "without references to prior studies",

        "You are provided with one or multiple chest X-ray image(s). Please write the impressions "
        "section of the diagnostic report in the style of ChatGPT. Do not refer to prior studies.",

        "You are provided with one or multiple chest X-ray image(s). Please craft the impressions "
        "section of the diagnostic report following the style of ChatGPT without prior references.",

        "Write the Impression section for the provided CXR without prior reference in the style of ChatGPT.",

        "Given the CXR images, provide an impression without prior references in the style of ChatGPT.",

        "In the style of ChatGPT, write an Impression without prior reference for the given images.",

        "Write a ChatGPT-style Impression without prior reference for the given images.",

        "Write an impression without prior reference for the given images as if you are ChatGPT.",

        "Provide a ChatGPT-style impressions section for the chest X-ray without references to prior examinations",

        "Impression Generation (CXR-PRO)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(impression):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=impression)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda impression: [func(impression) for func in form_qas[:-1]][:5]


def impression_generation_candid(instruct=True):
    prompt_templates = [
        "Summarize your overall assessment of the chest X-ray, following the style of New Zealand medical professionals",

        "You are provided with one or multiple chest X-ray image(s). Please write the impressions section of "
        "the report in the style of New Zealand physicians.",

        "You are provided with one or multiple chest X-ray image(s). Please craft the impressions section of "
        "the diagnostic report following the style of New Zealand radiologists",

        "Write the Impression section for the provided CXR in the style of New Zealand radiologists.",

        "Given the CXR images from New Zealand, provide an example Impression written by physicians.",

        "Suppose you are a radiologist from New Zealand. Write an Impression for the given images.",

        "Following the style of New Zealand radiologists, write a clear and concise summary of key findings "
        "in the given chest X-rays.",

        "Write an impression for the given images as if you are a radiologist from New Zealand.",

        "Generate the impressions section of the diagnostic report (in the style of New Zealand's "
        "doctors) associated with the provided CXRs",

        "Impression Generation (Candid)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(impression):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=impression)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda impression: [func(impression) for func in form_qas[:-1]][:5][:5]


def impression_generation_intermountain(instruct=True):
    prompt_templates = [
        "Write an example impression section for the CXR",

        "You are provided with one or multiple chest X-ray image(s). "
        "Write an example impression section for the diagnostic report.",

        "You are provided with one or multiple chest X-ray image(s). "
        "Please provide an example impression section to summarize the key findings "
        "in the style of American radiologists.",

        "Examine the chest X-ray thoroughly and write an example impression section of the diagnostic report.",

        "Write an example impression section in the clinical style for the provided CXR",

        "Summarize your overall assessment of the chest X-ray and "
        "write an example impression section following the style of American medical professionals",

        "For the given images, write an example impression section that may exist in clinical practice.",

        "Write an example impression section for the given images as if you are a radiologist.",

        "Write an example impression section similar to the impression sections in clinical practice.",

        "Impression Generation (InterMountain)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(impression):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=impression)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda impression: [func(impression) for func in form_qas[:-1]][:5]


def impression_generation_mimic_cxr_struct(instruct=True):
    prompt_templates = [
        "Write an impression section for the CXR.",

        "Write a structured impression section for the CXR.",

        "You are provided with one or multiple chest X-ray image(s). "
        "Write a structured impression section for the diagnostic report.",

        "You are provided with one or multiple chest X-ray image(s). "
        "Please provide a structured impression section to summarize the key findings.",

        "Examine the chest X-ray thoroughly and write a structured impression section of the diagnostic report.",

        "Assess the chest X-ray, identify key findings in the CXR and write a structured impression section.",

        "Write a structured impression section for the given images as if you are a radiologist.",

        "You are provided with one or multiple chest X-ray image(s). "
        "Write a structured impression section with subsections "
        "for each anatomical feature and with the positive findings highlighted.",

        "Write a structured impression section for the provided CXR with subsections for each anatomical feature. "
        "Highlight positive findings.",

        "Impression Generation (MIMIC-CXR-Struct)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(impression):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=impression)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda impression: [func(impression) for func in form_qas[:-1]][:5]


def impression_generation_with_indication_chexpert_struct(instruct=True):
    prompt_templates = [
        'Given the clinical history: "{indication}", write an impression section for the CXR.',

        'Given the clinical history: "{indication}", write a structured impression section for the CXR.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the clinical history: "{indication}", write a structured impression section for the diagnostic report.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the clinical history: "{indication}", '
        'please provide a structured impression section to summarize the key findings.',

        'Given the clinical history: "{indication}", '
        'examine the chest X-ray thoroughly and write a structured impression section of the diagnostic report.',

        'Given the clinical history: "{indication}", '
        'assess the chest X-ray, identify key findings in the CXR and write a structured impression section.',

        'Given the clinical history: "{indication}", '
        'write a structured impression section for the given images as if you are a radiologist.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the clinical history: "{indication}", '
        'write a structured impression section with subsections '
        'for each anatomical feature and with the positive findings highlighted.',

        'Given the clinical history: "{indication}", '
        'write a structured impression section for the provided CXR with subsections for each anatomical feature. '
        'Highlight positive findings.',

        'Impression Generation with Indication (MIMIC-CXR-Struct): "{indication}"',
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(indication, impression):
            return {
                "q": prompt_template.format(indication=indication),
                "a": target_template.format(answer=impression)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda indication, impression: [func(indication, impression) for func in form_qas[:-1]][:5]


def impression_generation_with_indication_mimic_cxr_struct(instruct=True):
    prompt_templates = [
        'Given the indication: "{indication}", write an impression section for the CXR.',

        'Given the indication: "{indication}", write a structured impression section for the CXR.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the indication: "{indication}", write a structured impression section for the diagnostic report.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the indication: "{indication}", '
        'please provide a structured impression section to summarize the key findings.',

        'Given the indication: "{indication}", '
        'examine the chest X-ray thoroughly and write a structured impression section of the diagnostic report.',

        'Given the indication: "{indication}", '
        'assess the chest X-ray, identify key findings in the CXR and write a structured impression section.',

        'Given the indication: "{indication}", '
        'write a structured impression section for the given images as if you are a radiologist.',

        'You are provided with one or multiple chest X-ray image(s). '
        'Given the indication: "{indication}", '
        'write a structured impression section with subsections '
        'for each anatomical feature and with the positive findings highlighted.',

        'Given the indication: "{indication}", '
        'write a structured impression section for the provided CXR with subsections for each anatomical feature. '
        'Highlight positive findings.',

        'Impression Generation with Indication (MIMIC-CXR-Struct): "{indication}"',
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(indication, impression):
            return {
                "q": prompt_template.format(indication=indication),
                "a": target_template.format(answer=impression)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda indication, impression: [func(indication, impression) for func in form_qas[:-1]][:5]


def impression_generation_mimic_cxr(instruct=True):
    prompt_templates = [
        "Write an example impression section for the CXR",

        "You are provided with one or multiple chest X-ray image(s). "
        "Write an example impression section for the diagnostic report.",

        "You are provided with one or multiple chest X-ray image(s). "
        "Please provide an example impression section to summarize the key findings "
        "in the style of American radiologists.",

        "Examine the chest X-ray thoroughly and write an example impression section of the diagnostic report.",

        "Write an example impression section in the clinical style for the provided CXR",

        "Summarize your overall assessment of the chest X-ray and "
        "write an example impression section following the style of American medical professionals",

        "For the given images, write an example impression section that may exist in clinical practice.",

        "Write an example impression section for the given images as if you are a radiologist.",

        "Write an example impression section similar to the impression sections in clinical practice.",

        "Impression Generation (MIMIC-CXR)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(impression):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=impression)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda impression: [func(impression) for func in form_qas[:-1]][:5]


def impression_generation_openi(instruct=True):
    prompt_templates = [
        "Write an example impression section for the CXR",

        "You are provided with one or multiple chest X-ray image(s). "
        "Write an example impression section for the diagnostic report.",

        "You are provided with one or multiple chest X-ray image(s). "
        "Please provide an example impression section to summarize the key findings "
        "in the style of American radiologists.",

        "Examine the chest X-ray thoroughly and write an example impression section of the diagnostic report.",

        "Write an example impression section in the clinical style for the provided CXR",

        "Summarize your overall assessment of the chest X-ray and "
        "write an example impression section following the style of American medical professionals",

        "For the given images, write an example impression section that may exist in clinical practice.",

        "Write an example impression section for the given images as if you are a radiologist.",

        "Write an example impression section similar to the impression sections in clinical practice.",

        "Impression Generation (OpenI)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(impression):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=impression)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda impression: [func(impression) for func in form_qas[:-1]][:5]


def local_findings_generation_mimic_cxr_struct(instruct=True):
    prompt_templates = [
        'Describe findings associated with "{anatomy}" in the given image(s)',
        'Explain characteristics associated with "{anatomy}" in the given image(s)',
        'Please provide a detailed description of "{anatomy}" in the chest X-ray',
        'Given the image(s), describe "{anatomy}"',
        'Write a caption for "{anatomy}" using the given CXR',
        'Describe "{anatomy}"',
        'You are given one or multiple CXR(s). Please write the observed findings for "{anatomy}"',
        'Describe "{anatomy}" in the images',
        'Write the findings for "{anatomy}" as shown in the provided image(s)',
        'Local Findings Generation (MIMIC-CXR-Struct): "{anatomy}"',
    ]
    target_templates = ["{description}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(anatomy, description):
            return {
                "q": prompt_template.format(anatomy=anatomy),
                "a": target_template.format(description=description)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda anatomy, description: [func(anatomy, description) for func in form_qas[:-1]][:5]


def local_progression_findings_generation_mimic_cxr_struct(instruct=True):
    prompt_templates = [
        'Describe the progression associated with "{anatomy}" in the given images',
        'Please provide a detailed description of the progression of "{anatomy}" in the chest X-ray',
        'Given the images, describe the progression of "{anatomy}"',
        'Describe the progression of "{anatomy}"',
        'You are given two CXRs. Please write the observed progression for "{anatomy}"',
        'Describe the progression of "{anatomy}" in the images',
        'Write the progression for "{anatomy}" as shown in the provided images',
        'Local Progression Findings Generation (MIMIC-CXR-Struct): "{anatomy}"',
    ]
    target_templates = ["{description}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(anatomy, description):
            return {
                "q": prompt_template.format(anatomy=anatomy),
                "a": target_template.format(description=description)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda anatomy, description: [func(anatomy, description) for func in form_qas[:-1]][:5]


def local_impression_generation_mimic_cxr_struct(instruct=True):
    prompt_templates = [
        'Write a diagnostic summarization for "{anatomy}" in the given image(s)',
        'Write the impressions associated with "{anatomy}" in the given image(s)',
        'Given the images, provide a summary of key observations noted in "{anatomy}"',
        'Given the image(s), write a findings summarization for "{anatomy}"',
        'Write a diagnostic summarization for "{anatomy}" for the given CXR',
        'Given the CXRs, create the impressions section of a diagnostic report for "{anatomy}"',
        'You are given one or multiple CXR(s), please write the Impression section for "{anatomy}"',
        'Write impressions associated with "{anatomy}" in the images',
        'Provide a diagnostic summarization of "{anatomy}" in the image(s)',
        'Local Impression Generation (MIMIC-CXR-Struct): "{anatomy}"',
    ]
    target_templates = ["{description}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(anatomy, description):
            return {
                "q": prompt_template.format(anatomy=anatomy),
                "a": target_template.format(description=description)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda anatomy, description: [func(anatomy, description) for func in form_qas[:-1]][:5]


def local_progression_impression_generation_mimic_cxr_struct(instruct=True):
    prompt_templates = [
        'Write a diagnostic summarization for the progression of "{anatomy}" in the given images',
        'Write the impressions associated with the progression of "{anatomy}" in the given images',
        'Given the images, provide a summary of key progression noted in "{anatomy}"',
        'Given the images, write a findings summarization for the progression of "{anatomy}"',
        'Write a diagnostic summarization for the progression of "{anatomy}" for the given CXRs',
        'Given the CXRs, create the impressions section of a diagnostic report for the progression of "{anatomy}"',
        'You are given two CXRs, please write the Impression section for the progression of "{anatomy}"',
        'Write impressions associated with the progression of "{anatomy}" in the images',
        'Provide a diagnostic summarization of the progression of "{anatomy}" in the images',
        'Local Progression Impression Generation (MIMIC-CXR-Struct): "{anatomy}"',
    ]
    target_templates = ["{description}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(anatomy, description):
            return {
                "q": prompt_template.format(anatomy=anatomy),
                "a": target_template.format(description=description)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda anatomy, description: [func(anatomy, description) for func in form_qas[:-1]][:5]


def named_entity_recognition_radgraph(instruct=True):
    prompt_templates = [
        'Given the list of entity types [Observation (Definitely Absent), Observation (Definitely Present), '
        'Observation (Uncertain), Anatomy], '
        'find out all words/phrases that indicate the above types of named entities. '
        'Sentence: {text}',
        'Named Entity Recognition (RadGraph): "{text}"',
    ]
    target_templates = ["{entities}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(text, entities):
            return {
                "q": prompt_template.format(text=text),
                "a": target_template.format(entities=", ".join([f'[{entity[0]}, {entity[1]}]' for entity in entities]))
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda text, entities: [func(text, entities) for func in form_qas[:-1]][:5]


def natural_language_explanation_mimic_nle(instruct=True):
    prompt_templates = [
        "The provided CXR shows signs of {diagnosis}. Please explain it.",
        "The patient has been diagnosed with {diagnosis}. Please explain why in natural Language.",
        "Explain the factors in the CXR that led to the following diagnosis: {diagnosis}",
        "Explain why the image(s) was diagnosed with {diagnosis}.",
        "Provide an explanation for why the image(s) are diagnosed with {diagnosis}. ",
        "The CXR shows signs of {diagnosis}. Please describe why.",
        "This patient was diagnosed with {diagnosis}. Explain why.",
        "The image(s) are diagnosed with {diagnosis}. Explain why.",
        "Explain {diagnosis} in the provided CXR.",
        "Natural Language Explanation (MIMIC-NLE): {diagnosis}",
    ]
    target_templates = ["{explanation}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options):
            return {
                "q": prompt_template.format(diagnosis=", ".join(
                    [MIMIC_LABEL2DIAGNOSIS[i] for i, l in enumerate(options["diagnosis_label"]) if l]).lower()),
                "a": target_template.format(explanation=options["nle"])
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: [func(options) for func in form_qas[:-1]][:5]


def natural_language_inference_radnli(instruct=True):
    prompt_templates = [
        "Read the following and determine the relation between the hypothesis and the premise: " \
        "Premise: {premise} Hypothesis: {hypothesis} Options:\n{options}",

        "Read the following and determine the relation between the hypothesis and the premise: " \
        "Hypothesis: {hypothesis} Premise: {premise} Options:\n{options}",

        "Determine the relationship between the following hypothesis and premise: " \
        "Premise: {premise} Hypothesis: {hypothesis} Options:\n{options}"

        "Determine the relationship between the following hypothesis and premise: " \
        "Hypothesis: {hypothesis} Premise: {premise} Options:\n{options}"
    ]
    target_templates = ["{formatted_answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(texts, options):
            # attach choice styles
            if "options" in prompt_template:
                options = add_choice_styles(options)
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = [k for k, v in options.items() if v >= -1]
            options_a = [k for k, v in options.items() if v >= 1]
            suffix = ""
            if random.random() > 0.5:
                options_a = [option.split(" ")[0] for option in options_a]
                suffix = "\nAnswer with the option’s letter/number from the given choices directly."
            return {
                "q": prompt_template.format(
                    premise=texts[0], hypothesis=texts[1], options="\n".join(options_q)) + suffix,
                "a": target_template.format(formatted_answer=", ".join(options_a))
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda texts, options: [func(texts, options) for func in form_qas[:-1]][:5]


def open_ended_qa_task(instruct=True):
    prompt_template_1 = "{question}"
    target_template_1 = "{answer}"
    form_qa_1 = lambda question, answer: {
        "q": prompt_template_1.format(question=question),
        "a": target_template_1.format(answer=answer)
    }

    form_qas = [form_qa_1]

    return random.choice(form_qas) if instruct else form_qas[-1]


def open_ended_vqa_mimic_cxr_vqa(instruct=True):
    return open_ended_qa_task(instruct)


def open_ended_vqa_medvqa2019(instruct=True):
    return open_ended_qa_task(instruct)


def open_ended_vqa_pmc_vqa(instruct=True):
    return open_ended_qa_task(instruct)


def open_ended_vqa_rad_restruct(instruct=True):
    return open_ended_qa_task(instruct)


def open_ended_vqa_slake(instruct=True):
    return open_ended_qa_task(instruct)


def open_ended_vqa_vqarad(instruct=True):
    return open_ended_qa_task(instruct)


def phrase_extraction_and_grounding_ms_cxr(instruct=True):
    prompt_template_1 = "Extract one of the phrase from the following sentence and ground it: {text}"
    target_template_1 = "{phrase}{boxes}"
    form_qa_1 = lambda text, phrase, boxes: {
        "q": prompt_template_1.format(text=text),
        "a": target_template_1.format(phrase=phrase, boxes=boxes)
    }

    form_qas = [form_qa_1]

    return random.choice(form_qas) if instruct else form_qas[-1]


def phrase_grounding_ms_cxr(instruct=True):
    prompt_templates = [
        "Please locate the following phrase: {phrase}",
        "Identify the position of the following finding in the CXR: {phrase}",
    ]
    target_templates = ["{boxes}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(phrase, boxes):
            return {
                "q": prompt_template.format(phrase=phrase),
                "a": target_template.format(phrase=phrase, boxes=boxes)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda phrase, boxes: [func(phrase, boxes) for func in form_qas][:5]


def pneumothorax_segmentation_candid(instruct=True):
    prompt_templates = [
        "Detect pneumothoraces in the given image.",
        "Perform pneumothorax detection on the given image.",
        "Please identify pneumothoraces and provide bounding boxes.",
        "Find the locations of pneumothoraces in the given image with bounding boxes.",
        "Locate a pneumothorax in the given CXR.",
        "Find pneumothorax in the given image.",
        "Detect the following diseases in the image: pneumothorax.",
        "Locate the following diseases in the image: pneumothorax.",
        "In the provided chest X-ray, identify bounding box coordinates for each pneumothorax.",
        "Pneumothorax Detection (Candid)",
    ]
    target_templates = ["{boxes}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(disease, boxes):
            return {
                "q": prompt_template.format(disease=disease),
                "a": target_template.format(boxes=boxes)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda disease, boxes: [func(disease, boxes) for func in form_qas[:-1]][:5]


def pneumothorax_segmentation_siim(instruct=True):
    prompt_templates = [
        "Detect pneumothoraces in the given image.",
        "Perform pneumothorax detection on the given image.",
        "Please identify pneumothoraces and provide bounding boxes.",
        "Find the locations of pneumothoraces in the given image with bounding boxes.",
        "Locate a pneumothorax in the given CXR.",
        "Find pneumothorax in the given image.",
        "Detect the following diseases in the image: pneumothorax.",
        "Locate the following diseases in the image: pneumothorax.",
        "In the provided chest X-ray, identify bounding box coordinates for each pneumothorax.",
        "Pneumothorax detection (SIIM).",
    ]
    target_templates = ["{boxes}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(disease, boxes):
            return {
                "q": prompt_template.format(disease=disease),
                "a": target_template.format(boxes=boxes)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda disease, boxes: [func(disease, boxes) for func in form_qas[:-1]][:5]


def progression_findings_generation_mimic_cxr(instruct=True):
    prompt_templates = [
        "You are provided with chest X-rays of a patient from two different timepoints. "
        "Please analyze the findings in both chest X-rays and describe how the patient's condition has "
        "progressed or changed over time.",

        "You are provided with chest X-rays taken during two separate exams. Examine the two chest X-rays and "
        "provide an assessment of any developments or changes in the findings.",

        "Evaluate the chest X-rays and describe any evolving patterns, changes, or developments in the findings",

        "Describe any alterations, improvements, or worsening of findings between the two examinations",

        "Please describe how the patient's condition has progressed by comparing the findings in both chest X-rays",

        "Please compare the findings in both CXRs and provide an assessment of the progression in the patient's condition",

        "Examine the findings in both chest X-rays and describe the changes that have occurred in the patient's condition",

        "Write the trajectory of findings by comparing the second image to the first one.",

        "Write a radiology report as an expert radiologist describing changes in findings between the two provided CXRs.",

        "Progression Findings Generation (MIMIC-CXR)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(findings):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=findings)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda findings: [func(findings) for func in form_qas[:-1]][:5]


def progression_findings_generation_mimic_cxr_struct(instruct=True):
    prompt_templates = [
        "You are provided with chest X-rays of a patient from two different timepoints. "
        "Please analyze the findings in both chest X-rays and describe how the patient's condition has "
        "progressed or changed over time.",

        "You are provided with chest X-rays taken during two separate exams. Examine the two chest X-rays and "
        "provide an assessment of any developments or changes in the findings.",

        "Evaluate the chest X-rays and describe any evolving patterns, changes, or developments in the findings",

        "Describe any alterations, improvements, or worsening of findings between the two examinations",

        "Please describe how the patient's condition has progressed by comparing the findings in both chest X-rays",

        "Please compare the findings in both CXRs and provide an assessment of the progression in the patient's condition",

        "Examine the findings in both chest X-rays and describe the changes that have occurred in the patient's condition",

        "Write the trajectory of findings by comparing the second image to the first one.",

        "Write a radiology report as an expert radiologist describing changes in findings between the two provided CXRs.",

        "Progression Findings Generation (MIMIC-CXR)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(findings):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=findings)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda findings: [func(findings) for func in form_qas[:-1]][:5]


def progression_impression_generation_mimic_cxr(instruct=True):
    prompt_templates = [
        "You are provided with chest X-rays of a patient from two different timepoints. Analyze both chest X-rays "
        "and describe how the patient's condition has progressed or changed over time. Provide your response in the "
        "format of an impressions section.",

        "You are provided with chest X-rays taken during two separate exams. Prove an assessment of any "
        "developments or changes in the format of an impressions section.",

        "Please write an Impression section describing changes between the two images.",

        "Write an impressions section that describes any alterations, improvements, or worsening observations "
        "between the two images",

        "Please describe how the patient's condition has progressed by comparing the chest X-rays. Write your "
        "response in the format of an impressions section from a diagnostic report.",

        "Please compare both CXRs and provide an assessment of the progression in the patient's condition "
        "in the format of an impressions section.",

        "For the two provided CXRs, write an Impression section that describes changes.",

        "Write the trajectory of impressions by comparing the second image to the first one.",

        "Write an Impression section as an expert radiologist describing changes between the provided CXRs.",

        "Progression Impression Generation (MIMIC-CXR)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(impression):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=impression)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda impression: [func(impression) for func in form_qas[:-1]][:5]


def progression_impression_generation_mimic_cxr_struct(instruct=True):
    prompt_templates = [
        "You are provided with chest X-rays of a patient from two different timepoints. Analyze both chest X-rays "
        "and describe how the patient's condition has progressed or changed over time. Provide your response in the "
        "format of an impressions section.",

        "You are provided with chest X-rays taken during two separate exams. Prove an assessment of any "
        "developments or changes in the format of an impressions section.",

        "Please write an Impression section describing changes between the two images.",

        "Write an impressions section that describes any alterations, improvements, or worsening observations "
        "between the two images",

        "Please describe how the patient's condition has progressed by comparing the chest X-rays. Write your "
        "response in the format of an impressions section from a diagnostic report.",

        "Please compare both CXRs and provide an assessment of the progression in the patient's condition "
        "in the format of an impressions section.",

        "For the two provided CXRs, write an Impression section that describes changes.",

        "Write the trajectory of impressions by comparing the second image to the first one.",

        "Write an Impression section as an expert radiologist describing changes between the provided CXRs.",

        "Progression Impression Generation (MIMIC-CXR)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(impression):
            return {
                "q": prompt_template,
                "a": target_template.format(answer=impression)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda impression: [func(impression) for func in form_qas[:-1]][:5]


def report_evaluation_rexval(instruct=True):
    prompt_templates = [
        'Reference report: "{reference}", candidate report: "{candidate}" '
        'Identify any errors in the candidate report when compared to the reference report.',

        'Given the reference report and the candidate report, identify any '
        'errors in the candidate report when compared to the reference. '
        'Reference report: "{reference}", candidate report: "{candidate}" ',

    ]
    target_templates = ["{errors}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(text, errors):
            return {
                "q": prompt_template.format(reference=text[0], candidate=text[1]),
                "a": target_template.format(errors=", ".join([f'{k}: {v}' for k, v in errors.items()]))
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda text, errors: [func(text, errors) for func in form_qas][:5]


def impression_generation_bimcv_covid19(instruct=True):
    prompt_templates = [
        "You are provided with one chest X-ray image. Please write its diagnostic report in the style of "
        "Spanish medical professionals.",

        "You are provided with one chest X-ray image. Please write a diagnostic report in the style of radiologists "
        "from Spain.",

        "Write a diagnostic report following the style of Spanish physicians.",

        "Given the image, please write its diagnostic report in the style of Spanish radiologists.",

        "You are provided with one chest X-ray image. Please write its diagnostic report following the "
        "guidelines of the Spanish medical system.",

        "You are provided with one chest X-ray image collected from Spain. Please write its diagnostic report.",

        "You are provided with one chest X-ray image from Spain. Compose a diagnostic report.",

        "Generate a diagnostic report for the given image as if you are a radiologist from Spain.",

        "Write a diagnostic report for this image in the style of Spanish physicians.",

        "Report Generation (BIMCV-COVID19)",
    ]
    target_templates = ["{report}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(report):
            return {
                "q": prompt_template,
                "a": target_template.format(report=report)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda report: [func(report) for func in form_qas[:-1]][:5]


def impression_generation_padchest(instruct=True):
    prompt_templates = [
        "You are provided with one chest X-ray image. Please write its diagnostic report in the style of Spanish "
        "medical professionals.",

        "You are provided with one chest X-ray image. Please write a diagnostic report in the style of radiologists "
        "from Spain.",

        "Write a diagnostic report following the style of Spanish physicians.",

        "Given the image, please write its diagnostic report in the style of Spanish radiologists.",

        "You are provided with one chest X-ray image. Please write its diagnostic report following the guidelines "
        "of the Spanish medical system.",

        "You are provided with one chest X-ray image collected from Spain. Please write its diagnostic report.",

        "You are provided with one chest X-ray image from Spain. Compose a diagnostic report.",

        "Generate a diagnostic report for the given image as if you are a radiologist from Spain.",

        "Write a diagnostic report for this image in the style of Spanish physicians.",

        "Report Generation (PadChest)",
    ]
    target_templates = ["{report}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(report):
            return {
                "q": prompt_template,
                "a": target_template.format(report=report)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda report: [func(report) for func in form_qas[:-1]][:5]


def rib_fracture_segmentation_candid(instruct=True):
    prompt_templates = [
        "Detect rib fractures in the given image.",
        "Perform rib fracture detection for the given image.",
        "Locate rib fractures and specify their positions with bounding box coordinates",
        "Identify any rib fractures in the chest X-ray and provide the associated bounding box coordinates.",
        "Locate rib fractures in the provided chest X-ray.",
        "Find rib fractures in the given CXR.",
        "Detect the following diseases in the image: rib fracture.",
        "Locate the following diseases in the image: rib fracture.",
        "In the provided chest X-ray, identify bounding box coordinates for each rib fracture.",
        "Rib Fracture Detection (Candid).",
    ]
    target_templates = ["{boxes}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(disease, boxes):
            return {
                "q": prompt_template.format(disease=disease),
                "a": target_template.format(boxes=boxes)
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda disease, boxes: [func(disease, boxes) for func in form_qas[:-1]][:5]


def temporal_image_classification_ms_cxr_t(instruct=True):
    prompt_templates = [
        "You are given two images: one reference image and one new image. "
        "Please identify the progression of {disease}:\n{options}",

        "You are given two images: one reference image and one new image. "
        "Please identify the progression of {disease}",
    ]
    target_templates = ["{formatted_answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options, disease):
            # attach choice styles
            if "options" in prompt_template:
                options = add_choice_styles(options)
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = [k for k, v in options.items() if v >= -1]
            options_a = [k for k, v in options.items() if v >= 1]
            suffix = ""
            if "options" in prompt_template:
                if random.random() > 0.5:
                    options_a = [option.split(" ")[0] for option in options_a]
                    suffix = "\nAnswer with the option’s letter/number from the given choices directly."
            return {
                "q": prompt_template.format(disease=disease, options="\n".join(options_q)) + suffix,
                "a": target_template.format(disease=disease, formatted_answer=", ".join(options_a))
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda options, disease: [func(options, disease) for func in form_qas][:5]


def temporal_sentence_similarity_ms_cxr_t(instruct=True):
    prompt_templates = [
        "Identify the relationship between the following two sentences: Sentence 1: {a}; Sentence 2: {b} ",

        "Identify the relationship between the following two sentences: Sentence 1: {a}; Sentence 2: {b} "
        "Select from the following list:\n{options}",

        "Identify the relationship between the following two sentences: Sentence 1: {a}; Sentence 2: {b} "
        "Options:\n{options}",
    ]
    target_templates = ["{formatted_answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(texts, options):
            # attach choice styles
            if "options" in prompt_template:
                options = add_choice_styles(options)
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = [k for k, v in options.items() if v >= -1]
            options_a = [k for k, v in options.items() if v >= 1]
            suffix = ""
            if "options" in prompt_template:
                if random.random() > 0.5:
                    options_a = [option.split(" ")[0] for option in options_a]
                    suffix = "\nAnswer with the option’s letter/number from the given choices directly."
            return {
                "q": prompt_template.format(a=texts[0], b=texts[1], options="\n".join(options_q)) + suffix,
                "a": target_template.format(formatted_answer=", ".join(options_a))
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas) if instruct else form_qas[-1]
    return lambda texts, options: [func(texts, options) for func in form_qas][:5]


def text_qa_radqa(instruct=True):
    return open_ended_qa_task(instruct)


def view_classification_mimic_cxr(instruct=True):
    prompt_templates = [
        "Please identify the view of this chest X-ray. Options:\n{options}",
        "What is the view of this chest X-ray? Options:\n{options}",
        "Determine the view of this CXR",
        "Can you interpret the view of this chest X-ray?",
        "Determine the view of this the chest X-ray. Options:\n{options}",
        "Classify the view of this chest X-ray. Options: {options}",
        "Please determine the view applied when capturing this chest X-ray",
        "Indicate the view of this CXR",
        "Identify the view of the provided chest X-ray. Options: {options}",
        "View Classification (MIMIC-CXR)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options):
            # attach choice styles
            if "options" in prompt_template:
                options = add_choice_styles(options)
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = [k for k, v in options.items() if v >= -1]
            options_a = [k for k, v in options.items() if v >= 1]
            suffix = ""
            if "options" in prompt_template:
                if random.random() > 0.5:
                    options_a = [option.split(" ")[0] for option in options_a]
                    suffix = "\nAnswer with the option’s letter/number from the given choices directly."
            q = prompt_template.format(options="\n".join(options_q)) + suffix
            a = target_template.format(answer=", ".join(options_a))
            if "options" not in prompt_template:
                a = a.replace("LL", "LATERAL")
            return {"q": q, "a": a}

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: [func(options) for func in form_qas[:-1]][:5]


def view_matching_mimic_cxr(instruct=True):
    prompt_templates = [
        "In this task, you are given two chest X-rays. Decide if they belong to the same study.",
        "Do these images belong to the same study?",
        "Decide the two images come from the same study. Options:\n{options}",
        "Please assess whether these two chest X-rays are part of the same patient study.",
        "Decide if the CXRs belong to the same examination. Options:\n{options}",
        "Please examine if these chest X-rays are related to a common diagnostic session. Options: {options}",
        "Do the two images belong to the same study? Options: {options}",
        "Verify if both chest X-rays are part of a single diagnostic study",
        "In this task, you are given two chest X-rays. Decide if they belong to the same study. Options: {options}",
        "View Matching (MIMIC-CXR)",
    ]
    target_templates = ["{answer}"] * len(prompt_templates)

    def create_qa(prompt_template, target_template):
        def form_qa(options):
            # attach choice styles
            if "options" in prompt_template:
                options = add_choice_styles(options)
            # shuffle the option
            items = list(options.items())
            random.shuffle(items)
            options = dict(items)
            options_q = [k for k, v in options.items() if v >= -1]
            options_a = [k for k, v in options.items() if v >= 1]
            suffix = ""
            if "options" in prompt_template:
                if random.random() > 0.5:
                    options_a = [option.split(" ")[0] for option in options_a]
                    suffix = "\nAnswer with the option’s letter/number from the given choices directly."
            return {
                "q": prompt_template.format(options="\n".join(options_q)) + suffix,
                "a": target_template.format(answer=", ".join(options_a))
            }

        return form_qa

    form_qas = [create_qa(p, t) for p, t in zip(prompt_templates, target_templates)]
    # return random.choice(form_qas[:-1]) if instruct else form_qas[-1]
    return lambda options: [func(options) for func in form_qas[:-1]][:5]
