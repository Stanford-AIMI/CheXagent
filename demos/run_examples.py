from model_chexagent.chexagent import CheXagent
from rich import print


def main():
    # Load the model
    chexagent = CheXagent()

    # Task 1: View Classification
    path = "https://prod-images-static.radiopaedia.org/images/23511538/8a28003cc78f3549ac9f436dfe7dad_big_gallery.jpeg"
    response = chexagent.view_classification(path)
    print(f'=' * 42)
    print(f'[Task 1: View Classification]')
    print(f'Image: {path}')
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 2: View Matching
    path_1 = "https://prod-images-static.radiopaedia.org/images/17054297/07b3ca19d485b21a30bd8412dbbc33_big_gallery.jpeg"
    path_2 = "https://prod-images-static.radiopaedia.org/images/17054298/c42c7d29bfe0203554649e38e9be6c_big_gallery.jpeg"
    response = chexagent.view_matching([path_1, path_2])
    print(f'=' * 42)
    print(f'[Task 2: View Matching]')
    print(f'Image 1: {path_1}')
    print(f'Image 2: {path_2}')
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 3: Binary Disease Classification
    path = "https://prod-images-static.radiopaedia.org/images/23511538/8a28003cc78f3549ac9f436dfe7dad_big_gallery.jpeg"
    response = chexagent.binary_disease_classification([path], "Pneumothorax")
    print(f'=' * 42)
    print(f'[Task 3: Binary Disease Classification]')
    print(f'Image: {path}')
    print(f'Evaluating for: Pneumothorax')
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 4: Disease Identification
    path = "https://prod-images-static.radiopaedia.org/images/23511538/8a28003cc78f3549ac9f436dfe7dad_big_gallery.jpeg"
    response = chexagent.disease_identification([path], ["Pneumothorax", "Pleural Effusion", "Pneumonia"])
    print(f'=' * 42)
    print(f'[Task 4: Disease Identification]')
    print(f'Image: {path}')
    print(f'Options: ["Pneumothorax", "Pleural Effusion", "Pneumonia"]')
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 5: Findings Generation
    path = "https://prod-images-static.radiopaedia.org/images/23511538/8a28003cc78f3549ac9f436dfe7dad_big_gallery.jpeg"
    indication = "Shortness of breath and mild left sided pleuritic chest pain. No trauma."
    response = chexagent.findings_generation([path], indication)
    print(f'=' * 42)
    print(f'[Task 5: Findings Generation]')
    print(f'Image: {path}')
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 6: Findings Generation Section by Section
    path = "https://prod-images-static.radiopaedia.org/images/23511538/8a28003cc78f3549ac9f436dfe7dad_big_gallery.jpeg"
    responses = chexagent.findings_generation_section_by_section([path])
    print(f'=' * 42)
    print(f'[Task 6: Findings Generation Section by Section]')
    print(f'Image: {path}')
    print(f'Result:')
    for response in responses:
        print(f'{response[0]}: {response[1]}')
    print(f'=' * 42)

    # Task 7: Image Text Matching
    path = "https://prod-images-static.radiopaedia.org/images/23511538/8a28003cc78f3549ac9f436dfe7dad_big_gallery.jpeg"
    text = "No pneumothorax."
    response = chexagent.image_text_matching([path], text)
    print(f'=' * 42)
    print(f'[Task 7: Image Text Matching]')
    print(f'Image: {path}')
    print(f'Text: {text}')
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 8: Phrase Grounding
    path = "https://prod-images-static.radiopaedia.org/images/23511538/8a28003cc78f3549ac9f436dfe7dad_big_gallery.jpeg"
    phrase = "There is a left apical pneumothorax."
    print(f'=' * 42)
    print(f'[Task 8: Phrase Grounding]')
    print(f'Image: {path}')
    print(f'Phrase: {phrase}')
    response = chexagent.phrase_grounding(path, phrase)
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 9: Abnormality Detection
    path = "https://prod-images-static.radiopaedia.org/images/23511538/8a28003cc78f3549ac9f436dfe7dad_big_gallery.jpeg"
    disease_name = "Pneumothorax"
    print(f'=' * 42)
    print(f'[Task 9: Abnormality Detection]')
    print(f'Image: {path}')
    print(f'Disease Name: {disease_name}')
    response = chexagent.abnormality_detection(path, disease_name)
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 10: Chest Tube Detection
    path = "https://medschool.co/images/detail/cxr/chest-drain.jpg"
    print(f'=' * 42)
    print(f'[Task 10: Chest Tube Detection]')
    print(f'Image: {path}')
    response = chexagent.chest_tube_detection(path)
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 11: Rib Fracture Detection
    path = "https://prod-images-static.radiopaedia.org/images/8719062/16f627c16c8b34bccd8857b95a259f_big_gallery.jpg"
    print(f'=' * 42)
    print(f'[Task 11: Rib Fracture Detection]')
    print(f'Image: {path}')
    response = chexagent.rib_fracture_detection(path)
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 12: Foreign Object Detection
    path = "https://images.squarespace-cdn.com/content/v1/56e8a86a746fb97ea9d14740/1517332911100-IZP0K9MTNA6M9Z1LHFF3/FB+CXR.png?format=2500w"
    print(f'=' * 42)
    print(f'[Task 12: Foreign Object Detection]')
    print(f'Image: {path}')
    response = chexagent.foreign_objects_detection(path)
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 13: Temporal Image Classification
    prior_image_path = "https://prod-images-static.radiopaedia.org/images/23511538/8a28003cc78f3549ac9f436dfe7dad_big_gallery.jpeg"
    current_image_path = "https://prod-images-static.radiopaedia.org/images/23511538/8a28003cc78f3549ac9f436dfe7dad_big_gallery.jpeg"
    response = chexagent.temporal_image_classification([prior_image_path, current_image_path], "Pneumothorax")
    print(f'=' * 42)
    print(f'[Task 13: Temporal Image Classification]')
    print(f'Prior Image: {prior_image_path}')
    print(f'Current Image: {current_image_path}')
    print(f'Evaluating for: The progression of Pneumothorax')
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 14: Findings Summarization
    findings = ("The endotracheal tube tip sits 5 cm above the carina. "
                "A left-sided IJ central venous catheter tip sits in the left brachiocephalic vein. "
                "The right-sided IJ central venous catheter tip sits in the upper SVC. "
                "The heart size is large but stable. The mediastinal contours are within normal limits. "
                "There continue to be bibasilar and perihilar opacities as well as a more rounded confluent opacity in "
                "the right upper lung. These findings likely represent increased pulmonary edema as well as right "
                "upper and lower lobe consolidations. Retrocardiac opacity is also compatible with a left lower lobe "
                "consolidation. The costophrenic angles are excluded from the study limiting assessment for subtle "
                "pleural effusion. There is no large pneumothorax.")
    response = chexagent.findings_summarization(findings)
    print(f'=' * 42)
    print(f'[Task 14: Findings Summarization]')
    print(f'Findings: {findings}')
    print(f'Result: {response}')
    print(f'=' * 42)

    # Task 15: Named Entity Recognition
    text = "Right lower lobe opacity, concerning for infection."
    response = chexagent.named_entity_recognition(text)
    print(f'=' * 42)
    print(f'[Task 15: Named Entity Recognition]')
    print(f'text: {text}')
    print(f'Result: {response}')
    print(f'=' * 42)

if __name__ == '__main__':
    main()
