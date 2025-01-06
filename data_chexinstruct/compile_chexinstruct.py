import os.path
import pickle

import pandas as pd
import tqdm

from dataset_processors import *


def main():
    # constant
    save_dir = "data_chexinstruct"

    # load processors
    bimcv_covid19_processor = BIMCVCOVID19Processor()
    brax_processor = BraxProcessor()
    candid_processor = CandidProcessor()
    chestxray14_processor = ChestXray14Processor()
    chexpert_public_processor = CheXpertPublicProcessor()
    covidx_cxr_3_processor = COVIDXCXR3Processor()
    cxr_lt_processor = CXRLTProcessor()
    medvqa_2019_processor = MedVQA2019Processor()
    mimic_cxr_processor = MIMICCXRProcessor()
    mimic_cxr_struct_processor = MIMICCXRStructProcessor()
    mimic_cxr_vqa_processor = MIMICCXRVQAProcessor()
    mimic_diff_vqa_processor = MIMICDiffVQAProcessor()
    mimic_iii_processor = MIMICIIIProcessor()
    mimic_nle_processor = MIMICNLEProcessor()
    ms_cxr_processor = MSCXRProcessor()
    ms_cxr_t_processor = MSCXRTProcessor()
    nlm_tb_processor = NLMTBProcessor()
    object_cxr_processor = ObjectCXRProcessor()
    openi_processor = OpenIProcessor()
    padchest_processor = PadChestProcessor()
    pmc_vqa_processor = PMCVQAProcessor()
    rad_restruct_processor = RadRestructProcessor()
    radgraph_processor = RadGraphProcessor()
    radnli_processor = RadNLIProcessor()
    radqa_processor = RadQAProcessor()
    rexval_processor = ReXValProcessor()
    roco_processor = ROCOProcessor()
    rsna_processor = RSNAProcessor()
    siim_processor = SIIMProcessor()
    slake_processor = SLAKEProcessor()
    vindr_cxr_processor = VinDRCXRProcessor()
    vindr_pcxr_processor = VinDRPCXRProcessor()
    vqa_rad_processor = VQARADProcessor()

    # load datasets
    datasets = [
        # Coarse-grained Image Perception: Image Classification
        chestxray14_processor.create_image_classification(),
        chexpert_public_processor.create_image_classification(),
        mimic_cxr_processor.create_image_classification(),
        padchest_processor.create_image_classification(),
        rsna_processor.create_image_classification(),
        covidx_cxr_3_processor.create_image_classification(),
        cxr_lt_processor.create_image_classification(),
        brax_processor.create_image_classification(),
        nlm_tb_processor.create_image_classification(),
        # Coarse-grained Image Perception: Temporal Image Classification
        ms_cxr_t_processor.create_temporal_image_classification(),
        # Coarse-grained Image Perception: View Classification
        mimic_cxr_processor.create_view_classification(),
        chexpert_public_processor.create_view_classification(),
        # Coarse-grained Image Perception: View Matching
        mimic_cxr_processor.create_view_matching(),
        # Fine-grained Image Perception: Abnormality Detection
        vindr_cxr_processor.create_abnormality_detection(),
        vindr_pcxr_processor.create_abnormality_detection(),
        # Fine-grained Image Perception: Abnormality Grounding
        vindr_cxr_processor.create_abnormality_grounding(),
        vindr_pcxr_processor.create_abnormality_grounding(),
        # Fine-grained Image Perception: Pneumothorax Segmentation
        candid_processor.create_pneumothorax_segmentation(),
        siim_processor.create_pneumothorax_segmentation(),
        # Fine-grained Image Perception: Rib Fracture Segmentation
        candid_processor.create_rib_fracture_segmentation(),
        # Fine-grained Image Perception: Chest Tube Segmentation
        candid_processor.create_chest_tube_segmentation(),
        # Fine-grained Image Perception: Foreign Object Detection
        object_cxr_processor.create_foreign_objects_detection(),
        # Fine-grained Image Perception: Phrase Grounding
        ms_cxr_processor.create_phrase_grounding(),
        # Fine-grained Image Perception: Grounded Captioning
        ms_cxr_processor.create_grounded_captioning(),
        # Fine-grained Image Perception: Grounded Classification
        ms_cxr_processor.create_grounded_diagnosis(),
        vindr_cxr_processor.create_grounded_diagnosis(),
        vindr_pcxr_processor.create_grounded_diagnosis(),
        # Fine-grained Image Perception: Grounded Phrase Extraction
        ms_cxr_processor.create_grounded_phrase_extraction(),
        # Text Generation: Findings Generation
        mimic_cxr_struct_processor.create_findings_generation(),
        mimic_cxr_struct_processor.create_findings_generation_with_indication(),
        chexpert_public_processor.create_findings_generation(),
        chexpert_public_processor.create_findings_generation_with_indication(),
        openi_processor.create_findings_generation(),
        # Text Generation: Impression Generation
        mimic_cxr_struct_processor.create_impression_generation(),
        mimic_cxr_struct_processor.create_impression_generation_with_indication(),
        chexpert_public_processor.create_impression_generation(),
        chexpert_public_processor.create_impression_generation_with_indication(),
        openi_processor.create_impression_generation(),
        candid_processor.create_impression_generation(),
        padchest_processor.create_impression_generation(),
        bimcv_covid19_processor.create_impression_generation(),
        # Text Generation: Progression Findings Generation
        mimic_cxr_struct_processor.create_progression_findings_generation(),
        chexpert_public_processor.create_progression_findings_generation(),
        # Text Generation: Progression Impression Generation
        mimic_cxr_struct_processor.create_progression_impression_generation(),
        chexpert_public_processor.create_progression_impression_generation(),
        # Text Generation: Local Findings Generation
        mimic_cxr_struct_processor.create_local_findings_generation(),
        chexpert_public_processor.create_local_findings_generation(),
        # Text Generation: Local Impression Generation
        mimic_cxr_struct_processor.create_local_impression_generation(),
        chexpert_public_processor.create_local_impression_generation(),
        # Text Generation: Local Progression Findings Generation
        mimic_cxr_struct_processor.create_local_progression_findings_generation(),
        chexpert_public_processor.create_local_progression_findings_generation(),
        # Text Generation: Local Progression Impression Generation
        mimic_cxr_struct_processor.create_local_progression_impression_generation(),
        chexpert_public_processor.create_local_progression_impression_generation(),
        # Text Generation: Caption Generation
        roco_processor.create_caption_generation(),
        # Text Generation: Findings Summarization
        mimic_cxr_processor.create_findings_summarization(),
        openi_processor.create_findings_summarization(),
        mimic_iii_processor.create_findings_summarization(),
        # Question Answering: Open-ended VQA
        vqa_rad_processor.create_open_ended_vqa(),
        slake_processor.create_open_ended_vqa(),
        medvqa_2019_processor.create_open_ended_vqa(),
        pmc_vqa_processor.create_open_ended_vqa(),
        rad_restruct_processor.create_open_ended_vqa(),
        mimic_cxr_vqa_processor.create_open_ended_vqa(),
        # Question Answering: Close-ended VQA
        vqa_rad_processor.create_close_ended_vqa(),
        slake_processor.create_close_ended_vqa(),
        medvqa_2019_processor.create_close_ended_vqa(),
        pmc_vqa_processor.create_close_ended_vqa(),
        rad_restruct_processor.create_close_ended_vqa(),
        mimic_cxr_vqa_processor.create_close_ended_vqa(),
        # Question Answering: Difference VQA
        mimic_diff_vqa_processor.create_difference_vqa(),
        # Question Answering: Text VQA
        radqa_processor.create_text_qa(),
        # Miscellaneous: Image-Text Matching
        mimic_cxr_processor.create_image_text_matching(),
        roco_processor.create_image_text_matching(),
        # Miscellaneous: Image-Text Selection
        mimic_cxr_processor.create_image_text_selection(),
        roco_processor.create_image_text_selection(),
        # Miscellaneous: Report Evaluation
        rexval_processor.create_report_evaluation(),
        # Miscellaneous: Natural Language Explanation
        mimic_nle_processor.create_natural_language_explanation(),
        # Miscellaneous: Natural Language Inference
        radnli_processor.create_natural_language_inference(),
        # Miscellaneous: Temporal Sentence Similarity
        ms_cxr_t_processor.create_temporal_sentence_similarity(),
        # Miscellaneous: Named Entity Recognition
        radgraph_processor.create_named_entity_recognition(),
    ]

    # Save the annotation file
    with open(f"{save_dir}/data_chexinstruct.json", "wb") as f:
        pickle.dump(datasets, f)

    # Print dataset information
    total_studies_train, total_studies_val, total_studies_test = [], [], []
    total_images_train, total_images_val, total_images_test = [], [], []
    dataset_info = dict()
    for dataset in datasets:
        images = {"train": [], "val": [], "test": []}
        for split, total_num_images in zip(["train", "val", "test"],
                                           [total_images_train, total_images_val, total_images_test]):
            for sample in dataset[split]:
                if "image_path" in sample and sample["image_path"]:
                    if isinstance(sample["image_path"], list):
                        images[split].extend(sample["image_path"])
                        total_num_images.extend(sample["image_path"])
                    else:
                        images[split].append(sample["image_path"])
                        total_num_images.append(sample["image_path"])
        total_studies_train.extend(list(set([item["study_id"] for item in dataset["train"]])))
        total_studies_val.extend(list(set([item["study_id"] for item in dataset["val"]])))
        total_studies_test.extend(list(set([item["study_id"] for item in dataset["test"]])))

        dataset_info[dataset["dataset_name"]] = {
            # Instances
            "train": len(dataset["train"]),
            "val": len(dataset["val"]),
            "test": len(dataset["test"]),
            # Images
            "images (train)": len(set(images["train"])),
            "images (val)": len(set(images["val"])),
            "images (test)": len(set(images["test"])),
            # Studies
            "studies (train)": len(set([item["study_id"] for item in dataset["train"]])),
            "studies (val)": len(set([item["study_id"] for item in dataset["val"]])),
            "studies (test)": len(set([item["study_id"] for item in dataset["test"]])),
        }

        print(f'[{dataset["dataset_name"]}] '
              f'Train: {dataset_info[dataset["dataset_name"]]["train"]}, '
              f'Val: {dataset_info[dataset["dataset_name"]]["val"]}, '
              f'Test: {dataset_info[dataset["dataset_name"]]["test"]}, '
              f'Num Images (Train): {dataset_info[dataset["dataset_name"]]["images (train)"]}, '
              f'Num Images (Val): {dataset_info[dataset["dataset_name"]]["images (val)"]}, '
              f'Num Images (Test): {dataset_info[dataset["dataset_name"]]["images (test)"]}, '
              f'Num Studies (Train): {dataset_info[dataset["dataset_name"]]["studies (train)"]}, '
              f'Num Studies (Val): {dataset_info[dataset["dataset_name"]]["studies (val)"]}, '
              f'Num Studies (Test): {dataset_info[dataset["dataset_name"]]["studies (test)"]}.')

    df_dataset_info = pd.DataFrame(dataset_info)
    df_dataset_info["Total"] = df_dataset_info.sum(1)
    batch = total_images_train, total_images_val, total_images_test
    df_dataset_info.loc[
        ["images (train)", "images (val)", "images (test)"], "Total"] = [len(set(item)) for item in batch]
    batch = total_studies_train, total_studies_val, total_studies_test
    df_dataset_info.loc[
        ["studies (train)", "studies (val)", "studies (test)"], "Total"] = [len(set(item)) for item in batch]
    print(df_dataset_info.applymap(lambda x: f"{x:,}" if isinstance(x, int) else x).T)

    # Print to files
    fin = open(f"{save_dir}/DATA.md", "wt")
    fin.write("# Instruction Tuning Data for CXR\n\n")
    fin.write("## Data Summary\n\n")
    fin.write(
        df_dataset_info["Total"].map(lambda x: f"{x:,}" if isinstance(x, int) else x).astype(str).to_markdown()
    )
    fin.write("\n\n## All the datasets\n\n")
    fin.write(df_dataset_info.applymap(lambda x: f"{x:,}" if isinstance(x, int) else x).T.astype(str).to_markdown())
    fin.close()

    # Confirm all the image paths exist
    for dataset in tqdm.tqdm(datasets):
        for split in ["train", "val", "test"]:
            for sample in dataset[split]:
                if "image_path" in sample and sample["image_path"]:
                    if isinstance(sample["image_path"], list):
                        for path in sample["image_path"]:
                            if not os.path.exists(path):
                                print(f'{dataset["dataset_name"]}: {path} does not exist.')
                    else:
                        if not os.path.exists(sample["image_path"]):
                            print(f'{dataset["dataset_name"]}: {sample["image_path"]} does not exist.')


if __name__ == '__main__':
    main()
