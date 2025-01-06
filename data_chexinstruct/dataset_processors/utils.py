import re

import pandas as pd


def create_dicom_to_path_mapping():
    image_dir = "data/mimic-cxr/files"
    meta_path = "data/mimic-cxr/mimic-cxr-2.0.0-metadata.csv"

    data = pd.read_csv(meta_path)
    data["path"] = image_dir + "/p" + data["subject_id"].map(lambda x: str(x)[:2]) + "/p" + data["subject_id"].astype(
        int).astype(str) + "/s" + data["study_id"].astype(int).astype(str) + "/" + data["dicom_id"] + ".jpg"
    data = data.set_index("dicom_id")
    dicom2path = data["path"].to_dict()
    data = data.reset_index("dicom_id").set_index("path")
    path2dicom = data["dicom_id"].to_dict()
    return dicom2path, path2dicom


def create_dicom_to_meta_mapping():
    meta_path = "data/mimic-cxr/mimic-cxr-2.0.0-metadata.csv"

    data = pd.read_csv(meta_path)
    data = data.set_index("dicom_id")
    return data


def create_study_to_texts_mapping():
    meta_path = "data/mimic-cxr/mimic-cxr/txt/mimic_cxr_sectioned.csv"
    data = pd.read_csv(meta_path)
    data = data.set_index("study")
    data.impression = data.impression.map(lambda x: re.sub("\s+", " ", x), na_action="ignore")
    data.findings = data.findings.map(lambda x: re.sub("\s+", " ", x), na_action="ignore")
    data.last_paragraph = data.last_paragraph.map(lambda x: re.sub("\s+", " ", x), na_action="ignore")
    data.comparison = data.comparison.map(lambda x: re.sub("\s+", " ", x), na_action="ignore")
    data.indication = data.indication.map(lambda x: re.sub("\s+", " ", x), na_action="ignore")
    return data


def create_study_to_paths_mapping():
    meta_path = "data/mimic-cxr/mimic-cxr-2.0.0-metadata.csv"
    data = pd.read_csv(meta_path)
    data = data.set_index("study_id")
    dicom2path = create_dicom_to_path_mapping()[0]
    ranked_views = ['PA', 'AP', 'LATERAL', 'LL', 'AP AXIAL', 'AP LLD', 'AP RLD', 'PA RLD', 'PA LLD', 'LAO', 'RAO',
                    'LPO', 'XTABLE LATERAL', 'SWIMMERS', '']
    data["ViewPosition"] = data.ViewPosition.fillna("")
    data["ViewPositionRank"] = data["ViewPosition"].map(lambda x: ranked_views.index(x))
    study2paths = {}
    for study_id, groups in data.groupby(level=0):
        view_ranks = groups['ViewPositionRank'].to_list()
        dicom_ids = groups['dicom_id'].to_list()
        sorted_pairs = sorted(zip(view_ranks, dicom_ids))
        sorted_dicom_ids = [item[1] for item in sorted_pairs]
        paths = [dicom2path[dicom_id] for dicom_id in sorted_dicom_ids]
        if paths:
            study2paths[study_id] = paths
    return study2paths


def create_study_to_split_mapping():
    data_dir = "data/mimic-cxr"
    split_path = f"{data_dir}/mimic-cxr-2.0.0-split.csv"
    split_data = pd.read_csv(split_path)
    split_data.loc[split_data.split == "validate", "split"] = "val"
    study2split = split_data.drop_duplicates(subset=["study_id"])
    study2split = study2split.set_index("study_id")
    study2split = study2split["split"].to_dict()
    return study2split


if __name__ == '__main__':
    create_dicom_to_path_mapping()
    create_dicom_to_meta_mapping()
    create_study_to_texts_mapping()
    create_study_to_paths_mapping()
    create_study_to_split_mapping()
