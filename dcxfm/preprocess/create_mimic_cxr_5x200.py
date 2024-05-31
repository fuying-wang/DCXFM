import os
import numpy as np
import random
import pandas as pd
import ipdb

np.random.seed(42)
random.seed(42)

CHEXPERT_TASKS = [
    "No Finding",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Lesion",
    "Lung Opacity",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "Support Devices",
]
CHEXPERT_COMPETITION_TASKS = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Pleural Effusion",
]
CHEXPERT_UNCERTAIN_MAPPINGS = {
    "Atelectasis": 1,
    "Cardiomegaly": 0,
    "Consolidation": 0,
    "Edema": 1,
    "Pleural Effusion": 1,
}


def main():
    mimic_dir = "/disk1/fywang/CXR_dataset/mimic_data/2.0.0"
    # mimic_dir = "/home/r15user2/Documents/CXR_dataset/mimic_data/2.0.0"
    # mimic_img_dir = "/home/r15user2/Documents/CXR_dataset/mimic_data"
    master_df = pd.read_csv(os.path.join(mimic_dir, "master.csv"))
    # keep one image one study
    master_df.drop_duplicates(subset=["subject_id", "study_id"], inplace=True)

    chexpert_df = pd.read_csv(os.path.join(
        mimic_dir, "mimic-cxr-2.0.0-chexpert.csv"))
    chexpert_df.fillna(0, inplace=True)
    # replace uncertains
    uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
    chexpert_df = chexpert_df.replace(
        uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

    master_df["report"] = master_df["impression"] + " " + master_df["findings"]
    master_df["study_id"] = master_df["study_id"].apply(lambda x: int(x[1:]))
    master_df = pd.merge(master_df, chexpert_df, how="inner", on=[
                         "subject_id", "study_id"])
    master_df = master_df[master_df["ViewPosition"].isin(["AP", "PA"])]
    # print(master_df.shape)

    merged_df = master_df[master_df["split"] == "train"]
    ms_cxr_df = pd.read_csv(
        "/disk1/fywang/CXR_dataset/ms-cxr/0.1/MS_CXR_Local_Alignment_v1.0.0.csv")
    print(merged_df.shape)
    merged_df = merged_df[~merged_df["dicom_id"].isin(ms_cxr_df["dicom_id"])]
    print(merged_df.shape)
    
    task_dfs = []
    for i, t in enumerate(CHEXPERT_COMPETITION_TASKS):
        index = np.zeros(14)
        index[i] = 1
        df_task = merged_df[
            (merged_df["Atelectasis"] == index[0])
            & (merged_df["Cardiomegaly"] == index[1])
            & (merged_df["Consolidation"] == index[2])
            & (merged_df["Edema"] == index[3])
            & (merged_df["Pleural Effusion"] == index[4])
            & (merged_df["Enlarged Cardiomediastinum"] == index[5])
            & (merged_df["Lung Lesion"] == index[7])
            & (merged_df["Lung Opacity"] == index[8])
            & (merged_df["Pneumonia"] == index[9])
            & (merged_df["Pneumothorax"] == index[10])
            & (merged_df["Pleural Other"] == index[11])
            & (merged_df["Fracture"] == index[12])
            & (merged_df["Support Devices"] == index[13])
        ]
        df_task = df_task.sample(n=200, random_state=42)
        task_dfs.append(df_task)
    df_200 = pd.concat(task_dfs)

    mimic_df_200 = df_200[["subject_id", "study_id", "dicom_id",
                           "ViewPosition", "report", "Path"] + CHEXPERT_TASKS]
    save_dir = "/disk1/fywang/CXR_dataset/preprocessed_csv/MIMIC-CXR"
    mimic_df_200.to_csv(os.path.join(
        save_dir, "mimic-cxr-5x200-val.csv"), index=False)
    
    merged_df = merged_df[~merged_df["Path"].isin(df_200["Path"])]
    train_df = merged_df[merged_df["split"] == "train"]
    
    # remove ms-cxr images
    train_df.to_csv(os.path.join(save_dir, "mimic-cxr-train-meta.csv"))
    print(train_df.shape)
    val_df = master_df[master_df["split"] == "valid"]
    val_df.to_csv(os.path.join(save_dir, "mimic-cxr-val-meta.csv"))
    print(val_df.shape)
    test_df = master_df[master_df["split"] == "test"]
    test_df.to_csv(os.path.join(save_dir, "mimic-cxr-test-meta.csv"))
    print(test_df.shape)


if __name__ == "__main__":
    main()
