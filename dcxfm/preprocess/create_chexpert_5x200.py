import os
import random
import numpy as np
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
    # chexpert_dir = "/home/r15user2/Documents/CXRSeg/cxr_data/CheXpert-v1.0"
    dataset_dir = "/disk1/fywang/CXR_dataset"
    chexpert_df = pd.read_csv(os.path.join(dataset_dir, "CheXpert-v1.0/train.csv"))
    chexpert_df.fillna(0, inplace=True)
    # replace uncertains
    uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
    chexpert_df = chexpert_df.replace(
        uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)
    chexpert_df = chexpert_df.loc[chexpert_df["Frontal/Lateral"] == "Frontal"]
    # chexpert_df["imgpath"] = chexpert_df["Path"].apply(lambda x: os.path.join(chexpert_dir, "/".join(x.split("/")[1:])))
    # chexpert_df.drop(columns=["Path"], inplace=True)
    
    task_dfs = []
    for i, t in enumerate(CHEXPERT_COMPETITION_TASKS):
        index = np.zeros(14)
        index[i] = 1
        df_task = chexpert_df[
            (chexpert_df["Atelectasis"] == index[0])
            & (chexpert_df["Cardiomegaly"] == index[1])
            & (chexpert_df["Consolidation"] == index[2])
            & (chexpert_df["Edema"] == index[3])
            & (chexpert_df["Pleural Effusion"] == index[4])
            & (chexpert_df["Enlarged Cardiomediastinum"] == index[5])
            & (chexpert_df["Lung Lesion"] == index[7])
            & (chexpert_df["Lung Opacity"] == index[8])
            & (chexpert_df["Pneumonia"] == index[9])
            & (chexpert_df["Pneumothorax"] == index[10])
            & (chexpert_df["Pleural Other"] == index[11])
            & (chexpert_df["Fracture"] == index[12])
            & (chexpert_df["Support Devices"] == index[13])
        ]
        df_task = df_task.sample(n=200, random_state=42)
        task_dfs.append(df_task)
    df_200 = pd.concat(task_dfs)

    chexpert_df = chexpert_df[~chexpert_df["Path"].isin(df_200["Path"])]

    save_dir = os.path.join(dataset_dir, "preprocessed_csv/CheXpert")
    os.makedirs(save_dir, exist_ok=True)
    df_200.to_csv(os.path.join(save_dir, "chexpert_5x200.csv"), index=False)
    chexpert_df.to_csv(os.path.join(
        save_dir, "chexpert_train.csv"), index=False)
    
    # only keep the frontal images
    val_df = pd.read_csv(os.path.join(dataset_dir, "CheXpert-v1.0/valid.csv"))
    val_df = val_df.loc[val_df["Frontal/Lateral"] == "Frontal"]
    val_df.to_csv(os.path.join(save_dir, "chexpert_val.csv"), index=False)

    test_df = pd.read_csv(os.path.join(dataset_dir, "CheXpert/test_labels.csv"))
    test_df["view"] = test_df["Path"].apply(lambda x: os.path.splitext(x)[0].split("_")[-1])
    test_df = test_df.loc[test_df["view"] == "frontal"]
    test_df.to_csv(os.path.join(save_dir, "chexpert_test.csv"), index=False)


if __name__ == "__main__":
    main()
