import pickle
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import ipdb

np.random.seed(0)


def main():
    # dataset_dir = "/data1/r20user2/CXR_dataset/RSNA_Pneumonia"
    dataset_dir = "/disk1/fywang/CXR_dataset/RSNA_Pneumonia"
    df = pd.read_csv(f"{dataset_dir}/stage_2_train_labels.csv")

    lovt_dir = "/home/fywang/Documents/lovt/datasets/RSNA-Pneunomia-Detection"
    train_ids = pd.read_csv(os.path.join(lovt_dir, "train.csv"))["ID"].values
    val_ids = pd.read_csv(os.path.join(lovt_dir, "validation.csv"))["ID"].values
    test_ids = pd.read_csv(os.path.join(lovt_dir, "test.csv"))["ID"].values

    train_df = df.loc[df["patientId"].isin(train_ids)]
    val_df = df.loc[df["patientId"].isin(val_ids)]
    test_df = df.loc[df["patientId"].isin(test_ids)]
    
    print(f"Train: {len(train_df)}")
    print(f"Val: {len(val_df)}")
    print(f"Test: {len(test_df)}")
    
    save_dir = "/disk1/fywang/CXR_dataset/preprocessed_csv/RSNA"
    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(f"{save_dir}/RSNA_train.csv", index=False)
    val_df.to_csv(f"{save_dir}/RSNA_val.csv", index=False)
    test_df.to_csv(f"{save_dir}/RSNA_test.csv", index=False)


def create_medklip_split():
    sample_df = pd.read_csv("/home/fywang/Documents/MedKLIP/Sample_Zero-Shot_Grounding_RSNA/data_sample/test.csv")
    test_ids = sample_df["ID"].values
    print(len(test_ids))

    dataset_dir = "/disk1/fywang/CXR_dataset/RSNA_Pneumonia"
    df = pd.read_csv(f"{dataset_dir}/stage_2_train_labels.csv")
    test_df = df.loc[df["patientId"].isin(test_ids)]
    print(len(test_df))

    save_dir = "/disk1/fywang/CXR_dataset/preprocessed_csv/RSNA"
    os.makedirs(save_dir, exist_ok=True)
    test_df.to_csv(f"{save_dir}/RSNA_medklip_test.csv", index=False)


if __name__ == '__main__':
    # main()
    create_medklip_split()