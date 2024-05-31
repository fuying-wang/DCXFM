'''
This script is used to translate reports in PadChest dataset from Spanish to English.
The translation API is from deep_translator.
'''
from ast import literal_eval
import numpy as np
import pandas as pd
from deep_translator import GoogleTranslator
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from glob import glob
import os
import multiprocessing as mp
import ipdb 
import warnings
from dcxfm.utils.constants import CHEXPERT_COMPETITION_TASKS
warnings.filterwarnings('ignore')


def translate_report():
    padchest_csvpath = "/home/r15user2/Documents/CXRSeg/cxr_data/PadChest/BIMCV-PadChest-FULL/"\
                       "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv.gz"
    df = pd.read_csv(padchest_csvpath, low_memory=False)

    translated_df = pd.read_csv("/home/r15user2/Documents/CXRSeg/cxrseg/tools/local_data/padchest_eng.csv")
    translation_dict = dict(zip(translated_df["Report"].tolist(), translated_df["Eng_Report"].tolist()))
    df["Report_Eng"] = df["Report"].map(translation_dict)
    df.to_csv(os.path.join(os.path.dirname(padchest_csvpath), "PADCHEST_chest_x_ray_translated.csv"), index=False)

    # df = df.dropna(subset=["Report"])
    # translated_df = pd.read_csv("local_data/padchest_eng.csv")

    # df = df.loc[~df["Report"].isin(translated_df["Report"].tolist())]
    # df = df.drop_duplicates(subset=["Report"])
    # print(df.shape)

    # translator = GoogleTranslator(source='spanish', target='english')

    # chunk_size = 50
    # temp_save_dir = "./"

    # def worker_fn(i):
    #     save_file_name = temp_save_dir + str(i) + ".csv"
    #     if os.path.exists(save_file_name):
    #         print("Skip", save_file_name)
    #         return
    #     chunk = df.iloc[i:i+chunk_size]
    #     all_reports = chunk["Report"].tolist()
    #     try:
    #         translated_reports = translator.translate_batch(all_reports)
    #     except Exception as e:
    #         print(e)
    #         return
    #     chunk["Eng_Report"] = translated_reports
    #     chunk = chunk[["ImageID", "Report", "Eng_Report"]]
    #     chunk.to_csv(temp_save_dir + str(i) + ".csv", index=False)

    # with mp.Pool(processes=48) as p:
    #     max_ = len(df)
    #     print(max_)
    #     with tqdm(total=max_ // chunk_size) as pbar:
    #         for _ in p.imap_unordered(worker_fn, range(0, max_, chunk_size)):
    #             pbar.update()


    # csv_files = glob("./*.csv")
    # df = pd.concat([pd.read_csv(f) for f in csv_files])

    # finished_df = pd.read_csv("local_data/padchest_eng.csv")
    # print(len(finished_df))
    # finished_df = pd.concat([finished_df, df])
    # print(len(finished_df))
    # finished_df.to_csv("local_data/padchest_eng.csv", index=False)


def create_train_test_split():
    ''' Maybe we need to reformulate this function ...'''
    # dataset_dir = "/home/r15user2/cxr_data"
    dataset_dir = "/disk1/fywang/CXR_dataset"
    csvpath = os.path.join(dataset_dir, "PadChest/BIMCV-PadChest-FULL/PADCHEST_chest_x_ray_translated.csv")
    df = pd.read_csv(csvpath, low_memory=False)
    df = df[df["Projection"].isin(["PA", "AP"])]
    df.reset_index(drop=True, inplace=True)
    print(df.shape)

    # ================================================
    # comment this snippet of code since we are using file from KAD
    # ================================================
    # manual_label_df = df[df["MethodLabel"] == "Physician"]
    # # manual_label_df: (24536, 37)
    # # manual_label_df["Labels"]
    # all_labels = []
    # for i in range(len(manual_label_df)):
    #     labels = literal_eval(manual_label_df["Labels"].iloc[i])
    #     all_labels.extend(labels)
    # print(len(set(all_labels)))

    # 192
    test_csv_path = os.path.join(dataset_dir,
                                 "preprocessed_csv/A1_DATA/PadChest/Physician_label193_all.csv")
    test_df = pd.read_csv(test_csv_path)
    # test_df = test_df[test_df["Projection"].isin(["PA", "AP"])]
    # remove the prefix of the image path
    test_df["img_path"] = test_df["img_path"].apply(lambda x: 
        x.replace("/mnt/petrelfs/zhangxiaoman/DATA/Chestxray/PadChest/images/", ""))
    test_df["img_path"] = test_df["img_path"].apply(lambda x: x.split("/")[1])
    test_df.rename(columns={"img_path": "ImageID"}, inplace=True)    
    test_ori_df = df[df["ImageID"].isin(test_df["ImageID"].tolist())]
    test_imgids = test_ori_df["ImageID"].values.tolist()
    test_df = test_df[test_df["ImageID"].isin(test_imgids)] # 24536

    train_val_df = df[~df["ImageID"].isin(test_df["ImageID"].tolist())]   # (71751, 37)
    train_val_patient_ids = train_val_df["PatientID"].unique()
    train_ids, val_ids = train_test_split(train_val_patient_ids, test_size=0.2, random_state=42)
    train_df = train_val_df[train_val_df["PatientID"].isin(train_ids)]    # (57342, 37)
    val_df = train_val_df[train_val_df["PatientID"].isin(val_ids)]        # (14409, 37)

    train_df.reset_index(drop=True, inplace=True) # (57342, 37)
    val_df.reset_index(drop=True, inplace=True)   # (14409, 37)
    test_df.reset_index(drop=True, inplace=True)  # (39053, 196)
    test_df.rename(columns={"ImageID": "Path"}, inplace=True)
    print(train_df.shape, val_df.shape, test_df.shape)

    save_dir = os.path.join(dataset_dir, "preprocessed_csv/PadChest")
    os.makedirs(save_dir, exist_ok=True)
    train_df.to_csv(os.path.join(save_dir, "padchest_train.csv"), index=False)
    val_df.to_csv(os.path.join(save_dir, "padchest_val.csv"), index=False)
    test_df.to_csv(os.path.join(save_dir, "padchest_test.csv"), index=False)


def analyze_padchest_pathologies():
    dataset_dir = "/disk1/fywang/CXR_dataset"
    save_dir = os.path.join(dataset_dir, "preprocessed_csv/PadChest")
    csv_path = os.path.join(save_dir, "padchest_test.csv")
    df = pd.read_csv(csv_path)
    # pathologies = df.columns.tolist()[3:]
    # print(df.describe())
    label_df = df.drop(columns=["Path", "Labels", "labelCUIS"])
    print(label_df.sum(axis=0).sort_values(ascending=False))
    ipdb.set_trace()


if __name__ == "__main__":
    # translate_report()
    create_train_test_split()
    # analyze_padchest_pathologies()