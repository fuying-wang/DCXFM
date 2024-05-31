import os
import numpy as np
import pandas as pd
import pickle
import torch
from evaluate_zero_shot_seg import compute_seg_metrics
import ipdb


def main():
    pred_file = "/home/fywang/Documents/CXRSeg/scripts/output/2024-05-06_15-11-51/inference/mscxr_val_sem_seg_predictions.pkl"
    with open(pred_file, 'rb') as f:
        pred = pickle.load(f)

    all_preds, gt_masks, all_labels = [], [], []
    all_imgids = []
    for sample in pred:
        label = sample['gt_classes']
        pred_map = sample["sem_seg"][label]
        file_name = sample['file_name']
        imgid = file_name.replace(
            "/disk1/fywang/CXR_dataset/mask/MaCheX/images/MS_CXR_Dataset_", "").replace(".png", "")
        all_imgids.append(imgid)
        all_preds.append(pred_map)
        gt_masks.append(sample['gt_masks'].astype(np.float32))
        all_labels.append(sample['gt_classes'])

    all_preds = torch.from_numpy(
        np.concatenate(all_preds, axis=0)).unsqueeze(1)
    gt_masks = torch.from_numpy(np.concatenate(gt_masks, axis=0)).unsqueeze(1)
    all_labels = torch.from_numpy(
        np.concatenate(all_labels, axis=0)).unsqueeze(1)
    class_names = ["Atelectasis",
                   "Cardiomegaly",
                   "Consolidation",
                   "Edema",
                   "Lung Opacity",
                   "Pleural Effusion",
                   "Pneumonia",
                   "Pneumothorax"]
    save_dir = "stage2"
    os.makedirs(save_dir, exist_ok=True)
    save_data_dict = {
        "all_pred": all_preds,
        "gt_masks": gt_masks,
        "all_labels": all_labels,
        "class_names": class_names,
        "all_imgids": all_imgids
    }
    torch.save(save_data_dict, os.path.join(save_dir, "saved_data.pth"))
    compute_seg_metrics(
        all_preds, gt_masks, all_labels, class_names=class_names, save_dir="stage2")


if __name__ == '__main__':
    main()
