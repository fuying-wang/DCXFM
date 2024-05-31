'''
The script is used to load the segmentation masks of CheXlocalize dataset.
'''
import os
import json
import pickle
import ipdb
from tqdm import tqdm
from pycocotools import mask as cocomask

localization_tasks = ["Enlarged Cardiomediastinum",
                      "Cardiomegaly",
                      "Lung Lesion",
                      "Airspace Opacity",
                      "Edema",
                      "Consolidation",
                      "Atelectasis",
                      "Pneumothorax",
                      "Pleural Effusion",
                      "Support Devices"]
localization_tasks = sorted(localization_tasks)
save_path = "/home/r15user2/Documents/CXRSeg/cxr_data/mask/CheXlocalize"
# the directory to save masks
os.makedirs(save_path, exist_ok=True)


def main(split):
    os.makedirs(os.path.join(save_path, split), exist_ok=True)

    segpath = f"/home/r15user2/Documents/CXRSeg/cxr_data/CheXlocalize/gt_segmentations_{split}.json"
    with open(segpath) as f:
        seg_dict = json.load(f)

    cxr_ids = seg_dict.keys()
    # seg_masks = dict()
    for cxr_id in tqdm(cxr_ids, total=len(cxr_ids)):
        seg_item = seg_dict[cxr_id]
        seg_mask_dict = dict()
        for _, task in enumerate(localization_tasks):
            seg_mask = cocomask.decode(seg_item[task])
            seg_mask_dict[task] = seg_mask

        with open(os.path.join(save_path, split, f"{cxr_id}.pkl"), "wb") as f:
            pickle.dump(seg_mask_dict, f)


if __name__ == "__main__":
    main("test")
    main("val")
