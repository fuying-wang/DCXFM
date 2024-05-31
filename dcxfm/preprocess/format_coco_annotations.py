import os
import json
import multiprocessing
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from functools import partial
from sahi.utils.coco import Coco, CocoCategory, CocoImage, CocoAnnotation
from sahi.utils.file import save_json
from imantics import Polygons, Mask
from dcxfm.datasets.transforms import get_transforms, get_bbox_transforms
from dcxfm.datasets.cxr_datasets import CheXlocalize_dataset, NIH_Localization_Dataset, VinBrain_Dataset, \
    RSNA_Pneumonia_Dataset, SIIM_Pneumothorax_Dataset, MergeDataset, CANDID_PTX_Dataset, MS_CXR_Dataset, \
    COVID_Rural_Dataset
from dcxfm.datasets.medclip_datasets import MedCLIP_Pretrain_Dataset
import ipdb

'''
Next step:
consider add medclip dataset without annotations
'''
dataset_dir = "/disk1/fywang/CXR_dataset"
# dataset_dir = "/home/r15user2/Documents/CXR_dataset"
save_dir = os.path.join(dataset_dir, "mask/MaCheX")
mean, std = 0, 1
os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)


def worker_fn(i, train_dataset, categories, coco, label_dict):
    sample = train_dataset[i]
    # if not os.path.exists(os.path.join(save_dir, "images", f"{dataset_name}_{imgid}.png")):
    img = sample["img"]
    ori_img = (img * std + mean) * 255
    ori_img = ori_img.cpu().numpy().astype(np.uint8)
    ori_img = ori_img.transpose(1, 2, 0)
    pil_img = Image.fromarray(ori_img)
    imgid = sample["imgid"].replace("/", "_")
    imgid = os.path.splitext(imgid)[0]
    dataset_name = train_dataset.datasets[sample["source"]].__class__.__name__
    pil_img.save(os.path.join(save_dir, "images",
                 f"{dataset_name}_{imgid}.png"))

    width, height = pil_img.size
    img_name = f"{dataset_name}_{imgid}.png"
    coco_image = CocoImage(id=i+1, file_name=img_name,
                           width=width, height=height)
    label_dict[img_name] = sample["lab"]

    if "mask" in sample:
        mask = sample["mask"].cpu().numpy()
        mask_indices = np.where(sample["lab"] == 1)[0]  # positive mask indices
        assert len(mask_indices) == len(
            sample["mask"]), f"{imgid}: Number of masks and labels are not the same!"

        # HACK: annotate each uncertain label with mask -1
        pos_mask_indices = []
        for (cur_mask, mask_i) in zip(mask, mask_indices):
            poly = Mask(cur_mask).polygons().segmentation
            poly = [x for x in poly if len(x) >= 8]
            if len(poly) == 0:
                continue
            coco_annotation = CocoAnnotation(image_id=i+1, category_id=mask_i+1,
                                             category_name=categories[mask_i],
                                             segmentation=poly)
            coco_image.add_annotation(coco_annotation)
            pos_mask_indices.append(mask_i)

        # we only consider images with at least one annotation
        if len(coco_image.annotations) > 0:
            coco.add_image(coco_image)
    else:
        coco.add_image(coco_image)


def create_train_annotations():
    transform = get_transforms(is_train=False, mean=0, std=1, imagesize=512)

    vinbigdata_dataset = VinBrain_Dataset(
        imgpath=os.path.join(dataset_dir, "vinbigdata/train"),
        csvpath=os.path.join(dataset_dir, "vinbigdata/train.csv"),
        mask_path=os.path.join(dataset_dir, "mask/vinbigdata"),
        transform=transform,
        bbox_only=False,
        pathology_masks=True
    )

    candid_dataset = CANDID_PTX_Dataset(
        imgpath=os.path.join(dataset_dir, "CANDID-PTX/images"),
        csvpath=os.path.join(
            dataset_dir, "CANDID-PTX/Pneumothorax_reports.csv"),
        transform=transform,
        nonzero_mask=True,
    )

    train_dataset = MergeDataset([vinbigdata_dataset, candid_dataset])
    print(train_dataset)

    categories = train_dataset.pathologies
    with open(os.path.join(save_dir, "train_categories.json"), "w") as f:
        json.dump(categories, f)

    coco = Coco()
    for i, pathology in enumerate(categories):
        coco.add_category(CocoCategory(id=i+1, name=pathology))

    max_ = len(train_dataset)

    label_dict = dict()
    for i in tqdm(range(max_)):
        worker_fn(i, train_dataset=train_dataset,
                  categories=categories, coco=coco, label_dict=label_dict)

    label_df = pd.DataFrame.from_dict(
        label_dict, orient="index", columns=categories)
    label_df.to_csv(os.path.join(save_dir, "train_labels.csv"))
    save_json(data=coco.json, save_path=os.path.join(
        save_dir, "train_annotations.json"))


def create_val_annotations():
    # for dataset_name in ["chexlocalize"]:
    # for dataset_name in ["mscxr", "nih"]:
    for dataset_name in ["mscxr"]:
        # for dataset_name in ["rsna", "covid19"]:
        # for dataset_name in ["chexlocalize", "nih", "mscxr"]:
        save_val_dataset_files(dataset_name, split="val")


def save_val_dataset_files(dataset_name, split="val"):
    '''
    '''
    if dataset_name == "chexlocalize":
        transform = get_transforms(
            is_train=False, mean=0, std=1, imagesize=512)
        dataset = CheXlocalize_dataset(imgpath=os.path.join(dataset_dir, "CheXpert"),
                                       csvpath=os.path.join(
                                           dataset_dir, "CheXpert/test_labels.csv"),
                                       segpath=os.path.join(
                                           dataset_dir, "mask/CheXlocalize/test"),
                                       transform=transform)
    elif dataset_name == "nih":
        transform = get_transforms(
            is_train=False, mean=0, std=1, imagesize=512)
        dataset = NIH_Localization_Dataset(
            imgpath=os.path.join(dataset_dir, "NIH/images"),
            bbox_list_path=os.path.join(dataset_dir, "NIH/BBox_List_2017.csv"),
            transform=transform,
            pathology_masks=True
        )
    elif dataset_name == "mscxr":
        transform = get_transforms(
            is_train=False, mean=0, std=1, imagesize=512)
        dataset = MS_CXR_Dataset(
            imgpath=os.path.join(dataset_dir, "mimic_data/2.0.0"),
            csvpath=os.path.join(
                dataset_dir, "ms-cxr/0.1/MS_CXR_Local_Alignment_v1.0.0.csv"),
            mask_path=os.path.join(dataset_dir, "mask/mscxr"),
            transform=transform,
            pathology_masks=True
        )
    elif dataset_name == "rsna":
        transform = get_transforms(
            is_train=False, mean=0, std=1, imagesize=512)
        # only use the test set xxx
        dataset = RSNA_Pneumonia_Dataset(
            imgpath=os.path.join(
                dataset_dir, "RSNA_Pneumonia/stage_2_train_images"),
            csvpath=os.path.join(
                dataset_dir, "preprocessed_csv/RSNA/RSNA_test.csv"),
            transform=transform,
            pathology_masks=True
        )
    elif dataset_name == "covid19":
        transform = get_transforms(
            is_train=False, imagesize=512, mean=mean, std=std)
        dataset = COVID_Rural_Dataset(
            imgpath=os.path.join(
                dataset_dir, "opacity_segmentation_covid_chest_X_ray/covid_rural_annot/jpgs"),
            annpath=os.path.join(
                dataset_dir, "opacity_segmentation_covid_chest_X_ray/covid_rural_annot/jpgs/jsons"),
            transform=transform,
        )

    # save chexlocalize dataset
    val_dataset = MergeDataset([dataset])
    categories = val_dataset.pathologies
    with open(os.path.join(save_dir, f"{split}_{dataset_name}_categories.json"), "w") as f:
        json.dump(categories, f)

    coco = Coco()
    for i, pathology in enumerate(categories):
        coco.add_category(CocoCategory(id=i+1, name=pathology))

    max_ = len(val_dataset)
    label_dict = dict()
    for i in tqdm(range(max_)):
        worker_fn(i, train_dataset=val_dataset, categories=categories,
                  coco=coco, label_dict=label_dict)

    label_df = pd.DataFrame.from_dict(
        label_dict, orient="index", columns=categories)
    label_df.to_csv(os.path.join(
        save_dir, f"{split}_{dataset_name}_labels.csv"))
    save_json(data=coco.json, save_path=os.path.join(
        save_dir, f"{split}_{dataset_name}_annotations.json"))


if __name__ == "__main__":
    # create_train_annotations()
    create_val_annotations()
