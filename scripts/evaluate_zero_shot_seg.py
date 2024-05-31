import os
import ipdb
import random
import numpy as np
import pandas as pd
from typing import List
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from einops import rearrange
import segmentation_models_pytorch as smp
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import GaussianBlur
from cxrseg.datasets.cxr_datasets import (NIH_Localization_Dataset, CheXlocalize_dataset,
                                          SIIM_Pneumothorax_Dataset, MS_CXR_Dataset,
                                          RSNA_Pneumonia_Dataset, COVID_Rural_Dataset, box2mask)
from cxrseg.datasets.transforms import get_bbox_transforms, get_transforms
from cxrseg.utils.prompts import generate_class_prompts, custom_mapping
from evaluate_zero_shot_cls import get_model
from cxrseg.utils.evaluation_utils import get_class_texts, color_print, bootstrap_metric, create_ci_record

random.seed(42)
torch.random.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')


def normalize_heatmap(heatmap):
    ''' Normalize for each heatmap into [-1, 1]'''

    min_val = heatmap.min(dim=-1)[0].min(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    max_val = heatmap.max(dim=-1)[0].max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    heatmap = (heatmap - min_val) / (max_val - min_val)
    heatmap = heatmap * 2 - 1

    return heatmap


def create_zero_shot_seg_dataloader(dataset_dir: str,
                                    dataset_name: str,
                                    batch_size: int,
                                    num_workers: int = 4,
                                    imagesize: int = 384,
                                    mean: float = 0.471,
                                    std: float = 0.302):

    if dataset_name == "nih_loc":
        transform = get_bbox_transforms(
            is_train=False, IMAGE_INPUT_SIZE=imagesize, mean=mean, std=std)
        nih_dataset = NIH_Localization_Dataset(
            imgpath=os.path.join(dataset_dir, "NIH/images"),
            bbox_list_path=os.path.join(dataset_dir, "NIH/BBox_List_2017.csv"),
            mask_path=os.path.join(dataset_dir, "mask/nih"),
            transform=transform,
            pathology_masks=False
        )
        dataset = nih_dataset
        annotation_type = "box"
    elif dataset_name == "chexlocalize":
        transform = get_transforms(
            is_train=False, imagesize=imagesize, mean=mean, std=std)
        chexlocalize_dataset = CheXlocalize_dataset(imgpath=os.path.join(dataset_dir, "CheXpert"),
                                                    csvpath=os.path.join(
                                                        dataset_dir, "CheXpert/test_labels.csv"),
                                                    segpath=os.path.join(
                                                        dataset_dir, "mask/CheXlocalize/test"),
                                                    transform=transform)
        dataset = chexlocalize_dataset
        annotation_type = "mask"
    elif dataset_name == "ms-cxr":
        transform = get_bbox_transforms(
            is_train=False, IMAGE_INPUT_SIZE=imagesize, mean=mean, std=std)
        ms_cxr_dataset = MS_CXR_Dataset(
            imgpath=os.path.join(dataset_dir, "mimic_data/2.0.0"),
            csvpath=os.path.join(
                dataset_dir, "ms-cxr/0.1/MS_CXR_Local_Alignment_v1.0.0.csv"),
            mask_path=os.path.join(dataset_dir, "mask/mscxr"),
            transform=transform,
            pathology_masks=False
        )
        dataset = ms_cxr_dataset
        annotation_type = "box"
    elif dataset_name == "siim":
        transform = get_transforms(
            is_train=False, imagesize=imagesize, mean=mean, std=std)
        siim_dataset = SIIM_Pneumothorax_Dataset(
            imgpath=os.path.join(
                dataset_dir, "SIIM_Pneumothorax/dicom-images-train"),
            csvpath=os.path.join(
                dataset_dir, "preprocessed_csv/SIIM/SIIM_test.csv"),
            transform=transform,
            nonzero_mask=True
        )
        dataset = siim_dataset
        annotation_type = "mask"
    elif dataset_name == "rsna":
        transform = get_bbox_transforms(
            is_train=False, IMAGE_INPUT_SIZE=imagesize, mean=mean, std=std)
        # only use the test set xxx
        rsna_dataset = RSNA_Pneumonia_Dataset(
            imgpath=os.path.join(
                dataset_dir, "RSNA_Pneumonia/stage_2_train_images"),
            csvpath=os.path.join(
                dataset_dir, "preprocessed_csv/RSNA/RSNA_test.csv"),
            # mask_path=os.path.join(dataset_dir, "mask/rsna"),
            transform=transform,
            bbox_only=True,
            pathology_masks=False
        )
        dataset = rsna_dataset
        annotation_type = "box"
    elif dataset_name == "covid19":
        transform = get_transforms(
            is_train=False, imagesize=imagesize, mean=mean, std=std)
        covid_dataset = COVID_Rural_Dataset(
            imgpath=os.path.join(
                dataset_dir, "opacity_segmentation_covid_chest_X_ray/covid_rural_annot/jpgs"),
            annpath=os.path.join(
                dataset_dir, "opacity_segmentation_covid_chest_X_ray/covid_rural_annot/jpgs/jsons"),
            transform=transform,
        )
        dataset = covid_dataset
        annotation_type = "mask"

    def zero_shot_seg_collate_fn(batch):
        return batch

    # print(dataset)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            collate_fn=zero_shot_seg_collate_fn)

    return dataloader, dataset, annotation_type


@torch.no_grad()
def zero_shot_seg_evaluation(model: nn.Module, dataloader: DataLoader, class_names: List,
                             annotation_type: bool,
                             model_name: str,
                             dataset_name: str,
                             prompt_style: str,
                             use_negative: bool,
                             save_pred_results: bool,
                             save_dir: str):

    color_print(
        f"Compute embeddings for {len(class_names)} classes", color="green")
    gaussian_blur = GaussianBlur(kernel_size=13, sigma=1.5)
    # get zero-shot text embeddings
    if model_name == "medclip":
        raise NotImplementedError(
            "MedCLIP is not supported for zero-shot segmentation")
        # if use_negative:
        #     tokenized_pos_texts, tokenized_neg_texts = get_class_texts(model.tokenizer, class_names,
        #                                                                prompt_style=prompt_style,
        #                                                                num_templates=4,
        #                                                                use_negative=use_negative)
        #     pos_text_embeds = []
        #     for pathology in class_names:
        #         cur_text = tokenized_pos_texts[pathology]
        #         cur_text = {k: v.cuda() for k, v in cur_text.items()}
        #         text_embed = model.encode_text(input_ids=cur_text["input_ids"],
        #                                        attention_mask=cur_text["attention_mask"])
        #         pos_text_embeds.append(text_embed)
        #     pos_text_embeds = torch.stack(pos_text_embeds, dim=0)

        #     neg_text_embeds = []
        #     for pathology in class_names:
        #         cur_text = tokenized_neg_texts[pathology]
        #         cur_text = {k: v.cuda() for k, v in cur_text.items()}
        #         text_embed = model.encode_text(input_ids=cur_text["input_ids"],
        #                                        attention_mask=cur_text["attention_mask"])
        #         neg_text_embeds.append(text_embed)
        #     neg_text_embeds = torch.stack(neg_text_embeds, dim=0)
        # else:
        #     tokenized_texts = get_class_texts(model.tokenizer, class_names, prompt_style=prompt_style,
        #                                       num_templates=4, use_negative=use_negative)
        #     text_embeds = []
        #     for pathology in class_names:
        #         cur_text = tokenized_texts[pathology]
        #         cur_text = {k: v.cuda() for k, v in cur_text.items()}
        #         text_embed = model.encode_text(input_ids=cur_text["input_ids"],
        #                                        attention_mask=cur_text["attention_mask"])
        #         text_embeds.append(text_embed)
        #     text_embeds = torch.stack(text_embeds, dim=0)  # 10, 4, 128

    elif model_name == "our_medclip":
        if use_negative:
            tokenized_pos_texts, tokenized_neg_texts = get_class_texts(model.tokenizer, class_names,
                                                                       prompt_style=prompt_style,
                                                                       num_templates=4,
                                                                       use_negative=True)
            pos_text_embeds = []
            for pathology in class_names:
                cur_text = tokenized_pos_texts[pathology]
                cur_text = {k: v.cuda() for k, v in cur_text.items()}
                text_embed = model.encode_text_student(input_ids=cur_text["input_ids"],
                                                       attention_mask=cur_text["attention_mask"])[1]
                pos_text_embeds.append(text_embed)
            pos_text_embeds = torch.stack(pos_text_embeds, dim=0)

            neg_text_embeds = []
            for pathology in class_names:
                cur_text = tokenized_neg_texts[pathology]
                cur_text = {k: v.cuda() for k, v in cur_text.items()}
                text_embed = model.encode_text_student(input_ids=cur_text["input_ids"],
                                                       attention_mask=cur_text["attention_mask"])[1]
                neg_text_embeds.append(text_embed)
            neg_text_embeds = torch.stack(neg_text_embeds, dim=0)
        else:
            tokenized_texts = get_class_texts(model.tokenizer, class_names, prompt_style=prompt_style,
                                              num_templates=4, use_negative=False)
            text_embeds = []
            for pathology in class_names:
                cur_text = tokenized_texts[pathology]
                cur_text = {k: v.cuda() for k, v in cur_text.items()}
                text_embed = model.encode_text_student(input_ids=cur_text["input_ids"],
                                                       attention_mask=cur_text["attention_mask"])[1]
                text_embeds.append(text_embed)
            text_embeds = torch.stack(text_embeds, dim=0)  # 10, 4, 128

    elif model_name in ["mgca_cnn", "mgca_vit"]:
        pass

    elif model_name in ["gloria", "convirt", "random", "gloria_chexpert"]:
        if use_negative:
            tokenized_pos_texts, tokenized_neg_texts = get_class_texts(model.tokenizer, class_names,
                                                                       prompt_style=prompt_style,
                                                                       num_templates=4,
                                                                       use_negative=True)
            pos_text_embeds = []
            for pathology in class_names:
                cur_text = tokenized_pos_texts[pathology]
                cur_text = {k: v.cuda() for k, v in cur_text.items()}
                _, text_embed, _ = model.text_encoder(cur_text["input_ids"],
                                                      cur_text["attention_mask"])
                pos_text_embeds.append(text_embed)
            pos_text_embeds = torch.stack(pos_text_embeds, dim=0)

            neg_text_embeds = []
            for pathology in class_names:
                cur_text = tokenized_neg_texts[pathology]
                cur_text = {k: v.cuda() for k, v in cur_text.items()}
                _, text_embed, _ = model.text_encoder(cur_text["input_ids"],
                                                      cur_text["attention_mask"])
                neg_text_embeds.append(text_embed)
            neg_text_embeds = torch.stack(neg_text_embeds, dim=0)
        else:
            tokenized_texts = get_class_texts(model.tokenizer, class_names,
                                              prompt_style=prompt_style,
                                              num_templates=4, use_negative=False)
            text_embeds = []
            for pathology in class_names:
                cur_text = tokenized_texts[pathology]
                cur_text = {k: v.cuda() for k, v in cur_text.items()}
                _, text_embed, _ = model.text_encoder(cur_text["input_ids"],
                                                      cur_text["attention_mask"])
                text_embeds.append(text_embed)
            text_embeds = torch.stack(text_embeds, dim=0)  # 10, 4, 128

    elif model_name == "afloc":
        if use_negative:
            pos_class_prompts = generate_class_prompts(
                class_names, mode="pos", prompt_style=prompt_style)
            pos_text_embeds = []
            for pathology in class_names:
                prompts = pos_class_prompts[pathology]
                prompts = random.choices(prompts, k=4)
                pt = model.process_text(prompts, device)
                res = model.text_encoder_forward(
                    pt["caption_ids"].to(device),
                    pt["attention_mask"].to(device),
                    pt["token_type_ids"].to(device))
                pos_text_embeds.append(res["report_embeddings"])
            pos_text_embeds = torch.stack(pos_text_embeds, dim=0)  # 10, 4, 128

            neg_class_prompts = generate_class_prompts(
                class_names, mode="neg", prompt_style=prompt_style)
            neg_text_embeds = []
            for pathology in class_names:
                prompts = neg_class_prompts[pathology]
                prompts = random.choices(prompts, k=4)
                pt = model.process_text(prompts, device)
                res = model.text_encoder_forward(
                    pt["caption_ids"].to(device),
                    pt["attention_mask"].to(device),
                    pt["token_type_ids"].to(device))
                neg_text_embeds.append(res["report_embeddings"])
            neg_text_embeds = torch.stack(neg_text_embeds, dim=0)  # 10, 4, 128
        else:
            class_prompts = generate_class_prompts(
                class_names, mode="pos", prompt_style=prompt_style)
            text_embeds = []
            for pathology in class_names:
                prompts = class_prompts[pathology]
                prompts = random.choices(prompts, k=4)
                pt = model.process_text(prompts, device)
                res = model.text_encoder_forward(
                    pt["caption_ids"].to(device),
                    pt["attention_mask"].to(device),
                    pt["token_type_ids"].to(device)
                )
                text_embeds.append(res["report_embeddings"])
            text_embeds = torch.stack(text_embeds, dim=0)  # 10, 4, 128

    elif model_name in ["biovil", "biovil_t"]:
        if use_negative:
            pos_class_prompts = generate_class_prompts(
                class_names, mode="pos", prompt_style=prompt_style)
            pos_text_embeds = []
            for pathology in class_names:
                prompts = pos_class_prompts[pathology]
                prompts = random.choices(prompts, k=4)
                text_embed = model.text_inference_engine.get_embeddings_from_prompt(
                    prompts)  # (N, D)
                pos_text_embeds.append(text_embed)
            pos_text_embeds = torch.stack(pos_text_embeds, dim=0)  # 10, 4, 128

            neg_class_prompts = generate_class_prompts(
                class_names, mode="neg", prompt_style=prompt_style)
            neg_text_embeds = []
            for pathology in class_names:
                prompts = neg_class_prompts[pathology]
                prompts = random.choices(prompts, k=4)
                text_embed = model.text_inference_engine.get_embeddings_from_prompt(
                    prompts)  # (N, D)
                neg_text_embeds.append(text_embed)
            neg_text_embeds = torch.stack(neg_text_embeds, dim=0)  # 10, 4, 128
        else:
            class_prompts = generate_class_prompts(
                class_names, mode="pos", prompt_style=prompt_style)
            text_embeds = []
            for pathology in class_names:
                prompts = class_prompts[pathology]
                prompts = random.choices(prompts, k=4)
                text_embed = model.text_inference_engine.get_embeddings_from_prompt(
                    prompts)  # (N, D)
                text_embeds.append(text_embed)
            text_embeds = torch.stack(text_embeds, dim=0)  # 10, 4, 128

    elif model_name in ["medklip"]:
        # FIXME: make sure the text is original class names
        from cxrseg.third_party.medklip.load_pretrained_medklip import original_class
        indices = []
        for pathology in class_names:
            pathology = pathology.lower()
            if pathology == "enlarged cardiomediastinum":
                pathology = "enlarge"
            elif pathology == "lung lesion":
                pathology = "lesion"
            elif pathology == "lung opacity":
                pathology = "opacity"
            elif pathology == "pleural effusion":
                pathology = "effusion"
            elif pathology == "support devices":
                pathology = "device"
            elif pathology == "infiltration":
                pathology = "infiltrate"

            indices.append(original_class.index(pathology))

    elif model_name in ["kad_resnet_224", "kad_resnet_512", "kad_resnet_1024"]:
        kad_model, image_encoder, text_encoder = model
        text_inputs = kad_model.tokenizer(class_names,
                                          add_special_tokens=True,
                                          padding='max_length',
                                          max_length=77,
                                          truncation=True,
                                          return_tensors="pt").to(device)
        text_features = text_encoder.encode_text(text_inputs)
        text_embeds = text_features.unsqueeze(1)
    else:
        raise NotImplementedError

    all_pred = []
    gt_masks = []
    all_labels = []
    all_imgids = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Load dataset"):
        imgs = [x["img"].to(device) for x in batch]
        imgs = torch.stack(imgs, dim=0)

        if model_name == "medclip":
            # FIXME:
            pass

        elif model_name == "our_medclip":
            img_feats, patch_feats = model.encode_image_student(
                imgs, dense=True)
            patch_embeds = model.img_projection_student(patch_feats)
            patch_embeds = F.normalize(patch_embeds, dim=-1)
            feature_resolution = [16, 16]
            img_feat = rearrange(patch_embeds, "b (h w) c->b c h w",
                                 h=feature_resolution[0], w=feature_resolution[1])
            if use_negative:
                pos_corr = torch.einsum(
                    'bchw, npc -> bnphw', img_feat, pos_text_embeds)
                pos_pred_map = pos_corr.mean(dim=2)
                neg_corr = torch.einsum(
                    'bchw, npc -> bnphw', img_feat, neg_text_embeds)
                neg_pred_map = neg_corr.mean(dim=2)
                concat_pred_map = torch.stack(
                    [pos_pred_map, neg_pred_map], dim=0)
                concat_pred_map = F.softmax(concat_pred_map / 0.5, dim=0)
                pred_map = concat_pred_map[0]
            else:
                corr = torch.einsum('bchw, npc -> bnphw',
                                    img_feat, text_embeds)
                pred_map = corr.mean(dim=2)

        elif model_name in ["biovil", "biovil_t"]:
            # create pred_map from image and text embeddings
            image_embedding, (width, height) = model.image_inference_engine.get_projected_patch_embeddings(
                imgs)
            if use_negative:
                pos_corr = torch.einsum(
                    'bhwc, npc -> bnphw', image_embedding, pos_text_embeds)
                pos_pred_map = pos_corr.mean(dim=2)
                neg_corr = torch.einsum(
                    'bhwc, npc -> bnphw', image_embedding, neg_text_embeds)
                neg_pred_map = neg_corr.mean(dim=2)
                concat_pred_map = torch.stack(
                    [pos_pred_map, neg_pred_map], dim=0)
                concat_pred_map = F.softmax(concat_pred_map / 0.5, dim=0)
                pred_map = concat_pred_map[0]
            else:
                corr = torch.einsum('bhwc, npc -> bnphw',
                                    image_embedding, text_embeds)
                pred_map = corr.mean(dim=2)

        elif model_name in ["gloria", "convirt", "random", "gloria_chexpert"]:
            img_emb_l, img_emb_g = model.image_encoder_forward(imgs)
            if use_negative:
                pos_corr = torch.einsum(
                    'bchw, npc -> bnphw', img_emb_l, pos_text_embeds)
                pos_corr = pos_corr.mean(dim=2).detach()
                neg_corr = torch.einsum(
                    'bchw, npc -> bnphw', img_emb_l, neg_text_embeds)
                neg_corr = neg_corr.mean(dim=2).detach()
                concat_pred_map = torch.stack(
                    [pos_corr, neg_corr], dim=0)
                concat_pred_map = F.softmax(concat_pred_map / 0.5, dim=0)
                pred_map = concat_pred_map[0]
            else:
                corr = torch.einsum('bchw, npc -> bnphw',
                                    img_emb_l, text_embeds)
                pred_map = corr.mean(dim=2).detach()

        elif model_name == "medklip":
            _, ws = model(imgs, None, is_train=False)
            ws = (ws[-4] + ws[-3] + ws[-2] + ws[-1])/4
            ws = ws.reshape(-1, ws.shape[1], 14, 14)
            # fetch the corresponding class
            pred_map = ws[:, indices, :, :].detach()

        elif model_name in ["kad_resnet_224", "kad_resnet_512", "kad_resnet_1024"]:
            kad_model, image_encoder, text_encoder = model
            image_features, image_features_pool = image_encoder(imgs)
            all_atten_map = []
            for i in range(len(text_embeds)):
                _, atten_map = kad_model(
                    image_features, text_embeds[i], return_atten=True)
                all_atten_map.append(atten_map)
            atten_map = torch.stack(all_atten_map, dim=1)
            atten_map = atten_map.mean(dim=2)
            # atten_map: B, C, H*W
            h = w = int(np.sqrt(atten_map.shape[-1]))
            pred_map = rearrange(atten_map, "b c (h w) -> b c h w", h=h, w=w)

        elif model_name == "afloc":
            iel, _, _, ieg = model.image_encoder_forward(imgs.to(device))
            if use_negative:
                pos_corr = torch.einsum(
                    'bchw, npc -> bnphw', iel, pos_text_embeds)
                pos_pred_map = pos_corr.mean(dim=2)
                neg_corr = torch.einsum(
                    'bchw, npc -> bnphw', iel, neg_text_embeds)
                neg_pred_map = neg_corr.mean(dim=2)
                concat_pred_map = torch.stack(
                    [pos_pred_map, neg_pred_map], dim=0)
                concat_pred_map = F.softmax(concat_pred_map / 0.5, dim=0)
                pred_map = concat_pred_map[0]
            else:
                corr = torch.einsum('bchw, npc -> bnphw', iel, text_embeds)
                pred_map = corr.mean(dim=2)
        else:
            raise ValueError(
                f"Model {model_name} not supported for zero-shot-segmentation")

        pred_map = gaussian_blur(pred_map)
        pred_map = F.interpolate(pred_map, size=(
            imgs.shape[-2], imgs.shape[-1]), mode='bilinear', align_corners=False)
        pred_map = normalize_heatmap(pred_map)  # [-1, 1]

        # pred_map for each method: [B, N, H, W]
        h, w = imgs.shape[-2], imgs.shape[-1]
        for i, sample in enumerate(batch):
            all_imgids.append(sample["imgid"])
            if annotation_type == "mask":
                # if the annotation is mask, we need to get the mask and label
                sample["lab"] = torch.tensor(sample["lab"]).to(device)
                cur_pred_mask = pred_map[i][sample["lab"].bool()]
                all_pred.append(cur_pred_mask.cpu())
                gt_masks.append(sample["mask"].cpu())
                mask_labels = torch.where(sample["lab"].bool())[0]
                all_labels.append(mask_labels.cpu())
            elif annotation_type == "box":
                # if the annotation is box, we need to get the box and label
                boxes = sample["bbox"]
                box_labels = sample["bbox_label"].squeeze(1)
                for box_label in box_labels.unique().sort()[0]:
                    # In some case, there are multiple boxes for the same label
                    cur_indices = torch.where(box_label == box_labels)[0]
                    cur_pred_mask = pred_map[i][box_label]
                    all_pred.append(cur_pred_mask.unsqueeze(0).cpu())
                    gt_mask = box2mask(boxes[cur_indices], h, w)
                    gt_masks.append(gt_mask.cpu())
                # Save the labels for each mask
                all_labels.append(box_labels.unique().sort()[0].cpu())
            else:
                raise ValueError(
                    f"Model {model_name} not supported for zero-shot-segmentation")

    all_pred = torch.cat(all_pred, axis=0).unsqueeze(1)  # [N, 1, H, W]
    gt_masks = torch.cat(gt_masks, axis=0).unsqueeze(1)  # [N, 1, H, W]
    all_labels = torch.cat(all_labels, axis=0)  # [N]
    # compute segmentation masks

    save_dir = os.path.join(
        save_dir, f"{model_name}_{dataset_name}_{prompt_style}")
    os.makedirs(save_dir, exist_ok=True)
    if save_pred_results:
        save_data_dict = {
            "all_pred": all_pred,
            "gt_masks": gt_masks,
            "all_labels": all_labels,
            "class_names": class_names,
            "all_imgids": all_imgids
        }
        torch.save(save_data_dict, os.path.join(save_dir, "saved_data.pth"))
    compute_seg_metrics(all_pred, gt_masks, all_labels, class_names, save_dir)


def compute_seg_metrics(all_pred, gt_masks, all_labels, class_names, save_dir):
    num_classes = len(class_names)

    imgids = []
    all_ious, all_dices, all_point_scores = dict(), dict(), dict()
    all_num_pairs = dict()
    color_print(f"All class names: {class_names}", color="yellow")
    for i in tqdm(range(num_classes), desc="Compute metrics per class"):
        cur_class_name = class_names[i]
        cur_indices = torch.where(all_labels == i)[0]
        imgids.append(cur_indices)
        all_num_pairs[cur_class_name] = len(cur_indices)

        cur_pred = all_pred[cur_indices]
        cur_gt_masks = gt_masks[cur_indices]

        cur_iou_scores = np.empty(len(all_labels))
        cur_iou_scores.fill(np.nan)
        cur_dice_scores = np.empty(len(all_labels))
        cur_dice_scores.fill(np.nan)
        cur_point_scores = np.empty(len(all_labels))
        cur_point_scores.fill(np.nan)

        if (len(cur_pred) > 0) & (len(cur_gt_masks) > 0):
            total_num = len(cur_indices)
            flatten_pred = cur_pred.view(total_num, -1)
            flatten_gt = cur_gt_masks.view(total_num, -1)
            max_vals = torch.max(flatten_pred, dim=-1)[0]
            sample_wise_point_score = []
            for i in range(total_num):
                mask = flatten_pred[i] == max_vals[i]
                temp = flatten_gt[i][mask].sum() > 0
                sample_wise_point_score.append(temp.float())
            sample_wise_point_score = torch.stack(sample_wise_point_score)
            cur_point_scores[cur_indices.cpu().numpy(
            )] = sample_wise_point_score.cpu().numpy()
            all_point_scores[cur_class_name] = cur_point_scores

            # average over 5 thresholds
            iou_scores = []
            dice_scores = []
            for thres in [0.1, 0.2, 0.3, 0.4, 0.5]:
                # The sample-wise metrics follows previous work: CheXzero, MedKLIP, and AFLoc
                tp, fp, fn, tn = smp.metrics.get_stats(cur_pred,
                                                       cur_gt_masks.long(),
                                                       mode='binary',
                                                       threshold=thres)
                sample_iou_score = smp.metrics.iou_score(
                    tp, fp, fn, tn, reduction="none")
                sample_dice_score = (2 * tp) / (2 * tp + fp + fn)

                iou_scores.append(sample_iou_score)
                dice_scores.append(sample_dice_score)

            iou_scores = torch.cat(iou_scores, dim=1).mean(dim=1)
            dice_scores = torch.cat(dice_scores, dim=1).mean(dim=1)
            cur_iou_scores[cur_indices.cpu().numpy()
                           ] = iou_scores.cpu().numpy()
            cur_dice_scores[cur_indices.cpu().numpy()
                            ] = dice_scores.cpu().numpy()
            all_ious[cur_class_name] = cur_iou_scores
            all_dices[cur_class_name] = cur_dice_scores
        else:
            raise ValueError("No samples for class {}".format(class_names[i]))

    iou_df = pd.DataFrame.from_dict(all_ious)
    dice_df = pd.DataFrame.from_dict(all_dices)
    point_df = pd.DataFrame.from_dict(all_point_scores)
    pair_df = pd.DataFrame.from_dict(
        all_num_pairs, orient='index', columns=['num_samples'])
    pair_df.insert(0, "name", pair_df.index)
    pair_df.reset_index(drop=True, inplace=True)

    iou_df.to_csv(os.path.join(save_dir, f"iou.csv"), index=False)
    mean_iou = np.nanmean(iou_df.values, axis=0)
    mean_dice = np.nanmean(dice_df.values, axis=0)
    mean_point = np.nanmean(point_df.values, axis=0)
    ori_metrics = np.vstack([mean_iou, mean_dice, mean_point])
    ori_metrics_df = pd.DataFrame(ori_metrics, columns=class_names, index=[
                                  "IoU", "Dice", "Pointing Game"]).T
    mean_metrics = ori_metrics_df.mean(axis=0)
    ori_metrics_df.loc["mean"] = mean_metrics
    ori_metrics_df.to_csv(os.path.join(
        save_dir, f"ori_metrics.csv"), index=True)

    # This part of code is borrowed from CheXZero
    boot_iou_df = bootstrap_metric(iou_df, 1000, class_names)
    boot_dice_df = bootstrap_metric(dice_df, 1000, class_names)
    boot_point_df = bootstrap_metric(point_df, 1000, class_names)

    boot_iou_df.to_csv(os.path.join(save_dir, f"boot_iou.csv"), index=False)
    boot_dice_df.to_csv(os.path.join(save_dir, f"boot_dice.csv"), index=False)
    boot_point_df.to_csv(os.path.join(
        save_dir, f"boot_point.csv"), index=False)

    # get 95% confidence intervals
    iou_records = []
    for task in boot_iou_df.columns:
        iou_records.append(create_ci_record(boot_iou_df[task], task, "IoU"))
    summary_iou_df = pd.DataFrame.from_records(
        iou_records).sort_values(by='name')
    # print(summary_iou_df)

    dice_records = []
    for task in boot_dice_df.columns:
        dice_records.append(create_ci_record(boot_dice_df[task], task, "Dice"))
    summary_dice_df = pd.DataFrame.from_records(
        dice_records).sort_values(by='name')

    point_records = []
    for task in boot_point_df.columns:
        point_records.append(create_ci_record(
            boot_point_df[task], task, "Pointing Game"))
    summary_point_df = pd.DataFrame.from_records(
        point_records).sort_values(by='name')

    summary_df = reduce(lambda left, right: pd.merge(
        left, right, on=['name'], how='inner'), [pair_df, summary_iou_df, summary_dice_df, summary_point_df])
    mean_metrics = summary_df.iloc[:, 2:].mean(axis=0)
    summary_df.loc[len(summary_df)] = [
        "mean", len(iou_df)] + mean_metrics.tolist()

    os.makedirs(save_dir, exist_ok=True)
    color_print(summary_df, color="cyan")
    summary_df.to_csv(os.path.join(save_dir, f"summary.csv"), index=False)


def main(hparams: Namespace):
    model = get_model(hparams)

    model_name = hparams.model_name
    if model_name == "biovil":
        mean, std = 0, 1
        imagesize = 480
        from transformers import AutoTokenizer
        bert_type = "microsoft/BiomedVLP-CXR-BERT-specialized"
        tokenizer = AutoTokenizer.from_pretrained(
            bert_type, trust_remote_code=True)
        model.tokenizer = tokenizer
    elif model_name == "biovil_t":
        mean, std = 0, 1
        imagesize = 448
        from transformers import AutoTokenizer
        bert_type = "microsoft/BiomedVLP-BioViL-T"
        tokenizer = AutoTokenizer.from_pretrained(
            bert_type, trust_remote_code=True)
        model.tokenizer = tokenizer
    elif model_name in ["medclip_vit", "medclip_cnn"]:
        mean, std = 0.5862785803043838, 0.27950088968644304
        from cxrseg.third_party.medclip.constants import BERT_TYPE
        from transformers import AutoTokenizer
        bert_type = BERT_TYPE
        imagesize = 224
        tokenizer = AutoTokenizer.from_pretrained(bert_type)
        model.tokenizer = tokenizer
    elif model_name in ["our_medclip", "our_medclip_s2"]:
        mean, std = 0, 1
        imagesize = 512
    elif model_name in ["gloria", "convirt", "random"]:
        mean, std = 0, 1
        # bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        imagesize = 224
    elif model_name == "gloria_chexpert":
        mean, std = 0.5, 0.5
        bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        imagesize = 224
    elif model_name == "medklip":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        imagesize = 224
    elif model_name == "kad_resnet_224":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        imagesize = 224
    elif model_name == "kad_resnet_512":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        imagesize = 512
    elif model_name == "kad_resnet_1024":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        imagesize = 1024
    elif model_name in ["mgca_cnn", "mgca_vit"]:
        mean, std = 0.5, 0.5
        imagesize = 224
    elif model_name == "chexzero":
        mean, std = (0.398, 0.398, 0.398), (0.327, 0.327, 0.327)
        imagesize = 224
    elif model_name == "afloc":
        mean, std = 0.5, 0.5
        imagesize = 224

    for dataset_name in hparams.dataset_list:
        color_print(f"Evaluating on {dataset_name}:", color="red")
        dataloader, dataset, annotation_type = create_zero_shot_seg_dataloader(
            dataset_dir=hparams.dataset_dir,
            dataset_name=dataset_name,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
            imagesize=imagesize,
            mean=mean,
            std=std
        )
        class_names = dataset.pathologies
        class_names = [custom_mapping[x]
                       if x in custom_mapping else x for x in class_names]
        zero_shot_seg_evaluation(model, dataloader, class_names, annotation_type,
                                 hparams.model_name,
                                 dataset_name,
                                 hparams.prompt_style,
                                 hparams.use_negative_prompt,
                                 hparams.save_pred_results,
                                 hparams.save_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, default="zero-shot-seg",
                        choices=["zero-shot-cls", "linear", "zero-shot-seg", "phrase_grounding"])
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset_dir", type=str,
                        default="/disk1/fywang/CXR_dataset")
    # default="/data1/r20user2/CXR_dataset")
    parser.add_argument("--use_negative_prompt", action="store_true")
    # use another conda environment if evaluating medclip
    parser.add_argument("--model_name", type=str, default="gloria_chexpert",
                        choices=["medclip_vit", "medclip_cnn", "convirt", "gloria_chexpert",
                                 "gloria", "biovil", "biovil_t", "our_medclip",
                                 "medklip", "kad_resnet_224", "kad_resnet_512", "kad_resnet_1024",
                                 "mgca_cnn", "mgca_vit", "chexzero", "afloc", "our_medclip_s2",
                                 "random"])
    parser.add_argument("--mask_label_type", type=str, default="box",
                        choices=["box", "mask"])
    parser.add_argument("--ckpt_path", type=str,
                        # gloria
                        default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_03_18_23_27_36/epoch=12-step=13715.ckpt")
    # convirt
    # default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_03_21_22_25_20/epoch=12-step=27417.ckpt")
    parser.add_argument("--save_dir", type=str,
                        default="/home/fywang/Documents/CXRSeg/evaluation_results/zero_shot_seg")
    parser.add_argument("--dataset_list", type=str, nargs="+",
                        default=["chexlocalize", "nih_loc", "ms-cxr", "siim", "rsna", "covid19"])
    parser.add_argument("--prompt_style", type=str, default="xplainer",
                        choices=["xplainer", "biovil", "chexzero", "gloria"])
    parser.add_argument("--save_pred_results", action="store_true")
    args = parser.parse_args()
    main(args)
