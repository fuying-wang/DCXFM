import os
import ipdb
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torch.utils.data import DataLoader
from cxrseg.datasets.cxr_datasets import MS_CXR_Dataset
from cxrseg.datasets.transforms import get_bbox_transforms
from evaluate_zero_shot_cls import get_model
from evaluate_zero_shot_seg import compute_seg_metrics
''''
CUDA_VISIBLE_DEVICES=0 python evaluate_phrase_grounding.py --model_name gloria_chexpert
'''
random.seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')


def normalize_heatmap(heatmap):
    ''' Normalize for each heatmap into [-1, 1]
        Only works for 2D heatmap, batch size = 1, channel = 1
    '''

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    heatmap = heatmap * 2 - 1

    return heatmap


def create_phrase_grounding_dataloader(dataset_dir: str,
                                       dataset_name: str,
                                       batch_size: int,
                                       num_workers: int = 4,
                                       imagesize: int = 512,
                                       mean: float = 0.471,
                                       std: float = 0.302):

    def mscxr_collate_fn(batch):
        return batch

    if dataset_name == "ms_cxr":
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
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers,
                            collate_fn=mscxr_collate_fn)

    return dataloader, dataset


@torch.no_grad()
def phrase_grounding_evaluation(model: nn.Module, dataloader: DataLoader, class_names: str,
                                model_name: str = None, label_type: str = "box",
                                save_dir: str = "results/phrase_grounding"):

    # create indices for medklip
    print(class_names)
    if model_name == "medklip":
        from cxrseg.third_party.medklip.load_pretrained_medklip import original_class
        indices = []
        for c_name in class_names:
            c_name = c_name.lower()
            if c_name == "lung opacity":
                c_name = "opacity"
            elif c_name == "pleural effusion":
                c_name = "effusion"
            indices.append(original_class.index(c_name.lower()))

    all_pred = []
    gt_masks = []
    all_labels = []
    all_imgid = []
    all_imgs = []
    for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
        cur_imgid = [x["imgid"] for x in batch]
        all_imgid.extend(cur_imgid)
        imgs = [x["img"].to(device) for x in batch]
        imgs = torch.stack(imgs, dim=0)  # B, 3, 384, 384
        all_imgs.append(imgs)
        text = batch[0]["label_text"]

        if model_name in ["biovil", "biovil_t"]:
            for single_text in text:
                similarity_map = model.get_similarity_map_from_raw_data(
                    image=imgs,
                    query_text=single_text,
                    interpolation="bilinear",
                )
                pred_map = torch.tensor(
                    similarity_map).unsqueeze(0).to(device)
                pred_map = normalize_heatmap(pred_map)
                all_pred.append(pred_map)

        elif model_name == "our_medclip":
            text_inputs = model.tokenizer(
                text, return_tensors="pt", padding=True, max_length=77)
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            # Evaluation using student network
            img_feats, patch_feats = model.encode_image_student(
                imgs, dense=True)
            patch_embeds = model.img_projection_student(patch_feats)

            patch_embeds = F.normalize(patch_embeds, dim=-1)
            # _, patch_embeds = model.ibot_head_student(img_feats, patch_feats)
            feature_resolution = [16, 16]
            img_feat = rearrange(patch_embeds, "b (h w) c->b c h w",
                                 h=feature_resolution[0], w=feature_resolution[1])
            text_features = model.encode_text_student(
                input_ids=input_ids, attention_mask=attention_mask)[1]
            corr = torch.einsum('bchw, bpc -> bphw', img_feat,
                                text_features.unsqueeze(0))
            pred_map = corr[:, 0]

            # Gaussian filter
            from scipy import ndimage
            pred_map = torch.tensor(
                ndimage.gaussian_filter(pred_map.cpu().numpy()[
                                        0], sigma=(1.5, 1.5), order=0)
            ).unsqueeze(0).type_as(imgs)

            # Resize into original size
            pred_map = F.interpolate(pred_map.unsqueeze(0), size=(
                imgs.shape[-2], imgs.shape[-1]), mode='bilinear', align_corners=False)
            pred_map = pred_map[:, 0]

            pred_map = normalize_heatmap(pred_map)
            all_pred.append(pred_map.detach())

        elif model_name in ["medclip_vit", "medclip_cnn"]:
            from cxrseg.third_party.medclip.constants import BERT_TYPE
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(BERT_TYPE)
            text_inputs = tokenizer(text,
                                    add_special_tokens=True,
                                    padding='max_length',
                                    max_length=77,
                                    truncation=True,
                                    return_tensors="pt")
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            _, patch_embeds = model.encode_image(imgs, dense=True)
            feature_resolution = [7, 7]
            img_feat = rearrange(patch_embeds, "b (h w) c->b c h w",
                                 h=feature_resolution[0], w=feature_resolution[1])
            text_features = model.encode_text(
                input_ids=input_ids, attention_mask=attention_mask)
            corr = torch.einsum('bchw, bpc -> bphw', img_feat,
                                text_features.unsqueeze(0))
            pred_map = corr[:, 0]

            # Gaussian filter
            from scipy import ndimage
            pred_map = torch.tensor(
                ndimage.gaussian_filter(pred_map.cpu().numpy()[
                                        0], sigma=(1.5, 1.5), order=0)
            ).unsqueeze(0).type_as(imgs)

            # Resize into original size
            pred_map = F.interpolate(pred_map.unsqueeze(0), size=(
                imgs.shape[-2], imgs.shape[-1]), mode='bilinear', align_corners=False)
            pred_map = pred_map[:, 0]

            pred_map = normalize_heatmap(pred_map)
            all_pred.append(pred_map.detach())

        elif model_name in ["gloria", "convirt", "random", "gloria_chexpert"]:
            # correct
            text_inputs = model.tokenizer(text,
                                          add_special_tokens=True,
                                          padding='max_length',
                                          max_length=77,
                                          truncation=True,
                                          return_tensors="pt")
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            # Evaluation using student network
            img_emb_l, img_emb_g = model.image_encoder_forward(imgs)
            text_emb_l, text_emb_g, sents = model.text_encoder(
                input_ids, attention_mask
            )
            feature_resolution = [19, 19]
            img_feat = img_emb_l.clone()
            corr = torch.einsum('bchw, bpc -> bphw',
                                img_feat, text_emb_g.unsqueeze(0))
            pred_map = corr[:, 0]

            from scipy import ndimage
            pred_map = torch.tensor(
                ndimage.gaussian_filter(pred_map.cpu().numpy()[
                                        0], sigma=(1.5, 1.5), order=0)
            ).unsqueeze(0).type_as(imgs)

            # Resize into original size
            pred_map = F.interpolate(pred_map.unsqueeze(0), size=(
                imgs.shape[-2], imgs.shape[-1]), mode='bilinear', align_corners=False)
            pred_map = pred_map[:, 0]

            pred_map = normalize_heatmap(pred_map)
            all_pred.append(pred_map.detach())

        elif model_name == "medklip":
            _, ws = model(imgs, None, is_train=False)
            ws = (ws[-4] + ws[-3] + ws[-2] + ws[-1])/4
            ws = ws.reshape(1, ws.shape[1], 14, 14)
            # fetch the corresponding class
            pred_map = ws[:, indices, :, :].detach().cpu().numpy()
            labels = batch[0]["bbox_label"].reshape(-1)
            if len(labels) > 1:
                # if there are multiple labels, we take the max value
                pred_map = np.mean(pred_map[:, labels], axis=1)
            else:
                pred_map = pred_map[:, labels]

            # TODO: check do we need this gaussian filter
            # Gaussian filter
            from scipy import ndimage
            pred_map = torch.tensor(ndimage.gaussian_filter(
                pred_map[0], sigma=(1.5, 1.5), order=0)).unsqueeze(0).type_as(imgs)

            pred_map = F.interpolate(pred_map.unsqueeze(0), size=(
                imgs.shape[-2], imgs.shape[-1]), mode='bilinear', align_corners=False)
            pred_map = pred_map[:, 0]

            pred_map = normalize_heatmap(pred_map)
            all_pred.append(pred_map.detach())

        elif "kad" in model_name:
            kad_model, img_encoder, text_encoder = model
            text_inputs = kad_model.tokenizer(text,
                                              add_special_tokens=True,
                                              padding='max_length',
                                              max_length=77,
                                              truncation=True,
                                              return_tensors="pt")
            text_inputs["input_ids"] = text_inputs["input_ids"].to(device)
            text_inputs["attention_mask"] = text_inputs["attention_mask"].to(
                device)
            image_features, image_features_pool = img_encoder(imgs)
            text_features = text_encoder.encode_text(text_inputs)
            _, atten_map = kad_model(
                image_features, text_features, return_atten=True)
            # atten_map: B, C, H*W
            h = w = int(np.sqrt(atten_map.shape[-1]))
            pred_map = rearrange(atten_map, "b c (h w) -> b c h w", h=h, w=w)
            pred_map = pred_map[:, 0]
            # Gaussian filter
            from scipy import ndimage
            pred_map = torch.tensor(
                ndimage.gaussian_filter(pred_map.cpu().numpy()[
                                        0], sigma=(1.5, 1.5), order=0)
            ).unsqueeze(0).type_as(imgs)

            # Resize into original size
            pred_map = F.interpolate(pred_map.unsqueeze(0), size=(
                imgs.shape[-2], imgs.shape[-1]), mode='bilinear', align_corners=False)
            pred_map = pred_map[:, 0]

            pred_map = normalize_heatmap(pred_map)
            all_pred.append(pred_map.detach())

        elif model_name in ["mgca_cnn", "mgca_vit"]:
            text_inputs = model.tokenizer(text,
                                          add_special_tokens=True,
                                          padding='max_length',
                                          max_length=77,
                                          truncation=True,
                                          return_tensors="pt")
            input_ids = text_inputs["input_ids"].to(device)
            attention_mask = text_inputs["attention_mask"].to(device)
            pred_map = model.predict_similarity_map(
                imgs, input_ids, attention_mask)
            pred_map = pred_map[:, 0]

            # Resize into original size
            pred_map = F.interpolate(pred_map.unsqueeze(0), size=(
                imgs.shape[-2], imgs.shape[-1]), mode='bilinear', align_corners=False)
            pred_map = pred_map[:, 0]

            pred_map = normalize_heatmap(pred_map)
            all_pred.append(pred_map.detach())

        elif model_name == "chexzero":
            from cxrseg.third_party.CheXzero.clip import tokenize
            text_inputs = tokenize(text).to(device)
            _, patch_embs = model.encode_image(imgs)
            text_embs = model.encode_text(text_inputs)
            patch_embeds = F.normalize(patch_embs, p=2, dim=-1)
            text_embs = F.normalize(text_embs, p=2, dim=-1)
            pred_map = torch.bmm(text_embs.unsqueeze(
                1), patch_embeds.permute(0, 2, 1))
            h = w = int(np.sqrt(pred_map.shape[-1]))
            pred_map = rearrange(pred_map, "b c (h w) -> b c h w", h=h, w=w)

            # Gaussian filter
            from scipy import ndimage
            pred_map = torch.tensor(
                ndimage.gaussian_filter(pred_map.cpu().numpy()[
                                        0], sigma=(1.5, 1.5), order=0)
            ).unsqueeze(0).type_as(imgs)

            # Resize into original size
            pred_map = F.interpolate(pred_map.unsqueeze(0), size=(
                imgs.shape[-2], imgs.shape[-1]), mode='bilinear', align_corners=False)
            pred_map = pred_map[:, 0]

            pred_map = normalize_heatmap(pred_map)
            all_pred.append(pred_map.detach())

        elif model_name == "afloc":
            pt = model.process_text(text, device)
            res = model.text_encoder_forward(
                pt["caption_ids"].to(device),
                pt["attention_mask"].to(device),
                pt["token_type_ids"].to(device))
            iel, _, _, ieg = model.image_encoder_forward(imgs.to(device))
            teg = res["report_embeddings"]
            corr = torch.einsum('bchw, bpc -> bphw',
                                iel, teg.unsqueeze(1))
            pred_map = corr[:, 0]

            # Gaussian filter
            from scipy import ndimage
            pred_map = torch.tensor(
                ndimage.gaussian_filter(pred_map.cpu().numpy()[
                                        0], sigma=(1.5, 1.5), order=0)
            ).unsqueeze(0).type_as(imgs)

            # Resize into original size
            pred_map = F.interpolate(pred_map.unsqueeze(0), size=(
                imgs.shape[-2], imgs.shape[-1]), mode='bilinear', align_corners=False)
            pred_map = pred_map[:, 0]

            pred_map = normalize_heatmap(pred_map)
            all_pred.append(pred_map.detach())

        if label_type == "box":
            batch_box_label = torch.cat(
                [x["bbox_label"].to(device) for x in batch], dim=0)
            batch_box = torch.cat([x["bbox"].to(device) for x in batch], dim=0)
            batch_mask = []
            for box_label in torch.unique(batch_box_label):
                # compute mask for each box
                cur_mask = torch.zeros(
                    1, all_pred[0].shape[-2], all_pred[0].shape[-1]).to(device)
                box_indices = torch.where(batch_box_label == box_label)[0]
                for i in box_indices:
                    cur_box = batch_box[i].long()
                    cur_mask[:, cur_box[1]:cur_box[3],
                             cur_box[0]:cur_box[2]] = 1.
                batch_mask.append(cur_mask)
            batch_mask = torch.cat(batch_mask, dim=0)
            # there are multiple distinct labels for one image
            if len(batch_mask) > 1:
                num_repli = len(batch_mask) - 1
                all_pred.append(pred_map.detach().repeat(num_repli, 1, 1))
                all_imgs.append(imgs.repeat(num_repli, 1, 1, 1))
                all_imgid.extend(cur_imgid * num_repli)
            gt_masks.append(batch_mask)
            all_labels.append(torch.unique(batch_box_label))

        elif label_type == "mask":
            # FIXME: refine this code
            batch_mask = torch.cat([x["mask"].to(device)
                                    for x in batch], dim=0)
            gt_masks.append(batch_mask)
        else:
            raise NotImplementedError

    all_pred = torch.cat(all_pred, axis=0).unsqueeze(1)
    gt_masks = torch.cat(gt_masks, axis=0).unsqueeze(1)
    all_labels = torch.cat(all_labels, axis=0)
    all_imgs = torch.cat(all_imgs, axis=0)
    assert len(all_pred) == len(gt_masks) == len(
        all_labels) == len(all_imgid) == len(all_imgs)

    save_dir = os.path.join(
        save_dir, f"{model_name}")
    os.makedirs(save_dir, exist_ok=True)
    compute_seg_metrics(all_pred, gt_masks, all_labels,
                        class_names, save_dir)


def main(hparams: Namespace):
    model = get_model(hparams)

    model_name = hparams.model_name
    if model_name == "biovil":
        mean, std = 0, 1
        imagesize = 480
    elif model_name == "biovil_t":
        mean, std = 0, 1
        imagesize = 448
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
        bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        imagesize = 224
    elif model_name == "gloria_chexpert":
        mean, std = 0.5, 0.5
        bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        imagesize = 224
    elif model_name in ["medklip", "kad_resnet_224"]:
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

    # TODO: We can't use batch size larger than 1, because one image might have multiple phrases
    hparams.batch_size = 1
    dataloader, dataset = create_phrase_grounding_dataloader(
        dataset_dir=hparams.dataset_dir,
        dataset_name="ms_cxr",
        batch_size=hparams.batch_size,
        num_workers=hparams.num_workers,
        imagesize=imagesize,
        mean=mean,
        std=std
    )
    class_names = dataset.pathologies

    phrase_grounding_evaluation(
        model, dataloader, class_names, model_name, hparams.mask_label_type, hparams.save_dir)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset_dir", type=str,
                        default="/disk1/fywang/CXR_dataset")
    # default="/data1/r20user2/CXR_dataset")
    parser.add_argument("--use_negative_prompt", action="store_true")
    # use another conda environment if evaluating medclip
    parser.add_argument("--model_name", type=str, default="our_medclip",
                        choices=["medclip_vit", "medclip_cnn", "convirt", "gloria_chexpert",
                                 "gloria", "biovil", "biovil_t", "our_medclip",
                                 "medklip", "kad_resnet_224", "kad_resnet_512", "kad_resnet_1024",
                                 "mgca_cnn", "mgca_vit", "chexzero", "afloc",
                                 "random"])
    parser.add_argument("--mask_label_type", type=str, default="box",
                        choices=["box", "mask"])
    parser.add_argument("--save_dir", type=str,
                        default="/home/fywang/Documents/CXRSeg/evaluation_results/phrase_grounding")
    parser.add_argument("--ckpt_path", type=str,
                        # gloria
                        # default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_03_18_23_27_36/epoch=12-step=13715.ckpt")
                        # convirt
                        # default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_03_21_22_25_20/epoch=12-step=27417.ckpt")
                        # medclip
                        # default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_04_04_02_39_54/epoch=15-step=6720.ckpt")
                        # default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_04_03_22_57_29/epoch=16-step=7140.ckpt")
                        # default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_04_03_23_00_12/epoch=10-step=4620.ckpt")
                        # default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_04_04_06_50_39/epoch=9-step=4200.ckpt")
                        # default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_04_05_20_39_04/epoch=18-step=7980.ckpt")
                        default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_04_12_22_57_00/epoch=9-step=21000.ckpt")
    args = parser.parse_args()
    main(args)
