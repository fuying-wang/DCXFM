# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# Modified by Jian Ding from: https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py
import fvcore.nn.weight_init as weight_init
import torch
import json
import ipdb
import os
import pandas as pd

from torch import nn
from torch.nn import functional as F

from detectron2.data import MetadataCatalog
from detectron2.config import configurable
from detectron2.layers import Conv2d

from .model import Aggregator

import numpy as np
from dcxfm.modeling.sdmp import SDMPModule
from dcxfm.utils.evaluation_utils import get_class_texts
# from cxrseg.utils.prompts import prompt_dict
os.environ["TOKENIZERS_PARALLELISM"] = "false"


class CATSegPredictor(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        medclip_ckpt: str,
        dataset_dir: str,
        train_class_json: str,
        test_class_json: str,
        text_guidance_dim: int,
        text_guidance_proj_dim: int,
        appearance_guidance_dim: int,
        appearance_guidance_proj_dim: int,
        prompt_depth: int,
        prompt_length: int,
        decoder_dims: list,
        decoder_guidance_dims: list,
        decoder_guidance_proj_dims: list,
        num_heads: int,
        num_layers: tuple,
        hidden_dims: tuple,
        pooling_sizes: tuple,
        feature_resolution: tuple,
        window_sizes: tuple,
        attention_type: str,
    ):
        """
        Args:

        """
        super().__init__()

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.silc_model = SDMPModule.load_from_checkpoint(
            medclip_ckpt, dataset_dir=dataset_dir, strict=False).to(device)
        self.tokenizer = self.silc_model.tokenizer

        transformer = Aggregator(
            text_guidance_dim=text_guidance_dim,
            text_guidance_proj_dim=text_guidance_proj_dim,
            appearance_guidance_dim=appearance_guidance_dim,
            appearance_guidance_proj_dim=appearance_guidance_proj_dim,
            decoder_dims=decoder_dims,
            decoder_guidance_dims=decoder_guidance_dims,
            decoder_guidance_proj_dims=decoder_guidance_proj_dims,
            num_layers=num_layers,
            nheads=num_heads,
            hidden_dim=hidden_dims,
            pooling_size=pooling_sizes,
            feature_resolution=feature_resolution,
            window_size=window_sizes,
            attention_type=attention_type
        )
        self.transformer = transformer

    @staticmethod
    def count_model_size(model):
        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024**2
        # print('model size: {:.3f}MB'.format(size_all_mb))
        return size_all_mb

    @classmethod
    def from_config(cls, cfg):  # , in_channels, mask_classification):
        ret = {}
        ret["medclip_ckpt"] = cfg.MODEL.SEM_SEG_HEAD.MEDCLIP_CKPT
        ret["dataset_dir"] = cfg.MODEL.SEM_SEG_HEAD.DATASET_DIR
        ret["train_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TRAIN_CLASS_JSON
        ret["test_class_json"] = cfg.MODEL.SEM_SEG_HEAD.TEST_CLASS_JSON

        # Aggregator parameters:
        ret["text_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_DIM
        ret["text_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.TEXT_GUIDANCE_PROJ_DIM
        ret["appearance_guidance_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_DIM
        ret["appearance_guidance_proj_dim"] = cfg.MODEL.SEM_SEG_HEAD.APPEARANCE_GUIDANCE_PROJ_DIM

        ret["decoder_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_DIMS
        ret["decoder_guidance_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_DIMS
        ret["decoder_guidance_proj_dims"] = cfg.MODEL.SEM_SEG_HEAD.DECODER_GUIDANCE_PROJ_DIMS

        ret["prompt_depth"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_DEPTH
        ret["prompt_length"] = cfg.MODEL.SEM_SEG_HEAD.PROMPT_LENGTH

        ret["num_layers"] = cfg.MODEL.SEM_SEG_HEAD.NUM_LAYERS
        ret["num_heads"] = cfg.MODEL.SEM_SEG_HEAD.NUM_HEADS
        ret["hidden_dims"] = cfg.MODEL.SEM_SEG_HEAD.HIDDEN_DIMS
        ret["pooling_sizes"] = cfg.MODEL.SEM_SEG_HEAD.POOLING_SIZES
        ret["feature_resolution"] = cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION
        ret["window_sizes"] = cfg.MODEL.SEM_SEG_HEAD.WINDOW_SIZES
        ret["attention_type"] = cfg.MODEL.SEM_SEG_HEAD.ATTENTION_TYPE

        return ret

    def get_class_name_list(self, dataset_name):
        class_names = [
            c.strip() for c in MetadataCatalog.get(dataset_name).thing_classes
        ]
        return class_names

    def forward(self, x, vis_guidance, dataset_name, image_ids):
        '''
        x: bz, 768, 7, 7
        vis_guidance: dict()
            - res2: bz, 256, 96, 96
        dataset_name: the name of datasets
        '''
        class_names = self.get_class_name_list(dataset_name)
        if dataset_name == "mscxr_val_pg":
            df = pd.read_csv(
                "/home/fywang/Documents/CXRSeg/cxr_data/ms-cxr/0.1/MS_CXR_Local_Alignment_v1.0.0.csv")
            agg_df = df.drop_duplicates(subset=['dicom_id', 'label_text'])
            all_text = []
            for imgid in image_ids:
                label_text = agg_df.iloc[imgid - 1]['label_text']
                all_text.append(label_text)

            pos_texts = self.silc_model.tokenizer(all_text, return_tensors="pt",
                                                  padding="max_length", max_length=77)

            input_ids = pos_texts["input_ids"].cuda()
            attention_mask = pos_texts["attention_mask"].cuda(
            )
            _, text_embs, _ = self.silc_model.encode_text_student(
                input_ids=input_ids, attention_mask=attention_mask)
            text = text_embs.unsqueeze(1).unsqueeze(1)
            # HACK: repeat the text for 10 times for evaluation
            text = text.repeat(1, len(class_names), 4, 1)
        else:
            text_features, _ = self.class_embeddings(class_names=class_names)
            # vis = [vis_guidance[k] for k in vis_guidance.keys()][::-1]
            text = text_features
            text = text.repeat(x.shape[0], 1, 1, 1)

        out = self.transformer(x, text, None)
        # out = self.transformer(x, text, vis)

        return out

    @torch.no_grad()
    def class_embeddings(self, class_names, num_templates=4):
        '''
        Encode each caption into embedding.
        Return:
          - zeroshot_weights: (number of pathologies, number of templates, embedding dim)
        '''
        tokenized_pos_texts, tokenized_neg_texts = get_class_texts(self.silc_model.tokenizer,
                                                                   class_names,
                                                                   prompt_style="xplainer",
                                                                   num_templates=num_templates,
                                                                   use_negative=True)
        pos_text_embeds = []
        for pathology in class_names:
            input_ids = tokenized_pos_texts[pathology]["input_ids"].cuda()
            attention_mask = tokenized_pos_texts[pathology]["attention_mask"].cuda(
            )
            _, text_embs, _ = self.silc_model.encode_text_student(
                input_ids=input_ids, attention_mask=attention_mask)
            pos_text_embeds.append(text_embs)
        pos_text_embeds = torch.stack(pos_text_embeds, dim=0)  # 10, 4, 128

        neg_text_embeds = []
        for pathology in class_names:
            input_ids = tokenized_neg_texts[pathology]["input_ids"].cuda()
            attention_mask = tokenized_neg_texts[pathology]["attention_mask"].cuda(
            )
            _, text_embs, _ = self.silc_model.encode_text_student(
                input_ids=input_ids, attention_mask=attention_mask)
            neg_text_embeds.append(text_embs)
        neg_text_embeds = torch.stack(neg_text_embeds, dim=0)  # 10, 4, 128

        return pos_text_embeds, neg_text_embeds
