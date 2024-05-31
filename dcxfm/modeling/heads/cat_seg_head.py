# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from copy import deepcopy
from typing import Callable, Dict, List, Optional, Tuple, Union
from einops import rearrange

import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer.cat_seg_predictor import CATSegPredictor


@SEM_SEG_HEADS_REGISTRY.register()
class CATSegHead(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        num_classes: int,
        ignore_value: int = -1,
        # extra parameters
        feature_resolution: list,
        transformer_predictor: nn.Module,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            num_classes: number of classes to predict
            pixel_decoder: the pixel decoder module
            loss_weight: loss weight
            ignore_value: category id to be ignored during training.
            transformer_predictor: the transformer decoder that makes prediction
            transformer_in_feature: input feature name to the transformer_predictor
        """
        super().__init__()
        self.ignore_value = ignore_value
        self.predictor = transformer_predictor
        self.num_classes = num_classes
        self.feature_resolution = feature_resolution

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        return {
            "ignore_value": cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
            "feature_resolution": cfg.MODEL.SEM_SEG_HEAD.FEATURE_RESOLUTION,
            "transformer_predictor": CATSegPredictor(
                cfg,
            ),
        }

    def forward(self, features, guidance_features, dataset_name, image_ids):
        """
        Arguments:
            features: (B, HW, C)
            guidance_features: dict()
                - res2: bz, 256, 96, 96
                - res3: bz, 512, 48, 48
                - res4: bz, 1024, 24, 24
        """
        img_feat = rearrange(features, "b (h w) c->b c h w",
                             h=self.feature_resolution[0], w=self.feature_resolution[1])
        # img_feat = rearrange(features[:, 1:, :], "b (h w) c->b c h w", h=self.feature_resolution[0], w=self.feature_resolution[1])
        return self.predictor(img_feat, guidance_features, dataset_name, image_ids)
