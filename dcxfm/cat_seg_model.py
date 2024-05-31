# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import ipdb

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import ImageList
from detectron2.utils.memory import _ignore_torch_cuda_oom
from segmentation_models_pytorch.losses import DiceLoss, FocalLoss, BINARY_MODE, SoftCrossEntropyLoss
from einops import rearrange


@META_ARCH_REGISTRY.register()
class CATSeg(nn.Module):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        size_divisibility: int,
        num_classes: int,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        clip_pixel_mean: Tuple[float],
        clip_pixel_std: Tuple[float],
        clip_finetune: str,
        sliding_window: bool,
        backbone_multiplier: float,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
        """
        super().__init__()

        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        if size_divisibility < 0:
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.num_classes = num_classes

        self.register_buffer("pixel_mean", torch.Tensor(
            pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(
            pixel_std).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_mean", torch.Tensor(
            clip_pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("clip_pixel_std", torch.Tensor(
            clip_pixel_std).view(-1, 1, 1), False)

        # finetune_backbone = backbone_multiplier > 0.
        # # backbone is not used in our case
        # for name, params in self.backbone.named_parameters():
        #     if "norm0" in name:
        #         params.requires_grad = False
        #     else:
        #         params.requires_grad = finetune_backbone

        # Finetune SILC
        for name, params in self.sem_seg_head.predictor.silc_model.named_parameters():
            if "text" in name:
                params.requires_grad = False
            else:
                if clip_finetune == "prompt":
                    params.requires_grad = True if "prompt" in name else False
                elif clip_finetune == "attention":
                    if "attn" in name:
                        # QV fine-tuning for attention blocks
                        params.requires_grad = True if "q_proj" in name or "v_proj" in name else False
                    elif "position" in name:
                        params.requires_grad = True
                    else:
                        params.requires_grad = False
                elif clip_finetune == "full":
                    params.requires_grad = True
                else:
                    params.requires_grad = False

        self.sliding_window = sliding_window
        self.sequential = False

        # medclip resolution
        self.clip_resolution = (512, 512)
        # self.proj_dim = 768 if clip_pretrained == "ViT-B/16" else 1024
        self.proj_dim = 128
        self.upsample1 = nn.ConvTranspose2d(
            self.proj_dim, 256, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(
            self.proj_dim, 128, kernel_size=4, stride=4)

        self.dice_loss = DiceLoss(mode=BINARY_MODE, from_logits=True)
        self.focal_loss = FocalLoss(mode=BINARY_MODE)
        # self.bce_loss = SoftCrossEntropyLoss()

    @classmethod
    def from_config(cls, cfg):
        # backbone = build_backbone(cfg)
        backbone = None
        sem_seg_head = build_sem_seg_head(cfg, None)

        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "clip_pixel_mean": cfg.MODEL.CLIP_PIXEL_MEAN,
            "clip_pixel_std": cfg.MODEL.CLIP_PIXEL_STD,
            "clip_finetune": cfg.MODEL.SEM_SEG_HEAD.CLIP_FINETUNE,
            "sliding_window": cfg.TEST.SLIDING_WINDOW,
            "backbone_multiplier": cfg.SOLVER.BACKBONE_MULTIPLIER,
            "num_classes": cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def prepare_targets(self, targets, images):
        '''
        Create gt masks per class
        '''
        h, w = images.tensor.shape[-2:]
        padded_masks = torch.zeros(
            len(targets), self.num_classes, h, w).to(self.device)
        for i, targets_per_image in enumerate(targets):
            for class_id, gt_mask in zip(targets_per_image.gt_classes, targets_per_image.gt_masks):
                padded_masks[i, class_id] = gt_mask

        return padded_masks

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
        """
        dataset_name = [x["meta"]["dataset_name"] for x in batched_inputs]
        assert len(set(dataset_name)) == 1
        dataset_name = dataset_name[0]

        images = [x["image"].to(self.device) for x in batched_inputs]
        image_ids = [x["image_id"] for x in batched_inputs]
        if not self.training and self.sliding_window:
            return self.inference_sliding_window(batched_inputs)

        clip_images = [(x - self.clip_pixel_mean) /
                       self.clip_pixel_std for x in images]
        clip_images = ImageList.from_tensors(
            clip_images, self.size_divisibility)

        # images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        # images = ImageList.from_tensors(images, self.size_divisibility)

        # forward silc
        clip_images_resized = F.interpolate(
            clip_images.tensor, size=self.clip_resolution, mode='bilinear', align_corners=False)
        _, patch_feats = self.sem_seg_head.predictor.silc_model.encode_image_student(
            clip_images_resized, dense=True)
        clip_features = self.sem_seg_head.predictor.silc_model.img_projection_student(
            patch_feats)

        # image_features = clip_features.clone()
        # self.layers = []
        # # CLIP ViT features for guidance
        # res3 = rearrange(image_features, "B (H W) C -> B C H W", H=24)
        # res4 = rearrange(self.layers[0][1:, :, :],
        #                  "(H W) B C -> B C H W", H=24)
        # res5 = rearrange(self.layers[1][1:, :, :],
        #                  "(H W) B C -> B C H W", H=24)
        # res4 = self.upsample1(res4)
        # res5 = self.upsample2(res5)
        # features = {'res5': res5, 'res4': res4, 'res3': res3, }

        # outputs = self.sem_seg_head(clip_features, features, dataset_name)
        outputs = self.sem_seg_head(
            clip_features, None, dataset_name, image_ids)
        if self.training:
            # gt_classes = [x["instances"].gt_classes for x in batched_inputs]
            gt_labs = torch.stack([torch.tensor(x["lab"])
                                  for x in batched_inputs]).to(self.device)

            gt_instances = [x["instances"].to(
                self.device) for x in batched_inputs]
            targets = self.prepare_targets(
                gt_instances, clip_images)  # bz, 18, 512, 512
            outputs = F.interpolate(outputs, size=(clip_images.tensor.shape[-2], clip_images.tensor.shape[-1]),
                                    mode="bilinear", align_corners=False)

            loss = 0.
            for (gt_lab_per_sample, output_per_sample, target_per_sample) in zip(gt_labs, outputs, targets):
                valid_indices = torch.where(gt_lab_per_sample != -1)[0]
                pos_indices = torch.where(
                    gt_lab_per_sample[valid_indices] == 1)[0]
                padded_targets = torch.zeros(len(
                    valid_indices), clip_images.tensor.shape[-2], clip_images.tensor.shape[-1]).to(self.device)
                padded_targets[pos_indices] = target_per_sample[valid_indices][pos_indices]
                cur_outputs = output_per_sample[valid_indices]
                cur_outputs = cur_outputs.unsqueeze(0)
                padded_targets = padded_targets.unsqueeze(0)
                loss_per_sample = self.dice_loss(
                    cur_outputs, padded_targets) + self.focal_loss(cur_outputs, padded_targets)
                # loss_per_sample = self.bce_loss(cur_outputs, padded_targets)
                loss += loss_per_sample
            loss /= len(gt_labs)
            losses = {"loss_sem_seg": loss}
            return losses
        else:
            outputs = outputs.sigmoid()
            image_size = clip_images.image_sizes[0]
            # height, width = image_size
            height = batched_inputs[0].get("height", image_size[0])
            width = batched_inputs[0].get("width", image_size[1])
            # By default, we use batch size 1 for evaluation
            output = sem_seg_postprocess(outputs[0], image_size, height, width)
            processed_results = [{'sem_seg': output}]

            return processed_results

    @torch.no_grad()
    def inference_sliding_window(self, batched_inputs, kernel=384, overlap=0.333, out_res=[640, 640]):
        images = [x["image"].to(self.device, dtype=torch.float32)
                  for x in batched_inputs]
        stride = int(kernel * (1 - overlap))
        unfold = nn.Unfold(kernel_size=kernel, stride=stride)
        fold = nn.Fold(out_res, kernel_size=kernel, stride=stride)

        image = F.interpolate(images[0].unsqueeze(
            0), size=out_res, mode='bilinear', align_corners=False).squeeze()
        image = rearrange(unfold(image), "(C H W) L-> L C H W", C=3, H=kernel)
        global_image = F.interpolate(images[0].unsqueeze(0), size=(
            kernel, kernel), mode='bilinear', align_corners=False)
        image = torch.cat((image, global_image), dim=0)

        images = (image - self.pixel_mean) / self.pixel_std
        clip_images = (image - self.clip_pixel_mean) / self.clip_pixel_std
        clip_images = F.interpolate(
            clip_images, size=self.clip_resolution, mode='bilinear', align_corners=False, )

        self.layers = []
        clip_features = self.sem_seg_head.predictor.clip_model.encode_image(
            clip_images, dense=True)
        res3 = rearrange(clip_features[:, 1:, :], "B (H W) C -> B C H W", H=24)
        res4 = self.upsample1(
            rearrange(self.layers[0][1:, :, :], "(H W) B C -> B C H W", H=24))
        res5 = self.upsample2(
            rearrange(self.layers[1][1:, :, :], "(H W) B C -> B C H W", H=24))

        features = {'res5': res5, 'res4': res4, 'res3': res3, }
        outputs = self.sem_seg_head(clip_features, features)

        outputs = F.interpolate(outputs, size=kernel,
                                mode="bilinear", align_corners=False)
        outputs = outputs.sigmoid()

        global_output = outputs[-1:]
        global_output = F.interpolate(
            global_output, size=out_res, mode='bilinear', align_corners=False,)
        outputs = outputs[:-1]
        outputs = fold(outputs.flatten(1).T) / \
            fold(unfold(torch.ones([1] + out_res, device=self.device)))
        outputs = (outputs + global_output) / 2.

        height = batched_inputs[0].get("height", out_res[0])
        width = batched_inputs[0].get("width", out_res[1])
        output = sem_seg_postprocess(outputs[0], out_res, height, width)
        return [{'sem_seg': output}]
