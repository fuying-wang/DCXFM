# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import numpy as np
import torch
import ipdb

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


class CXRSegVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, class_names=None):
        super().__init__(img_rgb, metadata, scale, instance_mode)

    def draw_sem_seg(self, sem_seg, text, mask_color, area_threshold=None, alpha=0.8):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        self.draw_binary_mask(
            sem_seg,
            color=mask_color,
            edge_color=(1.0, 1.0, 240.0 / 255),
            text=text,
            alpha=alpha,
            area_threshold=area_threshold,
        )
        return self.output



class VisualizationDemo(object):
    def __init__(self, cfg, instance_mode=ColorMode.IMAGE, parallel=False):
        """
        Args:
            cfg (CfgNode):
            instance_mode (ColorMode):
            parallel (bool): whether to run the model in different processes from visualization.
                Useful since the visualization logic can be slow.
        """
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode

        self.parallel = parallel
        if parallel:
            raise NotImplementedError
        else:
            self.predictor = DefaultPredictor(cfg)
        self.class_names = self.metadata.thing_classes
        self.mask_colors = self.metadata.mask_colors

    def run_on_image(self, image):
        """
        Args:
            image (np.ndarray): an image of shape (H, W, C) (in BGR order).
                This is the format used by OpenCV.
        Returns:
            predictions (dict): the output of the model.
            vis_output (VisImage): the visualized image output.
        """
        predictions = self.predictor(image)
        # # Convert image from OpenCV BGR format to Matplotlib RGB format.
        image = image[:, :, ::-1]
        pred_mask = predictions["sem_seg"].cpu().numpy()
        pred_mask = (pred_mask >= 0.5).astype(np.uint8)
        vis_output = []
        for i in range(len(pred_mask)):
            class_name = self.class_names[i]
            mask_color = self.mask_colors[i]
            # scale RGB values to the range [0, 1]
            mask_color = [x / 255. for x in mask_color]
            visualizer = CXRSegVisualizer(image, self.metadata, instance_mode=self.instance_mode)
            pred_mask_per_class = pred_mask[i]
            vis_output_per_class = visualizer.draw_sem_seg(
                sem_seg=pred_mask_per_class,
                text=class_name,
                mask_color=mask_color
            )
            vis_output.append(vis_output_per_class)

        return predictions, vis_output