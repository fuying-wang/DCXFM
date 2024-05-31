# Copyright (c) Facebook, Inc. and its affiliates.
# Copyright (c) Meta Platforms, Inc. All Rights Reserved

import logging
import itertools
import pickle
import json
from detectron2.evaluation.sem_seg_evaluation import load_image_into_numpy_array
import numpy as np
import ipdb
import os
from collections import OrderedDict
import PIL.Image as Image
import torch

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.utils.comm import all_gather, is_main_process, synchronize
from detectron2.utils.file_io import PathManager

from detectron2.evaluation import SemSegEvaluator
import pycocotools.mask as mask_util

_CV2_IMPORTED = True
try:
    import cv2  # noqa
except ImportError:
    # OpenCV is an optional dependency at the moment
    _CV2_IMPORTED = False


class CXRSegEvaluator(SemSegEvaluator):
    def __init__(self,
                 dataset_name,
                 distributed=True,
                 output_dir=None,
                 *,
                 ignore_label=None):
        """
        Args:
            dataset_name (str): name of the dataset to be evaluated.
            distributed (bool): if True, will collect results from all ranks for evaluation.
                Otherwise, will evaluate the results in the current process.
            output_dir (str): an output directory to dump results.
            sem_seg_loading_fn: function to read sem seg file and load into numpy array.
                Default provided, but projects can customize.
            num_classes, ignore_label: deprecated argument
        """
        self._logger = logging.getLogger(__name__)
        # if num_classes is not None:
        #     self._logger.warn(
        #         "SemSegEvaluator(num_classes) is deprecated! It should be obtained from metadata."
        #     )
        if ignore_label is not None:
            self._logger.warn(
                "SemSegEvaluator(ignore_label) is deprecated! It should be obtained from metadata."
            )
        self._dataset_name = dataset_name
        self._distributed = distributed
        self._output_dir = output_dir

        self._cpu_device = torch.device("cpu")

        meta = MetadataCatalog.get(dataset_name)
        # Dict that maps contiguous training ids to COCO category ids
        try:
            c2d = meta.thing_dataset_id_to_contiguous_id
            self._contiguous_id_to_dataset_id = {v: k for k, v in c2d.items()}
        except AttributeError:
            self._contiguous_id_to_dataset_id = None
        self._class_names = meta.thing_classes
        self._num_classes = len(meta.thing_classes)
        # if num_classes is not None:
        #     assert self._num_classes == num_classes, f"{self._num_classes} != {num_classes}"
        self._ignore_label = ignore_label if ignore_label is not None else meta.ignore_label

        # This is because cv2.erode did not work for int datatype. Only works for uint8.
        self._compute_boundary_iou = False
        if not _CV2_IMPORTED:
            self._compute_boundary_iou = False
            self._logger.warn(
                """Boundary IoU calculation requires OpenCV. B-IoU metrics are
                not going to be computed because OpenCV is not available to import."""
            )
        if self._num_classes >= np.iinfo(np.uint8).max:
            self._compute_boundary_iou = False
            self._logger.warn(
                f"""SemSegEvaluator(num_classes) is more than supported value for Boundary IoU calculation!
                B-IoU metrics are not going to be computed. Max allowed value (exclusive)
                for num_classes for calculating Boundary IoU is {np.iinfo(np.uint8).max}.
                The number of classes of dataset {self._dataset_name} is {self._num_classes}"""
            )

    def reset(self):
        self._iou_scores = [0 for _ in range(self._num_classes)]
        self._dice_scores = [0 for _ in range(self._num_classes)]
        self._point_games = [0 for _ in range(self._num_classes)]
        self._num_pairs = [0 for _ in range(self._num_classes)]
        self._predictions = []

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a model.
                It is a list of dicts. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name".
            outputs: the outputs of a model. It is either list of semantic segmentation predictions
                (Tensor [H, W]) or list of dicts with key "sem_seg" that contains semantic
                segmentation prediction in the same format.
        """

        for input, output in zip(inputs, outputs):
            gt_masks = input["instances"].gt_masks.tensor
            gt_classes = input["instances"].gt_classes

            for idx, class_id in enumerate(gt_classes):
                gt_mask = gt_masks[idx]
                pred_heatmap = output["sem_seg"][class_id]
                gt_mask = gt_mask.cpu().numpy()
                pred_heatmap = pred_heatmap.cpu().numpy()
                gt_mask_flatten = gt_mask.flatten()
                pred_heatmap_flatten = pred_heatmap.flatten()
                max_val = np.max(pred_heatmap_flatten)
                max_mask = gt_mask_flatten[pred_heatmap_flatten == max_val].astype(
                    np.float32)
                point_score = (np.sum(max_mask) > 0).astype(np.float32)
                self._point_games[class_id] += point_score
                self._num_pairs[class_id] += 1

                _iou_scores = []
                _dice_scores = []
                for thres in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    pred_mask = (pred_heatmap >= thres).astype(np.int64)
                    tn, fn, fp, tp = np.bincount(
                        2 * pred_mask.reshape(-1) + gt_mask.reshape(-1), minlength=4
                    )
                    # skip class if no ground truth instances
                    if (tp + fp + fn) == 0:
                        continue
                    _iou_scores.append(tp / (tp + fp + fn))
                    _dice_scores.append((2 * tp) / (2 * tp + fp + fn))

                self._iou_scores[class_id] += np.mean(_iou_scores)
                self._dice_scores[class_id] += np.mean(_dice_scores)

            pred = output["sem_seg"].float().cpu().numpy()
            self._predictions.append(
                {"file_name": input["file_name"],
                    "sem_seg": pred,
                    "gt_classes": gt_classes.cpu().numpy(),
                    "gt_masks": gt_masks.cpu().numpy()}
            )

    def evaluate(self):
        """
        Evaluates standard semantic segmentation metrics (http://cocodataset.org/#stuff-eval):

        * Mean intersection-over-union averaged across classes (mIoU)
        * Frequency Weighted IoU (fwIoU)
        * Mean pixel accuracy averaged across classes (mACC)
        * Pixel Accuracy (pACC)
        """
        if self._distributed:
            synchronize()
            iou_score_list = all_gather(self._iou_scores)
            dice_score_list = all_gather(self._dice_scores)
            point_game_list = all_gather(self._point_games)
            n_pairs_list = all_gather(self._num_pairs)
            self._predictions = all_gather(self._predictions)
            self._predictions = list(itertools.chain(*self._predictions))
            if not is_main_process():
                return

            self.iou_score_list = [0 for _ in range(self._num_classes)]
            self.dice_score_list = [0 for _ in range(self._num_classes)]
            self.point_game_list = [0 for _ in range(self._num_classes)]
            for iou_scores in iou_score_list:
                for i in range(self._num_classes):
                    self.iou_score_list[i] += iou_scores[i]
            for dice_scores in dice_score_list:
                for i in range(self._num_classes):
                    self.dice_score_list[i] += dice_scores[i]
            for point_games in point_game_list:
                for i in range(self._num_classes):
                    self.point_game_list[i] += point_games[i]
            for n_pairs in n_pairs_list:
                for i in range(self._num_classes):
                    self._num_pairs[i] += n_pairs[i]

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)

            file_path = os.path.join(
                self._output_dir, f"{self._dataset_name}_sem_seg_predictions.pkl")
            with PathManager.open(file_path, "wb") as f:
                pickle.dump(self._predictions, f)

        res = {}
        for i in range(self._num_classes):
            cur_class_name = self._class_names[i]
            res[f"{cur_class_name}_IoU"] = (
                self.iou_score_list[i] / self._num_pairs[i])
            res[f"{cur_class_name}_Dice"] = (
                self.dice_score_list[i] / self._num_pairs[i])
            res[f"{cur_class_name}_PointGame"] = (
                self.point_game_list[i] / self._num_pairs[i])

        res["Iou"] = np.mean(
            np.array(self.iou_score_list) / np.array(self._num_pairs))
        res["Dice"] = np.mean(
            np.array(self.dice_score_list) / np.array(self._num_pairs))
        res["PointGame"] = np.mean(
            np.array(self.point_game_list) / np.array(self._num_pairs))
        results = OrderedDict({"sem_seg": res})
        from pprint import pprint
        pprint(results)

        return results

    # def encode_json_sem_seg(self, sem_seg, input_file_name):
    #     """
    #     Convert semantic segmentation to COCO stuff format with segments encoded as RLEs.
    #     See http://cocodataset.org/#format-results
    #     """
    #     json_list = []
    #     for label in range(sem_seg.shape[0]):
    #         mask = sem_seg[label].astype(np.uint8)
    #         mask_rle = mask_util.encode(
    #             np.array(mask[:, :, None], order="F"))[0]
    #         mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
    #         json_list.append(
    #             {"file_name": input_file_name,
    #                 "category_id": label, "segmentation": mask_rle}
    #         )
    #     return json_list

    # def _mask_to_boundary(self, mask: np.ndarray, dilation_ratio=0.02):
    #     assert mask.ndim == 2, "mask_to_boundary expects a 2-dimensional image"
    #     h, w = mask.shape
    #     diag_len = np.sqrt(h**2 + w**2)
    #     dilation = max(1, int(round(dilation_ratio * diag_len)))
    #     kernel = np.ones((3, 3), dtype=np.uint8)

    #     padded_mask = cv2.copyMakeBorder(
    #         mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    #     eroded_mask_with_padding = cv2.erode(
    #         padded_mask, kernel, iterations=dilation)
    #     eroded_mask = eroded_mask_with_padding[1:-1, 1:-1]
    #     boundary = mask - eroded_mask
    #     return boundary
