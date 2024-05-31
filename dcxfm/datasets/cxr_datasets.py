import collections
import os
import os.path
import pprint
import random
import sys
import tarfile
import warnings
import zipfile
import pickle
import re
import nltk
import cv2
import json

import imageio
import numpy as np
import pandas as pd
from tqdm import tqdm
from copy import deepcopy
import skimage
from typing import Dict, List
import skimage.transform
# from skimage.io import imread
from cv2 import imread
import torch
from torchvision import transforms
from transformers import AutoTokenizer
from datasets import Dataset as Dataset_hg
from dcxfm.utils.constants import CHEXPERT_COMPETITION_TASKS, CHEXPERT_UNCERTAIN_MAPPINGS, CHEXPERT_TASKS
from dcxfm.datasets.transforms import DataAugmentationDINO
import albumentations as A
# from cxrseg.utils.constants importlocalization_tasks
# import torchxrayvision as xrv
import ipdb
import warnings
warnings.filterwarnings('ignore')

'''
Most of this script is borrowed from https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py. 
'''

default_pathologies = ["Enlarged Cardiomediastinum",
                       "Cardiomegaly",
                       "Lung Opacity",
                       "Lung Lesion",
                       "Edema",
                       "Consolidation",
                       "Pneumonia",
                       "Atelectasis",
                       "Pneumothorax",
                       "Pleural Effusion",
                       "Pleural Other",
                       "Fracture",
                       "Support Devices"]

# Use a file that ships with the library
USE_INCLUDED_FILE = "USE_INCLUDED_FILE"

thispath = os.path.dirname(os.path.realpath(__file__))
datapath = os.path.join(thispath, "data")

# this is for caching small things for speed
_cache_dict = {}


def normalize(img, maxval, reshape=False):
    """Scales images to be roughly [-1024 1024].

    Call xrv.utils.normalize moving forward.
    """
    if img.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(
            img.max(), maxval))

    # img = (2 * (img.astype(np.float32) / maxval) - 1.)
    # norm image into [0, 1]
    img = (img.astype(np.float32) / maxval)
    img = (255 * img).astype(np.uint8)

    if reshape:
        # Check that images are 2D arrays
        if len(img.shape) > 2:
            raise RuntimeError("error, dimension larger than 2 for image")
        if len(img.shape) < 2:
            raise RuntimeError("error, dimension lower than 2 for image")

        # convert img to standard RGB format
        img = img[:, :, None]
        img = np.repeat(img, 3, axis=2)

    return img


def box2mask(boxes, height, width):
    ''' Overlap multiple boxes of the same label into a mask ...'''
    # Initalize the pathological mask
    batch_mask = torch.zeros(1, height, width).type_as(boxes).long()
    for i, box in enumerate(boxes):
        box = box.long()
        batch_mask[:, box[1]:box[3], box[0]:box[2]] = 1
    return batch_mask


def box_collate_fn(batch):
    # return a list of dict
    return batch


def apply_transforms(sample, transform, seed=None) -> Dict:
    """Applies transforms to the image and masks.
    The seeds are set so that the transforms that are applied
    to the image are the same that are applied to each mask.
    This way data augmentation will work for segmentation or 
    other tasks which use masks information.

    We reimplement this function with albumentations to handle masks and boxes.
    """

    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is None:
        return sample

    if "mask" in sample:
        if "bbox" in sample:
            transformed_sample = transform(
                image=sample["img"],
                mask=sample["mask"],
                bboxes=sample["bbox"],
                class_labels=sample["bbox_label"])
            sample["img"] = transformed_sample["image"]
            sample["mask"] = transformed_sample["mask"].permute(2, 0, 1)
            sample["bbox"] = torch.tensor(
                np.array(transformed_sample["bboxes"]), dtype=torch.float32)
            sample["bbox_label"] = torch.tensor(
                np.array(transformed_sample["class_labels"]), dtype=torch.int64)
        else:
            transformed_sample = transform(
                image=sample["img"],
                mask=sample["mask"])
            sample["img"] = transformed_sample["image"]
            sample["mask"] = transformed_sample["mask"].permute(2, 0, 1)
    elif "bbox" in sample:
        transformed_sample = transform(
            image=sample["img"],
            bboxes=sample["bbox"],
            class_labels=sample["bbox_label"])
        sample["img"] = transformed_sample["image"]
        sample["bbox"] = torch.tensor(
            np.array(transformed_sample["bboxes"]), dtype=torch.float32)
        sample["bbox_label"] = torch.tensor(
            np.array(transformed_sample["class_labels"]), dtype=torch.int64)
    else:
        transformed_sample = transform(
            image=sample["img"])
        sample["img"] = transformed_sample["image"]

    return sample


def apply_dino_transforms(sample, transform, seed=None) -> Dict:
    ''' Apply dino data augmentations'''
    if seed is None:
        MAX_RAND_VAL = 2147483647
        seed = np.random.randint(MAX_RAND_VAL)

    if transform is None:
        return sample

    dino_imgs_dict = transform(sample["img"])
    sample["img_global"] = dino_imgs_dict["global_crops"]
    if "local_crops" in dino_imgs_dict:
        sample["img_local"] = dino_imgs_dict["local_crops"]

    return sample


def relabel_dataset(pathologies, dataset, silent=False):
    """This function will add, remove, and reorder the `.labels` field to
have the same order as the pathologies argument passed to it. If a pathology is specified but doesn’t
exist in the dataset then a NaN will be put in place of the label.

    Args:
        :pathologies: The list of pathologies that the dataset will be aligned.
        :dataset: The dataset object that will be edited.
        :silent: Set True to silence printing details of the alignment.
    """
    will_drop = set(dataset.pathologies).difference(pathologies)
    if will_drop != set():
        if not silent:
            print("{} will be dropped".format(will_drop))
    new_labels = []
    dataset.pathologies = list(dataset.pathologies)
    for pathology in pathologies:
        if pathology in dataset.pathologies:
            pathology_idx = dataset.pathologies.index(pathology)
            new_labels.append(dataset.labels[:, pathology_idx])
        else:
            if not silent:
                print("{} doesn't exist. Adding nans instead.".format(pathology))
            values = np.empty(dataset.labels.shape[0])
            values.fill(np.nan)
            new_labels.append(values)
    new_labels = np.asarray(new_labels).T

    dataset.labels = new_labels
    dataset.pathologies = pathologies


class BaseDataset:
    """The datasets in this library aim to fit a simple interface where the
    imgpath and csvpath are specified. Some datasets require more than one
    metadata file and for some the metadata files are packaged in the library
    so only the imgpath needs to be specified.
    """

    def __init__(self):
        pass

    pathologies: List[str]
    """A list of strings identifying the pathologies contained in this 
    dataset. This list corresponds to the columns of the `.labels` matrix. 
    Although it is called pathologies, the contents do not have to be 
    pathologies and may simply be attributes of the patient. """

    labels: np.ndarray
    """A NumPy array which contains a 1, 0, or NaN for each pathology. Each 
    column is a pathology and each row corresponds to an item in the dataset. 
    A 1 represents that the pathology is present, 0 represents the pathology 
    is absent, and NaN represents no information. """

    csv: pd.DataFrame
    """A Pandas DataFrame of the metadata .csv file that is included with the 
    data. For some datasets multiple metadata files have been merged 
    together. It is largely a "catch-all" for associated data and the 
    referenced publication should explain each field. Each row aligns with 
    the elements of the dataset so indexing using .iloc will work. Alignment 
    between the DataFrame and the dataset items will be maintained when using 
    tools from this library. """

    def totals(self) -> Dict[str, Dict[str, int]]:
        """Compute counts of pathologies.

        Returns: A dict containing pathology name -> (label->value)
        """
        counts = [dict(collections.Counter(items[~np.isnan(items)]
                                           ).most_common()) for items in self.labels.T]
        return dict(zip(self.pathologies, counts))

    def __repr__(self) -> str:
        """Returns the name and a description of the dataset such as:

        .. code-block:: python

            CheX_Dataset num_samples=191010 views=['PA', 'AP']

        If in a jupyter notebook it will also print the counts of the
        pathology counts returned by .totals()

        .. code-block:: python

            {'Atelectasis': {0.0: 17621, 1.0: 29718},
             'Cardiomegaly': {0.0: 22645, 1.0: 23384},
             'Consolidation': {0.0: 30463, 1.0: 12982},
             ...}

        """
        def in_notebook():
            try:
                from IPython import get_ipython
                if 'IPKernelApp' not in get_ipython().config:  # pragma: no cover
                    return False
            except ImportError:
                return False
            except AttributeError:
                return False
            return True

        if in_notebook():
            pprint.pprint(self.totals())
        return self.string()

    def check_paths_exist(self):
        if not os.path.isdir(self.imgpath):
            raise Exception("imgpath must be a directory")
        if not os.path.isfile(self.csvpath):
            raise Exception("csvpath must be a file")

    def limit_to_selected_views(self, views):
        """This function is called by subclasses to filter the
        images by view based on the values in .csv['view']
        """
        if type(views) is not list:
            views = [views]
        if '*' in views:
            # if you have the wildcard, the rest are irrelevant
            views = ["*"]
        self.views = views

        # missing data is unknown
        # self.csv.view.fillna("UNKNOWN", inplace=True)
        self.csv["view"] = self.csv["view"].fillna("UNKNOWN")

        if "*" not in views:
            self.csv = self.csv[self.csv["view"].isin(
                self.views)]  # Select the view


# class MedCLIP_Img_Dataset(BaseDataset):
#     '''
#     For MedCLIP, we need to sample sentences from the reports.
#     This class defines several useful functions for MedCLIP pretraining.

#     ** Note that you need to predefine pathologies before running the `__init__` function.
#     '''
#     def __init__(self,
#                  bert_type,
#                  imgpath,
#                  csvpath=None,
#                  views=["PA", "AP"],
#                  seed=0,
#                  sentence_label_csvpath=None,
#                  sample_text_prompt=False,
#                  mapping=None):
#         super().__init__()
#         np.random.seed(seed)  # Reset the seed so all runs are the same.

#         self.imgpath = imgpath
#         if csvpath:
#             self.csvpath = csvpath
#             self.csv = pd.read_csv(self.csvpath)
#         self.views = views
#         self.sample_text_prompt = sample_text_prompt
#         self.mapping = mapping
#         self.tokenizer = AutoTokenizer.from_pretrained(bert_type, trust_remote_code=True)

#         # sentence label
#         if self.sample_text_prompt:
#             self.sentence_label_csvpath = sentence_label_csvpath
#             self.sentence_label = pd.read_csv(self.sentence_label_csvpath)
#             # convert csv into a dictionary
#             self._preprocess_sentence_label()
#             self._build_prompt_sentence()


def _process_medclip_labels(labels):
    labels[labels == -1] = -2
    labels[labels == 0] = -1
    labels[labels == -2] = 0
    return labels


def sample_sent_prompts(sample, pathologies, sentence_label, prompt_sentence_label, num_reports=2):
    '''
    Sample sentences from dictionary.
    '''
    # do prompt sampling
    if (sample["lab"] == -1).all():  # no label available, use no finding
        candid_sents = sentence_label[(sentence_label['No Finding'] > 0) & (
            sentence_label[pathologies] == 0).all(axis=1)]["report"].values
        report = np.random.choice(
            candid_sents, num_reports, replace=True).tolist()
    else:
        # get prompt sentence x * 0 = 0, 1 * -1 = -1, 1 * 1 = 1, -1 * -1 = 1
        bool_sent_label = prompt_sentence_label[pathologies] * sample["lab"]
        sorted_indices = np.argsort(
            bool_sent_label.sum(axis=1).values)[::-1][:10]
        sents = prompt_sentence_label.iloc[sorted_indices]
        if len(sents) == 0:  # only no finding
            candid_sents = prompt_sentence_label[prompt_sentence_label['No Finding']
                                                 == 1]["report"].values
            report = np.random.choice(
                candid_sents, num_reports, replace=True).tolist()
        else:
            # random sample
            candid_sents = sents["report"].values
            report = np.random.choice(
                candid_sents, num_reports, replace=True).tolist()

    return report


class MergeDataset(BaseDataset):
    """The class `MergeDataset` can be used to merge multiple datasets
    together into a single dataset. This class takes in a list of dataset
    objects and assembles the datasets in order. This class will correctly
    maintain the `.labels`, `.csv`, and `.pathologies` fields and offer
    pretty printing.

    .. code-block:: python

        dmerge = xrv.datasets.MergeDataset([dataset1, dataset2, ...])
        # Output:
        MergeDataset num_samples=261583
        - 0 PC_Dataset num_samples=94825 views=['PA', 'AP']
        - 1 RSNA_Pneumonia_Dataset num_samples=26684 views=['PA', 'AP']
        - 2 NIH_Dataset num_samples=112120 views=['PA', 'AP']
        - 3 SIIM_Pneumothorax_Dataset num_samples=12954
        - 4 VinBrain_Dataset num_samples=15000 views=['PA', 'AP']
    """

    def __init__(self, datasets, seed=0, label_concat=False, pathologies=[]):
        super(MergeDataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.datasets = datasets

        # We need formalize all pathologies we want to detect.
        # This need to be customized later.
        self.custom_mapping = {
            "Other lesion": "Lung Lesion",
            "Pleural effusion": "Pleural Effusion",
            "Effusion": "Pleural Effusion",
            "Pleural_Thickening": "Pleural Other",
            "Fibrosis": "Pulmonary fibrosis",
            "Nodule/Mass": "Nodule",
            "Pleural thickening": "Pleural Other",
        }

        self.length = 0
        # we need to align pathologies manually
        # self.pathologies = datasets[0].pathologies
        self.which_dataset = np.zeros(0)
        self.offset = np.zeros(0)
        currentoffset = 0

        # collect all datasets
        self.pathologies = []
        for dataset in datasets:
            dataset_pathologies = []
            for pathology in dataset.pathologies:
                if pathology in self.custom_mapping:
                    pathology = self.custom_mapping[pathology]
                dataset_pathologies.append(pathology)
            dataset.pathologies = dataset_pathologies
            self.pathologies += dataset.pathologies

        if len(pathologies) > 0:
            self.pathologies = pathologies

        self.pathologies = sorted(list(set(self.pathologies)))

        total_len = sum([len(x) for x in datasets])
        self.labels = np.zeros([total_len, len(self.pathologies)])
        self.pathology_indices = []
        self.report_labels = np.zeros([total_len, len(self.pathologies)])
        for i, dataset in enumerate(datasets):
            self.which_dataset = np.concatenate(
                [self.which_dataset, np.zeros(len(dataset)) + i])
            self.length += len(dataset)
            self.offset = np.concatenate(
                [self.offset, np.zeros(len(dataset)) + currentoffset])
            currentoffset += len(dataset)

            indices = []
            for pathology in dataset.pathologies:
                if pathology in self.custom_mapping:
                    pathology = self.custom_mapping[pathology]
                p_index = self.pathologies.index(pathology)
                indices.append(p_index)
                # self.labels[currentoffset - len(dataset):currentoffset, p_index] = \
                #     dataset.labels[:, dataset.pathologies.index(pathology)]
            self.pathology_indices.append(indices)

        self.which_dataset = self.which_dataset.astype(int)

    def __setattr__(self, name, value):
        if hasattr(self, 'labels'):
            # check only if have finished init, otherwise __init__ breaks
            if name in ['transform', 'data_aug', 'labels', 'pathologies', 'targets']:
                raise NotImplementedError(
                    f'Cannot set {name} on a merged dataset. Set the transforms directly on the dataset object. If it was to be set via this merged dataset it would have to modify the internal dataset which could have unexpected side effects')

        object.__setattr__(self, name, value)

    def string(self):
        s = self.__class__.__name__ + " num_samples={}\n".format(len(self))
        for i, d in enumerate(self.datasets):
            if i < len(self.datasets) - 1:
                s += "├{} ".format(i) + \
                    d.string().replace("\n", "\n|  ") + "\n"
            else:
                s += "└{} ".format(i) + \
                    d.string().replace("\n", "\n   ") + "\n"
        return s

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        dataset_idx = int(self.which_dataset[idx])
        item = self.datasets[dataset_idx][idx - int(self.offset[idx])]
        item["dataset_idx"] = dataset_idx
        item["source"] = self.which_dataset[idx]
        cur_lab = item["lab"].copy()
        # -1 means uncertain
        item["lab"] = -1 * np.ones(len(self.pathologies))
        item["lab"][self.pathology_indices[dataset_idx]] = cur_lab

        if "report_lab" in item:
            report_labels = np.zeros(len(self.pathologies))
            report_labels[self.pathology_indices[dataset_idx]
                          ] = item["report_lab"]
            item["report_lab"] = report_labels

        return item


# alias so it is backwards compatible
Merge_Dataset = MergeDataset


class FilterDataset(BaseDataset):
    def __init__(self, dataset, labels=None):
        super(FilterDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        self.idxs = []
        if labels:
            for label in labels:
                print("filtering for ", label)

                self.idxs += list(
                    np.where(dataset.labels[:, list(dataset.pathologies).index(label)] == 1)[0])

        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]

    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n", "\n  ")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]


class SubsetDataset(BaseDataset):
    """When you only want a subset of a dataset the `SubsetDataset` class can
    be used. A list of indexes can be passed in and only those indexes will
    be present in the new dataset. This class will correctly maintain the
    `.labels`, `.csv`, and `.pathologies` fields and offer pretty printing.

    .. code-block:: python

        dsubset = xrv.datasets.SubsetDataset(dataset, [0, 5, 60])
        # Output:
        SubsetDataset num_samples=3
        of PC_Dataset num_samples=94825 views=['PA', 'AP']

    For example this class can be used to create a dataset of only female
    patients by selecting that column of the csv file and using np.where to
    convert this boolean vector into a list of indexes.

    .. code-block:: python

        idxs = np.where(dataset.csv.PatientSex_DICOM=="F")[0]
        dsubset = xrv.datasets.SubsetDataset(dataset, idxs)
        # Output:
        SubsetDataset num_samples=48308
        - of PC_Dataset num_samples=94825 views=['PA', 'AP'] data_aug=None

    """

    def __init__(self, dataset, idxs=None):
        super(SubsetDataset, self).__init__()
        self.dataset = dataset
        self.pathologies = dataset.pathologies

        self.idxs = idxs
        self.labels = self.dataset.labels[self.idxs]
        self.csv = self.dataset.csv.iloc[self.idxs]
        self.csv = self.csv.reset_index(drop=True)

        if hasattr(self.dataset, 'which_dataset'):
            # keep information about the source dataset from a merged dataset
            self.which_dataset = self.dataset.which_dataset[self.idxs]

    def __setattr__(self, name, value):
        if hasattr(self, 'labels'):
            # check only if have finished init, otherwise __init__ breaks
            if name in ['transform', 'data_aug', 'labels', 'pathologies', 'targets']:
                raise NotImplementedError(
                    f'Cannot set {name} on a subset dataset. Set the transforms directly on the dataset object. If it was to be set via this subset dataset it would have to modify the internal dataset which could have unexpected side effects')

        object.__setattr__(self, name, value)

    def string(self):
        return self.__class__.__name__ + " num_samples={}\n".format(len(self)) + "└ of " + self.dataset.string().replace("\n", "\n  ")

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        return self.dataset[self.idxs[idx]]


def detect_bbox_from_masks(seg_mask, mask_labels, bbox_shift=10):
    new_masks, bboxes, bbox_labels = [], [], []
    for i in range(seg_mask.shape[0]):
        gt2D = seg_mask[i].cpu().numpy()
        cur_label = mask_labels[i]
        if gt2D.sum() < 50:
            continue
            # # in this case, it will be filtered
            # bbox = torch.tensor([0, 0, 0, 0])
            # new_masks.append(gt2D)
            # bboxes.append(bbox)
            # bbox_labels.append(cur_label)

        contours, hierarchy = cv2.findContours(gt2D.astype(
            np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x_min, y_min = np.min(contour, axis=0)[0]
            x_max, y_max = np.max(contour, axis=0)[0]
            # add perturbation to bounding box coordinates
            H, W = gt2D.shape
            x_min = max(0, x_min - random.randint(0, bbox_shift))
            x_max = min(W, x_max + random.randint(0, bbox_shift))
            y_min = max(0, y_min - random.randint(0, bbox_shift))
            y_max = min(H, y_max + random.randint(0, bbox_shift))

            # create new mask
            new_mask = gt2D.copy()
            new_mask[:y_min] = 0
            new_mask[y_max:] = 0
            new_mask[:, :x_min] = 0
            new_mask[:, x_max:] = 0

            if new_mask.sum() < 50:
                continue

            bbox = torch.tensor([x_min, y_min, x_max, y_max])
            bboxes.append(bbox)
            bbox_labels.append(cur_label)
            new_masks.append(new_mask)

    bboxes = torch.stack(bboxes, dim=0)
    bbox_labels = torch.tensor(np.array(bbox_labels))
    new_masks = torch.tensor(np.array(new_masks))

    return new_masks, bboxes, bbox_labels


class NIH_Dataset(BaseDataset):
    """NIH ChestX-ray14 dataset

    ChestX-ray dataset comprises 112,120 frontal-view X-ray images of 30,
    805 unique patients with the text-mined fourteen disease image labels (
    where each image can have multi-labels), mined from the associated
    radiological reports using natural language processing. Fourteen common
    thoracic pathologies include Atelectasis, Consolidation, Infiltration,
    Pneumothorax, Edema, Emphysema, Fibrosis, Effusion, Pneumonia,
    Pleural_thickening, Cardiomegaly, Nodule, Mass and Hernia, which is an
    extension of the 8 common disease patterns listed in our CVPR2017 paper.
    Note that original radiology reports (associated with these chest x-ray
    studies) are not meant to be publicly shared for many reasons. The
    text-mined disease labels are expected to have accuracy >90%.Please find
    more details and benchmark performance of trained models based on 14
    disease labels in our arxiv paper: https://arxiv.org/abs/1705.02315

    Dataset release website:
    https://www.nih.gov/news-events/news-releases/nih-clinical-center-provides-one-largest-publicly-available-chest-x-ray-datasets-scientific-community

    Download full size images here:
    https://academictorrents.com/details/557481faacd824c83fbf57dcf7b6da9383b3235a

    Download resized (224x224) images here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0

    ** The logic of this dataset is a bit complex. Maybe need to refined later.
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 unique_patients=False,
                 medclip_format=False,
                 pathologies=None,
                 sentence_label=None,
                 data_pct=1.,
                 prompt_sentence_label=None,
                 sent_info_dict=None
                 ):
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        super().__init__()
        # original diseases in NIH dataset
        if not medclip_format:
            default_pathologies = ["Atelectasis", "Consolidation", "Infiltration",
                                   "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
                                   "Effusion", "Pneumonia", "Pleural_Thickening",
                                   "Cardiomegaly", "Nodule", "Mass", "Hernia"]
        else:
            # Note: since chexbert doesn't have Infiltration, Emphysema, Fibrosis and Hernia,
            # we don't consider them in the MedCLIP training.
            default_pathologies = ["No Finding",
                                   "Enlarged Cardiomediastinum",
                                   "Cardiomegaly",
                                   "Lung Opacity",
                                   "Lung Lesion",
                                   "Edema",
                                   "Consolidation",
                                   "Pneumonia",
                                   "Atelectasis",
                                   "Pneumothorax",
                                   "Pleural Effusion",
                                   "Pleural Other",
                                   "Fracture",
                                   "Support Devices"]

        mapping = {
            "Pleural Effusion": ["Effusion"],
            "Pleural Other": ["Pleural_Thickening"],
            "Lung Lesion": ["Nodule", "Mass",],
            "Lung Opacity": ["Nodule", "Mass", "Infiltration", "Fibrosis"],
            "Enlarged Cardiomediastinum": ["Hernia"],
            "Consolidation": ["air bronchogram"]
        }

        if pathologies is None:
            self.pathologies = default_pathologies
        else:
            self.pathologies = pathologies
        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views
        self.medclip_format = medclip_format
        self.transform = transform
        self.data_aug = data_aug
        self.sentence_label = sentence_label
        self.prompt_sentence_label = prompt_sentence_label
        self.sent_info_dict = sent_info_dict

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()

        self.csv = self.csv.reset_index()

        # sample a subset of dataset
        if data_pct < 1.0:
            self.csv = self.csv.sample(
                frac=data_pct, random_state=seed).reset_index(drop=True)

        ####### pathology masks ########
        # Get our classes.
        labels = []
        # all matching are done in lower case.
        self.csv["Finding Labels"] = self.csv["Finding Labels"].apply(
            lambda x: x.lower())
        for pathology in self.pathologies:
            mask = self.csv["Finding Labels"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    mask |= self.csv["Finding Labels"].str.contains(
                        syn.lower())
            labels.append(mask.values)

        # FIXME: check labels
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # sanity check
        normal_idx = self.pathologies.index("No Finding")
        abnormal_idx = [i for i in range(
            len(self.pathologies)) if i != normal_idx]
        abnormal_labels = self.labels[:, abnormal_idx]
        self.labels[:, normal_idx] = 1 - np.any(abnormal_labels != 0, axis=1)

        # create uncertain masks
        sum_samples_per_pathology = np.sum(self.labels, axis=0)
        uncertain_mask = sum_samples_per_pathology == 0
        self.labels[:, uncertain_mask] = -1

        # print(pd.DataFrame(self.labels, columns=self.pathologies).describe().T)

        if self.medclip_format:
            self.labels = _process_medclip_labels(self.labels)

        # patientid
        self.csv["patientid"] = self.csv["Patient ID"].astype(str)
        # age
        self.csv['age_years'] = self.csv['Patient Age'] * 1.0
        # sex
        self.csv['sex_male'] = self.csv['Patient Gender'] == 'M'
        self.csv['sex_female'] = self.csv['Patient Gender'] == 'F'

        self.num_patients = len(self.csv["patientid"].unique())

    def string(self):
        return self.__class__.__name__ + " num_patients={} num_samples={} views={} data_aug={}".format(
            self.num_patients, len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        imgid = self.csv['Image Index'].iloc[idx]
        sample["imgid"] = imgid
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = normalize(img, maxval=255, reshape=True)
        sample["img"] = img
        if isinstance(self.transform, DataAugmentationDINO):
            sample = apply_dino_transforms(sample, self.transform)
        elif isinstance(self.transform, A.Compose):
            sample = apply_transforms(sample, self.transform)
        else:
            raise RuntimeError(f"{self.transform.__class__} doesn't belong to any"
                               " transform class (DataAugmentationDINO, A.Compose). Please check it.")

        if self.medclip_format:
            sampled_report = sample_sent_prompts(
                sample, self.pathologies, self.sentence_label, self.prompt_sentence_label,
                num_reports=2)
            sample["report"] = sampled_report
            sample["report_lab"] = self.sent_info_dict[sampled_report[0]]

        return sample


class NIH_Localization_Dataset(BaseDataset):
    """ The same NIH ChestX-ray14 dataset with bounding box annotations
    Only consider the 880 images with bounding box annotations.
    """

    def __init__(self,
                 imgpath,
                 bbox_list_path=USE_INCLUDED_FILE,
                 mask_path=USE_INCLUDED_FILE,
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 unique_patients=False,
                 pathology_masks=False,
                 bbox_only=False,
                 ):
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        super().__init__()

        self.imgpath = imgpath
        self.views = views
        self.mask_path = mask_path
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        self.bbox_only = bbox_only

        ####### pathology masks ########
        # load nih pathology masks
        if bbox_list_path == USE_INCLUDED_FILE:
            self.bbox_list_path = os.path.join(
                datapath, "BBox_List_2017.csv.gz")
        else:
            self.bbox_list_path = bbox_list_path
        self.pathology_maskscsv = pd.read_csv(
            self.bbox_list_path,
            names=["Image Index", "Finding Label",
                   "x", "y", "w", "h", "_1", "_2", "_3"],
            skiprows=1
        )

        # change label name to match
        self.pathology_maskscsv.loc[self.pathology_maskscsv["Finding Label"]
                                    == "Infiltrate", "Finding Label"] = "Infiltration"
        self.box_csv = pd.DataFrame(self.pathology_maskscsv.groupby("Image Index")[
                                    "Finding Label"].apply(lambda x: "|".join(np.unique(x))))
        self.box_csv["Image Index"] = self.box_csv.index
        self.box_csv["patientid"] = self.box_csv["Image Index"].apply(
            lambda x: x.split("_")[0])

        ####### pathology masks ########
        # Get our classes.
        self.labels = []
        # since not each kind of pathology has dense annotations
        self.box_csv["Finding Label"] = self.box_csv["Finding Label"].apply(
            lambda x: x.lower())
        self.pathologies = self.pathology_maskscsv["Finding Label"].unique(
        ).tolist()
        for pathology in self.pathologies:
            mask = self.box_csv["Finding Label"].str.contains(
                pathology.lower())
            self.labels.append(mask.values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)
        self.num_patients = len(self.box_csv["patientid"].unique())

    def string(self):
        return self.__class__.__name__ + " num_patients={} num_samples={} views={} data_aug={}".format(
            self.num_patients, len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.box_csv['Image Index'].iloc[idx]
        sample["imgid"] = imgid

        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = normalize(img, maxval=255, reshape=True)
        H, W, _ = img.shape
        sample["img"] = img
        sample["img_size"] = [H, W]
        sub_df = self.pathology_maskscsv[self.pathology_maskscsv["Image Index"] == imgid]
        # maybe there are multiple boxes for a single image?
        bbox = sub_df[["x", "y", "w", "h"]].values
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]
        bbox = torch.from_numpy(bbox)
        box_labels = torch.from_numpy(
            sample["lab"]).reshape(-1, len(self.pathologies))

        if self.pathology_masks:
            mask_labels = torch.where(box_labels == 1)[1].reshape(-1)
            unique_labels = torch.unique(mask_labels)
            gt_masks = []
            for m_label in unique_labels:
                # In some case, there are multiple boxes for the same label
                cur_indices = torch.where(m_label == mask_labels)[0]
                gt_mask = box2mask(bbox[cur_indices], H, W)
                gt_masks.append(gt_mask)
            gt_masks = torch.cat(gt_masks, dim=0)
            sample["mask"] = gt_masks.numpy().transpose(1, 2, 0)
            # imgid = os.path.splitext(imgid)[0]
            # mask_file = os.path.join(self.mask_path, imgid + ".pkl")
            # assert os.path.exists(mask_file), f"{mask_file} doesn't exist!"
            # with open(mask_file, 'rb') as f:
            #     mask_dict = pickle.load(f)
            #     # img: (1024, 1024, 3)
            #     # mask: (1024, 1024, N)
            #     # mask_label: (N,)
            # sample["img"] = mask_dict["img"]
            # # H, W, N = mask_dict["mask"].shape
            # # padded_mask = np.zeros((H, W, len(self.pathologies)))
            # # padded_mask[:, :, mask_dict["mask_label"]] = mask_dict["mask"]
            # # sample["mask"] = padded_mask
            # sample["mask"] = mask_dict["mask"]
        else:
            sample["bbox"] = bbox
            sample["bbox_label"] = np.where(self.labels[idx] == 1)[
                0].reshape(-1, 1)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class RSNA_Pneumonia_Dataset(BaseDataset):
    """RSNA Pneumonia Detection Challenge

    Citation:

    Augmenting the National Institutes of Health Chest Radiograph Dataset
    with Expert Annotations of Possible Pneumonia. Shih, George, Wu,
    Carol C., Halabi, Safwan S., Kohli, Marc D., Prevedello, Luciano M.,
    Cook, Tessa S., Sharma, Arjun, Amorosa, Judith K., Arteaga, Veronica,
    Galperin-Aizenberg, Maya, Gill, Ritu R., Godoy, Myrna C.B., Hobbs,
    Stephen, Jeudy, Jean, Laroia, Archana, Shah, Palmi N., Vummidi, Dharshan,
    Yaddanapudi, Kavitha, and Stein, Anouk. Radiology: Artificial
    Intelligence, 1 2019. doi: 10.1148/ryai.2019180041.

    More info: https://www.rsna.org/en/education/ai-resources-and-training/ai-image-challenge/RSNA-Pneumonia-Detection-Challenge-2018

    Challenge site:
    https://www.kaggle.com/c/rsna-pneumonia-detection-challenge

    JPG files stored here:
    https://academictorrents.com/details/95588a735c9ae4d123f3ca408e56570409bcf2a9
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 mask_path=USE_INCLUDED_FILE,
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 pathology_masks=False,
                 extension=".dcm",
                 bbox_only=False
                 ):

        super(RSNA_Pneumonia_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.mask_path = mask_path
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        self.bbox_only = bbox_only

        # self.pathologies = ["Pneumonia", "Lung Opacity"]
        self.pathologies = ["Pneumonia"]

        self.pathologies = sorted(self.pathologies)

        self.extension = extension
        self.use_pydicom = (extension == ".dcm")
        self.views = views

        # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(
                datapath, "kaggle_stage_2_train_labels.csv.zip")
        else:
            self.csvpath = csvpath
        self.raw_csv = pd.read_csv(self.csvpath, nrows=nrows)

        # The labels have multiple instances for each mask
        # So we just need one to get the target label
        self.csv = self.raw_csv.groupby("patientId").first()

        # Remove images with view position other than specified
        if "ViewPosition" in self.csv:
            self.csv["view"] = self.csv['ViewPosition']
            self.limit_to_selected_views(views)
        self.csv = self.csv.reset_index()

        # set if we have masks
        self.csv["has_masks"] = ~np.isnan(self.csv["x"])

        if self.bbox_only or self.pathology_masks:
            self.csv = self.csv[self.csv["has_masks"]]
            # Get our classes.
            labels = [self.csv["Target"].values]
        else:
            # Get our classes.
            labels = [self.csv["Target"].values]
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # patientid
        if "patientId" in self.csv:
            self.csv["patientid"] = self.csv["patientId"].astype(str)

        self.num_patients = len(self.csv["patientid"].unique())

    def string(self):
        return self.__class__.__name__ + " num_patients={} num_samples={} views={} data_aug={}".format(
            self.num_patients, len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        imgid = self.csv['patientId'].iloc[idx]
        sample["imgid"] = imgid

        # ** We assume return mask will have bbox by default.
        img_path = os.path.join(self.imgpath, imgid + self.extension)
        if self.use_pydicom:
            try:
                import pydicom
            except ImportError as e:
                raise Exception(
                    "Please install pydicom to work with this dataset")

            img = pydicom.filereader.dcmread(img_path).pixel_array
        else:
            img = imread(img_path, cv2.IMREAD_GRAYSCALE)

        img = normalize(img, maxval=255, reshape=True)
        sample["img"] = img
        H, W, _ = img.shape
        sample["img_size"] = [H, W]
        bbox = self.raw_csv[self.raw_csv["patientId"] == imgid][[
            "x", "y", "width", "height"]].values.reshape(-1, 4)
        bbox[:, 2] = bbox[:, 0] + bbox[:, 2]
        bbox[:, 3] = bbox[:, 1] + bbox[:, 3]

        if self.bbox_only:
            sample["bbox"] = bbox
            sample["bbox_label"] = np.zeros((len(bbox), 1))
        elif self.pathology_masks:
            bbox = torch.from_numpy(bbox)
            mask = box2mask(bbox, H, W)
            sample["mask"] = mask.numpy().transpose(1, 2, 0)
        else:
            raise NotImplementedError

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class NIH_Google_Dataset(BaseDataset):
    """A relabelling of a subset of images from the NIH dataset.  The data
    tables should be applied against an NIH download.  A test and validation
    split are provided in the original.  They are combined here, but one or
    the other can be used by providing the original csv to the csvpath
    argument.

    Citation:

    Chest Radiograph Interpretation with Deep Learning Models: Assessment
    with Radiologist-adjudicated Reference Standards and Population-adjusted
    Evaluation Anna Majkowska, Sid Mittal, David F. Steiner, Joshua J.
    Reicher, Scott Mayer McKinney, Gavin E. Duggan, Krish Eswaran, Po-Hsuan
    Cameron Chen, Yun Liu, Sreenivasa Raju Kalidindi, Alexander Ding, Greg S.
    Corrado, Daniel Tse, and Shravya Shetty. Radiology 2020

    https://pubs.rsna.org/doi/10.1148/radiol.2019191293

    NIH data can be downloaded here:
    https://academictorrents.com/details/e615d3aebce373f1dc8bd9d11064da55bdadede0
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 views=["PA"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=True
                 ):

        super(NIH_Google_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        self.pathologies = ["Fracture", "Pneumothorax", "Airspace opacity",
                            "Nodule or mass"]

        self.pathologies = sorted(self.pathologies)

        # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(
                datapath, "google2019_nih-chest-xray-labels.csv.gz")
        else:
            self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, nrows=nrows)

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['View Position']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first().reset_index()

        # Get our classes.
        self.labels = []
        for pathology in self.pathologies:
            mask = self.csv[pathology] == "YES"

            self.labels.append(mask.values)

        self.labels = np.asarray(self.labels).T
        self.labels = self.labels.astype(np.float32)

        # rename pathologies
        self.pathologies = np.char.replace(
            self.pathologies, "Airspace opacity", "Lung Opacity")
        self.pathologies = np.char.replace(
            self.pathologies, "Nodule or mass", "Nodule/Mass")
        self.pathologies = list(self.pathologies)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Image Index'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class PC_Dataset(BaseDataset):
    """PadChest dataset from the Hospital San Juan de Alicante - University of
    Alicante

    Note that images with null labels (as opposed to normal), and images that
    cannot be properly loaded (listed as 'missing' in the code) are excluded,
    which makes the total number of available images slightly less than the
    total number of image files.

    Citation:

    PadChest: A large chest x-ray image dataset with multi-label annotated
    reports. Aurelia Bustos, Antonio Pertusa, Jose-Maria Salinas, and Maria
    de la Iglesia-Vayá. arXiv preprint, 2019. https://arxiv.org/abs/1901.07441

    Dataset website:
    http://bimcv.cipf.es/bimcv-projects/padchest/

    Download full size images here:
    https://academictorrents.com/details/dec12db21d57e158f78621f06dcbe78248d14850

    Download resized (224x224) images here (recropped):
    https://academictorrents.com/details/96ebb4f92b85929eadfb16761f310a6d04105797
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 bert_type="",
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=False,
                 medclip_format=False,
                 sent_info_dict=dict(),
                 data_pct=1.,
                 pathologies=None,
                 ):

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        super().__init__()
        default_pathologies = ["Normal",
                               "Enlarged Cardiomediastinum",
                               "Cardiomegaly",
                               "Lung Opacity",
                               "Lung Lesion",
                               "Edema",
                               "Consolidation",
                               "Pneumonia",
                               "Atelectasis",
                               "Pneumothorax",
                               "Pleural Effusion",
                               "Pleural Other",
                               "Fracture",
                               "Support Devices"]
        if pathologies is None:
            self.pathologies = sorted(default_pathologies)
        else:
            self.pathologies = sorted(pathologies)

        # self.pathologies = ["Atelectasis", "Consolidation", "Infiltration",
        #                     "Pneumothorax", "Edema", "Emphysema", "Fibrosis",
        #                     "Effusion", "Pneumonia", "Pleural_Thickening",
        #                     "Cardiomegaly", "Nodule", "Mass", "Hernia", "Fracture",
        #                     "Granuloma", "Flattened Diaphragm", "Bronchiectasis",
        #                     "Aortic Elongation", "Scoliosis",
        #                     "Hilar Enlargement", "Tuberculosis",
        #                     "Air Trapping", "Costophrenic Angle Blunting", "Aortic Atheromatosis",
        #                     "Hemidiaphragm Elevation",
        #                     "Support Devices", "Tube'"]  # the Tube' is intentional

        mapping = dict()
        mapping["No Finding"] = ["Normal"]
        mapping["Lung Opacity"] = ["Nodule", "Mass", "Infiltrates"]
        mapping["Lung Lesion"] = ["Nodule", "Mass"]
        mapping["Enlarged Cardiomediastinum"] = ["Hiatal Hernia"]
        mapping["Pleural Effusion"] = ["Effusion"]
        mapping["Consolidation"] = ["air bronchogram", "Infiltrates"]
        mapping["Pneumothorax"] = ["hyperinflated lung"]
        mapping["Support Devices"] = ["device",
                                      "pacemaker"]
        mapping["Pleural Other"] = ["pleural thickening"]

        self.imgpath = imgpath
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views
        self.transform = transform
        self.data_aug = data_aug
        self.flat_dir = flat_dir
        self.check_paths_exist()
        self.medclip_format = medclip_format
        self.sent_info_dict = sent_info_dict

        # Standardize view names
        self.csv.loc[self.csv["Projection"].isin(
            ["AP_horizontal"]), "Projection"] = "AP Supine"

        self.csv["view"] = self.csv['Projection']
        self.limit_to_selected_views(views)

        # Remove null stuff
        self.csv = self.csv[~self.csv["Labels"].isnull()]

        # Remove missing files
        missing = ["216840111366964012819207061112010307142602253_04-014-084.png",
                   "216840111366964012989926673512011074122523403_00-163-058.png",
                   "216840111366964012959786098432011033083840143_00-176-115.png",
                   "216840111366964012558082906712009327122220177_00-102-064.png",
                   "216840111366964012339356563862009072111404053_00-043-192.png",
                   "216840111366964013076187734852011291090445391_00-196-188.png",
                   "216840111366964012373310883942009117084022290_00-064-025.png",
                   "216840111366964012283393834152009033102258826_00-059-087.png",
                   "216840111366964012373310883942009170084120009_00-097-074.png",
                   "216840111366964012819207061112010315104455352_04-024-184.png",
                   "216840111366964012819207061112010306085429121_04-020-102.png",
                   # broken PNG file (chunk b'\x00\x00\x00\x00')
                   "216840111366964012989926673512011083134050913_00-168-009.png",
                   # "OSError: image file is truncated"
                   "216840111366964012373310883942009152114636712_00-102-045.png",
                   # "OSError: image file is truncated"
                   "216840111366964012819207061112010281134410801_00-129-131.png",
                   # "OSError: image file is truncated"
                   "216840111366964012487858717522009280135853083_00-075-001.png",
                   # broken PNG file (chunk b'\x00\x00\x00\x00')
                   "216840111366964012989926673512011151082430686_00-157-045.png",
                   # "OSError: image file is truncated"
                   "216840111366964013686042548532013208193054515_02-026-007.png",
                   # "OSError: image file is truncated"
                   "216840111366964013590140476722013058110301622_02-056-111.png",
                   # "OSError: image file is truncated"
                   "216840111366964013590140476722013043111952381_02-065-198.png",
                   # "OSError: image file is truncated"
                   "216840111366964013829543166512013353113303615_02-092-190.png",
                   # "OSError: image file is truncated"
                   "216840111366964013962490064942014134093945580_01-178-104.png",
                   ]
        self.csv = self.csv[~self.csv["ImageID"].isin(missing)]

        if unique_patients:
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # Filter out age < 10 (paper published 2019)
        self.csv = self.csv[(2019 - self.csv.PatientBirth > 10)]

        # report
        # self.reports = self.csv["Report_Eng"].values
        self.csv["Report_Eng"] = self.csv["Report_Eng"].apply(
            self._split_report_into_segment)
        self.medclip_format = medclip_format

        # filter rows with empty reports
        self.csv = self.csv[self.csv["Report_Eng"].apply(lambda x: len(x) > 0)]

        # sample a subset of dataset
        if data_pct < 1.0:
            self.csv = self.csv.sample(
                frac=data_pct, random_state=seed).reset_index(drop=True)

        # Get our classes.
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["Labels"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    # print("mapping", syn)
                    mask |= self.csv["Labels"].str.contains(syn.lower())
            labels.append(mask.values)

        # FIXME: check labels
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # sanity check
        normal_idx = self.pathologies.index("No Finding")
        abnormal_idx = [i for i in range(
            len(self.pathologies)) if i != normal_idx]
        abnormal_labels = self.labels[:, abnormal_idx]
        self.labels[:, normal_idx] = 1 - np.any(abnormal_labels != 0, axis=1)

        # create uncertain masks
        sum_samples_per_pathology = np.sum(self.labels, axis=0)
        uncertain_mask = sum_samples_per_pathology == 0
        self.labels[:, uncertain_mask] = -1

        # print(pd.DataFrame(self.labels, columns=self.pathologies).describe().T)

        if medclip_format:
            self.labels = _process_medclip_labels(self.labels)

        # offset_day_int
        dt = pd.to_datetime(self.csv["StudyDate_DICOM"], format="%Y%m%d")
        self.csv["offset_day_int"] = dt.astype(np.int64) // 10**9 // 86400

        # patientid
        self.csv["patientid"] = self.csv["PatientID"].astype(str)

        # age
        self.csv['age_years'] = (2017 - self.csv['PatientBirth'])

        # sex
        self.csv['sex_male'] = self.csv['PatientSex_DICOM'] == 'M'
        self.csv['sex_female'] = self.csv['PatientSex_DICOM'] == 'F'

        self.num_patients = len(self.csv["patientid"].unique())

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n', ' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower())

                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 4:  # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def string(self):
        return self.__class__.__name__ + " num_patients={} num_samples={} views={} data_aug={}".format(
            self.num_patients, len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['ImageID'].iloc[idx]
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        sample["img"] = normalize(img, maxval=65535, reshape=True)
        if isinstance(self.transform, DataAugmentationDINO):
            sample = apply_dino_transforms(sample, self.transform)
        elif isinstance(self.transform, A.Compose):
            sample = apply_transforms(sample, self.transform)
        else:
            raise RuntimeError(f"{self.transform.__class__} doesn't belong to any"
                               " transform class (DataAugmentationDINO, A.Compose). Please check it.")

        if self.medclip_format:
            report = self.csv.iloc[idx]["Report_Eng"]
            report = [sent for sent in report if "compare" not in sent]
            if len(report) == 0:
                # if all sentences belong to xxx compare xxx, then randomly sample one sentence
                report = self.csv.iloc[idx]["Report_Eng"]
            # collect labels of all sentences
            sent_labels = []
            for sent in report:
                sent_labels.append(self.sent_info_dict[sent])
            sent_labels = np.asarray(sent_labels)
            # sample the sentence which has the same label with the image, instead of randomly sampling.
            sent_labels_sim = np.sum(sample["lab"] * sent_labels, axis=1)
            max_vals = np.max(sent_labels_sim)
            indices = np.where(sent_labels_sim == max_vals)[0]
            sent_list = np.random.choice(
                [report[int(x)] for x in indices], 2, replace=True).tolist()
            sample["report"] = sent_list
            sample["report_lab"] = self.sent_info_dict[sent_list[0]]

        return sample


class CheX_Dataset(BaseDataset):
    """CheXpert Dataset

    Citation:

    CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and
    Expert Comparison. Jeremy Irvin *, Pranav Rajpurkar *, Michael Ko,
    Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad
    Haghgoo, Robyn Ball, Katie Shpanskaya, Jayne Seekins, David A. Mong,
    Safwan S. Halabi, Jesse K. Sandberg, Ricky Jones, David B. Larson,
    Curtis P. Langlotz, Bhavik N. Patel, Matthew P. Lungren, Andrew Y. Ng.
    https://arxiv.org/abs/1901.07031

    Dataset website here:
    https://stanfordmlgroup.github.io/competitions/chexpert/

    A small validation set is provided with the data as well, but is so tiny,
    it is not included here.
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 seed=0,
                 unique_patients=False,
                 medclip_format=False,
                 pathologies=None,
                 sentence_label=None,
                 data_pct=1,
                 prompt_sentence_label=None,
                 sent_info_dict=None
                 ):
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        super().__init__()
        default_pathologies = ["No Finding",
                               "Enlarged Cardiomediastinum",
                               "Cardiomegaly",
                               "Lung Opacity",
                               "Lung Lesion",
                               "Edema",
                               "Consolidation",
                               "Pneumonia",
                               "Atelectasis",
                               "Pneumothorax",
                               "Pleural Effusion",
                               "Pleural Other",
                               "Fracture",
                               "Support Devices"]
        if pathologies is None:
            self.pathologies = default_pathologies
        else:
            self.pathologies = pathologies
        self.pathologies = sorted(self.pathologies)
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views
        self.medclip_format = medclip_format
        self.transform = transform
        self.data_aug = data_aug
        self.sentence_label = sentence_label
        self.prompt_sentence_label = prompt_sentence_label
        self.sent_info_dict = sent_info_dict

        self.csv["view"] = self.csv["Frontal/Lateral"]  # Assign view column
        # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
        self.csv.loc[(self.csv["view"] == "Frontal"),
                     "view"] = self.csv["AP/PA"]
        self.csv["view"] = self.csv["view"].replace(
            {'Lateral': "L"})  # Rename Lateral with L

        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(
                pat=r'(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        self.csv.fillna(0, inplace=True)

        # sample a subset of dataset
        if data_pct < 1.0:
            self.csv = self.csv.sample(
                frac=data_pct, random_state=seed).reset_index(drop=True)

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology not in ["No Finding", "Support Devices"]:
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
            else:
                raise ValueError(
                    f"Pathology {pathology} not found in the csv file.")

            labels.append(mask.values)

        # FIXME: check labels
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # sanity check:
        normal_idx = self.pathologies.index("No Finding")
        abnormal_idx = [i for i in range(
            len(self.pathologies)) if i != normal_idx]
        abnormal_labels = self.labels[:, abnormal_idx]
        self.labels[:, normal_idx] = 1 - np.any(abnormal_labels != 0, axis=1)
        # print(pd.DataFrame(self.labels, columns=self.pathologies).describe().T)

        if self.medclip_format:
            self.labels = _process_medclip_labels(self.labels)

        # patientid
        if 'train' in self.csvpath:
            patientid = self.csv.Path.str.split("train/", expand=True)[1]
        elif 'val' in self.csvpath:
            patientid = self.csv.Path.str.split("valid/", expand=True)[1]
        else:
            raise NotImplementedError

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient", "")

        # patientid
        self.csv["patientid"] = patientid

        # age
        # self.csv['age_years'] = self.csv['Age'] * 1.0
        # self.csv['Age'][(self.csv['Age'] == 0)] = None

        # sex
        self.csv['sex_male'] = self.csv['Sex'] == 'Male'
        self.csv['sex_female'] = self.csv['Sex'] == 'Female'

        self.num_patients = len(self.csv["patientid"].unique())

    def string(self):
        return self.__class__.__name__ + " num_patients={} num_samples={} views={} data_aug={}".format(
            self.num_patients, len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Path'].iloc[idx]
        # clean up path in csv so the user can specify the path
        imgid = imgid.replace("CheXpert-v1.0-small/",
                              "").replace("CheXpert-v1.0/", "")
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        sample["img"] = normalize(img, maxval=255, reshape=True)

        if isinstance(self.transform, DataAugmentationDINO):
            sample = apply_dino_transforms(sample, self.transform)
        elif isinstance(self.transform, A.Compose):
            sample = apply_transforms(sample, self.transform)
        else:
            raise RuntimeError(f"{self.transform.__class__} doesn't belong to any"
                               " transform class (DataAugmentationDINO, A.Compose). Please check it.")

        if self.medclip_format:
            sampled_report = sample_sent_prompts(
                sample, self.pathologies, self.sentence_label, self.prompt_sentence_label,
                num_reports=2)
            sample["report"] = sampled_report
            sample["report_lab"] = self.sent_info_dict[sampled_report[0]]

        return sample


class CheXlocalize_dataset(BaseDataset):
    "Raw dataset can be downloaded in https://stanfordaimi.azurewebsites.net/datasets/23c56a0d-15de-405b-87c8-99c30138950c."

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 segpath=USE_INCLUDED_FILE,
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 flat_dir=True,
                 bbox_shift=10,
                 seed=0,
                 unique_patients=False,
                 return_bbox=False
                 ):

        super(CheXlocalize_dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.

        self.pathologies = ["Enlarged Cardiomediastinum",
                            "Cardiomegaly",
                            "Lung Lesion",
                            "Lung Opacity",
                            "Edema",
                            "Consolidation",
                            "Atelectasis",
                            "Pneumothorax",
                            "Pleural Effusion",
                            "Support Devices"]
        self.mapping = {
            "Lung Opacity": "Airspace Opacity"
        }
        self.pathologies = sorted(self.pathologies)
        self.imgpath = imgpath
        self.segpath = segpath
        self.transform = transform
        self.data_aug = data_aug
        self.bbox_shift = bbox_shift
        if csvpath == USE_INCLUDED_FILE:
            raise RuntimeError("Please specify the path of the csv file.")
        else:
            self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views
        self.return_bbox = return_bbox

        if "Frontal/Lateral" in self.csv:
            assert "val" in csvpath
            # val_labels.csv
            # Assign view column
            self.csv["view"] = self.csv["Frontal/Lateral"]
            self.csv.loc[(self.csv["view"] == "Frontal"),
                         "view"] = self.csv["AP/PA"]
            # If Frontal change with the corresponding value in the AP/PA column otherwise remains Lateral
            self.csv["view"] = self.csv["view"].replace(
                {'Lateral': "L"})  # Rename Lateral with L
            self.limit_to_selected_views(views)
        else:
            assert "test" in csvpath
            self.csv["view"] = self.csv.Path.str.split("/", expand=True)[3].str.split(
                "_", expand=True)[1].str.replace(".jpg", "")
            self.csv["view"] = self.csv["view"].replace({'frontal': "PA"})
            self.csv["view"] = self.csv["view"].replace({'lateral': "L"})
            self.limit_to_selected_views(views)

        if unique_patients:
            self.csv["PatientID"] = self.csv["Path"].str.extract(
                pat=r'(patient\d+)')
            self.csv = self.csv.groupby("PatientID").first().reset_index()

        # convert path into the same format in seg_dict
        if "test" in csvpath:
            self.csv["seg_cxr_id"] = self.csv.Path.str.replace(
                "test/", "").str.replace("/", "_").str.replace(".jpg", "")
        else:
            self.csv["seg_cxr_id"] = self.csv.Path.str.replace(
                "CheXpert-v1.0/valid/", "").str.replace("/", "_").str.replace(".jpg", "")

        cxr_ids = [os.path.splitext(x)[0] for x in os.listdir(segpath)]
        valid_seg_ids = set(list(cxr_ids)).intersection(
            set(list(self.csv["seg_cxr_id"])))
        self.csv = self.csv.loc[self.csv["seg_cxr_id"].isin(valid_seg_ids)]
        self.csv.reset_index(inplace=True, drop=True)
        self.seg_cxr_ids = self.csv["seg_cxr_id"].values

        # uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        # self.csv = self.csv.replace(uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology != "Support Devices":
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)
        self.labels[self.labels == -1] = 0

        # patientid
        if 'val' in self.csvpath:
            patientid = self.csv.Path.str.split(
                "CheXpert-v1.0/valid/", expand=True)[1]
        elif 'test' in self.csvpath:
            patientid = self.csv.Path.str.split("test/", expand=True)[1]
        else:
            raise NotImplementedError

        patientid = patientid.str.split("/study", expand=True)[0]
        patientid = patientid.str.replace("patient", "")

        # patientid
        self.csv["patientid"] = patientid
        self.num_patients = len(self.csv["patientid"].unique())

    def string(self):
        return self.__class__.__name__ + " num_patients={} num_samples={} views={} data_aug={}".format(
            self.num_patients, len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['Path'].iloc[idx]
        # clean up path in csv so the user can specify the path
        imgid = imgid.replace("CheXpert-v1.0-small/",
                              "").replace("CheXpert-v1.0/", "")
        imgid = imgid.replace("valid/", "val/")
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = normalize(img, maxval=255, reshape=True)
        sample["img"] = img
        # img: (H, W, 3)
        H, W, _ = img.shape
        sample["img_size"] = [H, W]

        # Load segmentation mask
        cxr_id = self.seg_cxr_ids[idx]
        sample["imgid"] = cxr_id
        with open(os.path.join(self.segpath, cxr_id + ".pkl"), "rb") as f:
            seg_mask_dict = pickle.load(f)

        seg_mask = []
        for i, task in enumerate(self.pathologies):
            if task in self.mapping:
                task = self.mapping[task]
            cur_mask = seg_mask_dict[task]
            if (cur_mask.sum() > 0) and (sample["lab"][i] == 1):
                seg_mask.append(cur_mask)
            else:
                sample["lab"][i] = 0
        seg_mask = np.stack(seg_mask, axis=2)
        assert seg_mask.shape[2] == np.sum(sample["lab"])
        sample["mask"] = seg_mask

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        # create bbox for masks
        if self.return_bbox:
            # FIXME: don't consider multiple connected components
            mask_labels = np.where(sample["lab"] == 1)[0]
            sample["mask"], sample["bbox"], sample["bbox_label"] = detect_bbox_from_masks(
                sample["mask"], mask_labels, bbox_shift=self.bbox_shift)

        return sample


class MIMIC_Dataset(BaseDataset):
    """MIMIC-CXR Dataset
    Note that this dataset is modified with outputing the report and corresponding labels.

    Citation:

    Johnson AE, Pollard TJ, Berkowitz S, Greenbaum NR, Lungren MP, Deng CY,
    Mark RG, Horng S. MIMIC-CXR: A large publicly available database of
    labeled chest radiographs. arXiv preprint arXiv:1901.07042. 2019 Jan 21.

    https://arxiv.org/abs/1901.07042

    Dataset website here:
    https://physionet.org/content/mimic-cxr-jpg/2.0.0/
    """

    def __init__(self,
                 imgpath,
                 csvpath,
                 pathologies=None,
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 unique_patients=False,
                 medclip_format=False,
                 sent_info_dict=dict(),
                 data_pct=1.0,
                 text_type="random"
                 ):

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        super().__init__()
        default_pathologies = ["No Finding",
                               "Enlarged Cardiomediastinum",
                               "Cardiomegaly",
                               "Lung Opacity",
                               "Lung Lesion",
                               "Edema",
                               "Consolidation",
                               "Pneumonia",
                               "Atelectasis",
                               "Pneumothorax",
                               "Pleural Effusion",
                               "Pleural Other",
                               "Fracture",
                               "Support Devices"]
        if pathologies is None:
            self.pathologies = sorted(default_pathologies)
        else:
            self.pathologies = sorted(pathologies)

        self.imgpath = imgpath
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.medclip_format = medclip_format
        self.sent_info_dict = sent_info_dict
        self.text_type = text_type
        assert self.text_type in ["label", "random", "impression", "report"]

        # Keep only the desired view
        self.csv["view"] = self.csv["ViewPosition"]
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("subject_id").first().reset_index()

        # segment report into sentences
        self.csv["report"] = self.csv["report"].apply(
            self._split_report_into_segment)

        # filter rows with empty reports
        self.csv = self.csv[self.csv["report"].apply(lambda x: len(x) > 0)]

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["StudyDate"]

        # sample a subset of dataset
        if data_pct < 1.0:
            self.csv = self.csv.sample(
                frac=data_pct, random_state=seed).reset_index(drop=True)

        # patientid
        self.csv["patientid"] = self.csv["subject_id"].astype(str)
        self.num_patients = len(self.csv["patientid"].unique())

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology not in ["No Finding", "Support Devices"]:
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]
            else:
                raise ValueError(
                    f"Pathology {pathology} not found in the dataset.")
            labels.append(mask.values)

        # FIXME: check labels
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # sanity check:
        normal_idx = self.pathologies.index("No Finding")
        abnormal_idx = [i for i in range(
            len(self.pathologies)) if i != normal_idx]
        abnormal_labels = self.labels[:, abnormal_idx]
        self.labels[:, normal_idx] = 1 - np.any(abnormal_labels != 0, axis=1)
        # print(pd.DataFrame(self.labels, columns=self.pathologies).describe().T)

        if self.medclip_format:
            self.labels = _process_medclip_labels(self.labels)

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n', ' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower())

                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 4:  # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def string(self):
        return self.__class__.__name__ + " num_patients={} num_samples={} views={} data_aug={}".format(
            self.num_patients, len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        '''
        Return: 
        - idx
        - img
        - lab
        - report
        - report_lab
        '''
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        subjectid = str(self.csv.iloc[idx]["subject_id"])
        studyid = str(self.csv.iloc[idx]["study_id"])
        dicom_id = str(self.csv.iloc[idx]["dicom_id"])
        sample["dicom_id"] = dicom_id

        img_path = os.path.join(
            self.imgpath, "p" + subjectid[:2], "p" + subjectid, "s" + studyid, dicom_id + ".jpg")
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        sample["img"] = normalize(img, maxval=255, reshape=True)

        if isinstance(self.transform, DataAugmentationDINO):
            sample = apply_dino_transforms(sample, self.transform)
        elif isinstance(self.transform, A.Compose):
            sample = apply_transforms(sample, self.transform)
        else:
            raise RuntimeError(f"{self.transform.__class__} doesn't belong to any"
                               " transform class (DataAugmentationDINO, A.Compose). Please check it.")

        if self.medclip_format:
            report = self.csv.iloc[idx]["report"]
            # remove comparison sentences
            report = [sent for sent in report if "compare" not in sent]
            if len(report) == 0:
                # if all sentences belong to xxx compare xxx, then randomly sample one sentence
                # TODO: this can be sampled from the sent dict
                report = self.csv.iloc[idx]["report"]

            if self.text_type == "label":
                # collect labels of all sentences
                sent_labels = []
                for sent in report:
                    sent_labels.append(self.sent_info_dict[sent])
                sent_labels = np.asarray(sent_labels)
                # sample the sentence which has the same label with the image, instead of randomly sampling.
                sent_labels_sim = np.sum(sample["lab"] * sent_labels, axis=1)
                max_vals = np.max(sent_labels_sim)
                indices = np.where(sent_labels_sim == max_vals)[0]
                sent_list = np.random.choice(
                    [report[int(x)] for x in indices], 2, replace=True).tolist()
                sample["report"] = sent_list
                sample["report_lab"] = self.sent_info_dict[sent_list[0]]

            elif self.text_type == "random":
                # randomly choose two sentences
                sent_list = np.random.choice(
                    report, 2, replace=True).tolist()
                sample["report"] = sent_list
                sample["report_lab"] = self.sent_info_dict[sent_list[0]]

        return sample


class MS_CXR_Dataset(BaseDataset):
    """ MS-CXR Dataset
    Boecking, B., Usuyama, N., Bannur, S., Coelho de Castro, D., Schwaighofer, A., Hyland, 
    S., Wetscherek, M. T., Naumann, T., Nori, A., Alvarez Valle, J., Poon, H., & Oktay, O. (2022). 
    MS-CXR: Making the Most of Text Semantics to Improve Biomedical Vision-Language Processing (version 0.1). 
    PhysioNet. https://doi.org/10.13026/b90j-vb87.

    Dataset website: 
    https://physionet.org/content/ms-cxr/0.1/

    """

    def __init__(self,
                 imgpath,
                 csvpath,
                 mask_path=None,
                 views=["PA", "AP"],
                 seed=0,
                 transform=None,
                 data_aug=None,
                 pathology_masks=False):

        np.random.seed(seed)
        super().__init__()
        default_pathologies = ["Atelectasis",
                               "Cardiomegaly",
                               "Consolidation",
                               "Edema",
                               "Lung Opacity",
                               "Pleural Effusion",
                               "Pneumonia",
                               "Pneumothorax"]
        self.pathologies = default_pathologies
        self.pathologies = sorted(self.pathologies)

        self.imgpath = imgpath
        self.csvpath = csvpath
        self.mask_path = mask_path
        self.views = views
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks

        self.csv = pd.read_csv(self.csvpath)
        self.csv["category_name"] = self.csv["category_name"].apply(
            lambda x: x.lower())
        # It should be based on both dicom_id and label_text
        self.agg_csv = self.csv.drop_duplicates(
            subset=['dicom_id', 'label_text'])
        self.agg_csv.reset_index(drop=True, inplace=True)

        # get labels for the whole csv
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["category_name"].str.contains(pathology.lower())
            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.dicom_ids = self.agg_csv["dicom_id"].values
        self.label_text = self.agg_csv["label_text"].values

    def __len__(self):
        return len(self.agg_csv)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx

        dicom_id = self.dicom_ids[idx]
        label_text = self.label_text[idx]
        sample["imgid"] = dicom_id

        cond = ((self.csv["dicom_id"] == dicom_id) & (
            self.csv["label_text"] == label_text)).values
        sub_labels = self.labels[cond].reshape(-1, len(self.pathologies))
        sample["lab"] = np.any(sub_labels, axis=0).astype(np.float32)
        sub_df = self.csv.loc[cond]

        filename = sub_df["path"].values[0]
        img_path = os.path.join(self.imgpath, filename)
        sample["path"] = img_path

        bboxes = sub_df[["x", "y", "w", "h"]].values
        bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
        bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
        sample["label_text"] = list(set(sub_df["label_text"].values.tolist()))

        img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = normalize(img, maxval=255, reshape=True)
        sample["img"] = img
        H, W, _ = img.shape
        sample["img_size"] = [H, W]
        bboxes = torch.from_numpy(bboxes)
        sub_labels = torch.from_numpy(sub_labels)

        if self.pathology_masks:
            mask_labels = torch.where(sub_labels == 1)[1].reshape(-1)
            unique_labels = torch.unique(mask_labels)

            gt_masks = []
            for m_label in unique_labels:
                # In some case, there are multiple boxes for the same label
                cur_indices = torch.where(m_label == mask_labels)[0]
                gt_mask = box2mask(bboxes[cur_indices], H, W)
                gt_masks.append(gt_mask)
            gt_masks = torch.cat(gt_masks, dim=0)
            sample["mask"] = gt_masks.numpy().transpose(1, 2, 0)

            # # FIXME: this part of code need to be refined
            # mask_file = os.path.join(self.mask_path, dicom_id + ".pkl")
            # assert os.path.exists(mask_file), f"{mask_file} doesn't exist!"
            # with open(mask_file, 'rb') as f:
            #     mask_dict = pickle.load(f)
            #     # img: (1024, 1024, 3)
            #     # mask: (1024, 1024, N)
            #     # mask_label: (N,)
            # sample["img"] = mask_dict["img"]
            # H, W, _ = mask_dict["img"].shape
            # sample["img_size"] = [H, W]
            # sample["mask"] = mask_dict["mask"]
            # sample["bbox"] = mask_dict["box"]
            # sample["bbox_label"] = mask_dict["box_label"]
            # valid_text = []
            # for p in mask_dict["mask_label"]:
            #     # randomly pick one
            #     p_idx = np.where(mask_dict["box_label"] == p)[0][0]
            #     valid_text.append(sample["label_text"][p_idx])
            # sample["label_text"] = valid_text
            # assert sample["mask"].shape[-1] == len(valid_text)
        else:
            sample["bbox"] = bboxes
            sample["bbox_label"] = np.where(sub_labels == 1)[1].reshape(-1, 1)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


# class MS_CXR_T_Dataset(BaseDataset):
#     """ MS-CXR-T: Learning to Exploit Temporal Structure for Biomedical Vision-Language Processing

#     Bannur, S., Hyland, S., Liu, Q., Pérez-García, F., Ilse, M., Coelho de Castro, D., Boecking,
#     B., Sharma, H., Bouzid, K., Schwaighofer, A., Wetscherek, M. T., Richardson, H., Naumann,
#     T., Alvarez Valle, J., & Oktay, O. (2023). MS-CXR-T: Learning to Exploit Temporal Structure
#     for Biomedical Vision-Language Processing (version 1.0.0).
#     PhysioNet. https://doi.org/10.13026/pg10-j984.

#     Dataset website:
#     https://physionet.org/content/ms-cxr-t/1.0.0/

#     """
#     def __init__(self,
#                  imgpath,
#                  csvpath,
#                  views=["PA", "AP"],
#                  seed=0,
#                  transform=None,
#                  data_aug=None):

#         np.random.seed(seed)
#         super().__init__()
#         default_pathologies = ["Consolidation",
#                                "Edema",
#                                "Pleural Effusion",
#                                "Pneumonia",
#                                "Pneumothorax"]
#         self.pathologies = default_pathologies
#         self.pathologies = sorted(self.pathologies)

#         self.imgpath = imgpath
#         self.csvpath = csvpath
#         self.views = views
#         self.transform = transform
#         self.data_aug = data_aug

#         self.csv = pd.read_csv(self.csvpath)
#         # self.agg_csv =
#         self.agg_csv = self.csv.drop_duplicates(subset=['dicom_id'])
#         # get labels
#         self.agg_csv["category_name"] = self.agg_csv["category_name"].apply(lambda x: x.lower())
#         labels = []
#         for pathology in self.pathologies:
#             mask = self.agg_csv["category_name"].str.contains(pathology.lower())
#             labels.append(mask.values)
#         self.labels = np.asarray(labels).T
#         self.labels = self.labels.astype(np.float32)
#         self.dicom_ids = self.agg_csv["dicom_id"].values

#     def __len__(self):
#         return len(self.labels)

#     def string(self):
#         return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

#     def __getitem__(self, idx):
#         sample = {}
#         sample["idx"] = idx
#         sample["lab"] = self.labels[idx]

#         dicom_id = self.dicom_ids[idx]
#         sub_df = self.csv[self.csv["dicom_id"] == dicom_id]
#         filename = sub_df["path"].values[0]
#         img_path = os.path.join(self.imgpath, filename)
#         img = imread(img_path, cv2.IMREAD_GRAYSCALE)
#         img = normalize(img, maxval=255, reshape=True)
#         sample["img"] = img

#         bboxes = sub_df[["x", "y", "w", "h"]].values
#         bboxes[:, 2] = bboxes[:, 0] + bboxes[:, 2]
#         bboxes[:, 3] = bboxes[:, 1] + bboxes[:, 3]
#         sample["bbox"] = bboxes
#         sample["bbox_label"] = np.where(sample["lab"] == 1)[0] * np.ones(len(bboxes), dtype=np.int64)
#         sample["label_text"] = sub_df["label_text"].values[0]
#         sample = apply_transforms(sample, self.transform)
#         sample = apply_transforms(sample, self.data_aug)
#         return sample


class Openi_Dataset(BaseDataset):
    """OpenI Dataset

    Dina Demner-Fushman, Marc D. Kohli, Marc B. Rosenman, Sonya E. Shooshan,
    Laritza Rodriguez, Sameer Antani, George R. Thoma, and Clement J.
    McDonald. Preparing a collection of radiology examinations for
    distribution and retrieval. Journal of the American Medical Informatics
    Association, 2016. doi: 10.1093/jamia/ocv080.

    Views have been determined by projection using T-SNE.  To use the T-SNE
    view rather than the view defined by the record,
    set use_tsne_derived_view to true.

    **Note: since the official website is hard to access, we use the dataset provided in Kaggle: 
    https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university/data. 

    Dataset website:
    https://openi.nlm.nih.gov/faq

    Download images:
    https://academictorrents.com/details/5a3a439df24931f410fac269b87b050203d9467d
    """

    def __init__(self, imgpath,
                 projection_csvpath=USE_INCLUDED_FILE,
                 report_csvpath=USE_INCLUDED_FILE,
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 nrows=None,
                 seed=0,
                 unique_patients=False,
                 pathologies=None,
                 medclip_format=False,
                 sent_info_dict=dict(),
                 ):

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        super().__init__()

        default_pathologies = ["No Finding",
                               "Enlarged Cardiomediastinum",
                               "Cardiomegaly",
                               "Lung Opacity",
                               "Lung Lesion",
                               "Edema",
                               "Consolidation",
                               "Pneumonia",
                               "Atelectasis",
                               "Pneumothorax",
                               "Pleural Effusion",
                               "Pleural Other",
                               "Fracture",
                               "Support Devices"]
        if pathologies is None:
            self.pathologies = sorted(default_pathologies)
        else:
            self.pathologies = sorted(pathologies)

        self.imgpath = imgpath
        self.csvpath = report_csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views
        self.transform = transform
        self.data_aug = data_aug
        self.medclip_format = medclip_format
        self.sent_info_dict = sent_info_dict

        mapping = {
            "Atelectasis": ["Atelectases"],
            "Pleural Effusion": ["Effusion"],
            "Pleural Other": ["Pleural_Thickening", "Thickening"],
            "Lung Lesion": ["Nodule", "Mass", "Emphysema", "Lesion", "Calcified Granuloma", "Granuloma"],
            "Lung Opacity": ["Nodule", "Mass", "Infiltration", "Opacity", "Fibrosis"],
            "Enlarged Cardiomediastinum": ["Hernia"],
            "Consolidation": ["Infiltration", "Fibrosis"]
        }

        projection_csv = pd.read_csv(projection_csvpath)
        self.csv = pd.merge(self.csv, projection_csv, on="uid")
        self.csv.loc[self.csv["projection"] == "Frontal", "projection"] = "PA"
        self.csv.loc[self.csv["projection"] == "Lateral", "projection"] = "L"
        self.csv["view"] = self.csv["projection"]
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("uid").first().reset_index()

        self.csv["report"] = self.csv["findings"].apply(
            self._split_report_into_segment)
        # filter rows with empty reports
        self.csv = self.csv[self.csv["report"].apply(lambda x: len(x) > 0)]

        # Get our classes.
        self.csv["Problems"] = self.csv["Problems"].apply(lambda x: x.lower())
        labels = []
        for pathology in self.pathologies:
            mask = self.csv["Problems"].str.contains(pathology.lower())
            if pathology in mapping:
                for syn in mapping[pathology]:
                    mask |= self.csv["Problems"].str.contains(syn.lower())
            labels.append(mask.values)

        # FIXME: check labels
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # sanity check
        normal_idx = self.pathologies.index("No Finding")
        abnormal_idx = [i for i in range(
            len(self.pathologies)) if i != normal_idx]
        abnormal_labels = self.labels[:, abnormal_idx]
        self.labels[:, normal_idx] = 1 - np.any(abnormal_labels != 0, axis=1)

        # create uncertain masks
        sum_samples_per_pathology = np.sum(self.labels, axis=0)
        uncertain_mask = sum_samples_per_pathology == 0
        self.labels[:, uncertain_mask] = -1

        # print(pd.DataFrame(self.labels, columns=self.pathologies).describe().T)

        if self.medclip_format:
            self.labels = _process_medclip_labels(self.labels)

        # patientid
        self.csv["patientid"] = self.csv["uid"].astype(str)

        self.num_patients = len(self.csv["patientid"].unique())

    def string(self):
        return self.__class__.__name__ + " num_patients={} num_samples={} views={} data_aug={}".format(
            self.num_patients, len(self), self.views, self.data_aug)

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace('\n', ' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower())

                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 4:  # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            return study_sent

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imageid = self.csv.iloc[idx].filename
        img_path = os.path.join(self.imgpath, imageid)
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        sample["img"] = normalize(img, maxval=255, reshape=True)
        if isinstance(self.transform, DataAugmentationDINO):
            sample = apply_dino_transforms(sample, self.transform)
        elif isinstance(self.transform, A.Compose):
            sample = apply_transforms(sample, self.transform)
        else:
            raise RuntimeError(f"{self.transform.__class__} doesn't belong to any"
                               " transform class (DataAugmentationDINO, A.Compose). Please check it.")

        if self.medclip_format:
            report = self.csv.iloc[idx]["report"]
            report = [sent for sent in report if "compare" not in sent]
            if len(report) == 0:
                # if all sentences belong to xxx compare xxx, then randomly sample one sentence
                report = self.csv.iloc[idx]["report"]
            # collect labels of all sentences
            sent_labels = []
            for sent in report:
                sent_labels.append(self.sent_info_dict[sent])
            sent_labels = np.asarray(sent_labels)
            # sample the sentence which has the same label with the image, instead of randomly sampling.
            sent_labels_sim = np.sum(sample["lab"] * sent_labels, axis=1)
            max_vals = np.max(sent_labels_sim)
            indices = np.where(sent_labels_sim == max_vals)[0]
            sent_list = np.random.choice(
                [report[int(x)] for x in indices], 2, replace=True).tolist()
            sample["report"] = sent_list
            sample["report_lab"] = self.sent_info_dict[sent_list[0]]

        return sample


class COVID19_Dataset(BaseDataset):
    """COVID-19 Image Data Collection

    This dataset currently contains hundreds of frontal view X-rays and is
    the largest public resource for COVID-19 image and prognostic data,
    making it a necessary resource to develop and evaluate tools to aid in
    the treatment of COVID-19. It was manually aggregated from publication
    figures as well as various web based repositories into a machine learning
    (ML) friendly format with accompanying dataloader code. We collected
    frontal and lateral view imagery and metadata such as the time since
    first symptoms, intensive care unit (ICU) status, survival status,
    intubation status, or hospital location. We present multiple possible use
    cases for the data such as predicting the need for the ICU, predicting
    patient survival, and understanding a patient's trajectory during
    treatment.

    Citations:

    COVID-19 Image Data Collection: Prospective Predictions Are the Future
    Joseph Paul Cohen and Paul Morrison and Lan Dao and Karsten Roth and Tim
    Q Duong and Marzyeh Ghassemi arXiv:2006.11988, 2020

    COVID-19 image data collection,
    Joseph Paul Cohen and Paul Morrison and Lan Dao
    arXiv:2003.11597, 2020

    Dataset: https://github.com/ieee8023/covid-chestxray-dataset

    Paper: https://arxiv.org/abs/2003.11597
    """

    dataset_url = "https://github.com/ieee8023/covid-chestxray-dataset"

    def __init__(self,
                 imgpath: str,
                 csvpath: str,
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 seed: int = 0,
                 semantic_masks=False,
                 ):
        """
        Args:
            imgpath: Path to the directory containing images
            csvpath: Path to the image directory
        """

        super(COVID19_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.views = views
        self.semantic_masks = semantic_masks
        self.semantic_masks_v7labs_lungs_path = os.path.join(
            datapath, "semantic_masks_v7labs_lungs.zip")

        if not os.path.exists(csvpath):
            raise FileNotFoundError(
                f'The csvpath does not point to a valid metadata.csv file. Please download it from {self.dataset_url}')

        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)

        # Keep only the selected views.
        self.limit_to_selected_views(views)

        # Filter out in progress samples
        self.csv = self.csv[~(self.csv.finding == "todo")]
        self.csv = self.csv[~(self.csv.finding == "Unknown")]

        self.pathologies = self.csv.finding.str.split(
            "/", expand=True).values.ravel()
        self.pathologies = self.pathologies[~pd.isnull(self.pathologies)]
        self.pathologies = sorted(np.unique(self.pathologies))

        labels = []
        for pathology in self.pathologies:
            mask = self.csv["finding"].str.contains(pathology)
            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.csv = self.csv.reset_index()

        if self.semantic_masks:
            temp = zipfile.ZipFile(self.semantic_masks_v7labs_lungs_path)
            self.semantic_masks_v7labs_lungs_namelist = temp.namelist()

        # add consistent csv values

        # offset_day_int
        self.csv["offset_day_int"] = self.csv["offset"]

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['filename'].iloc[idx]
        sample["imgid"] = imgid
        img_path = os.path.join(self.imgpath, imgid)
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        if self.semantic_masks:
            sample["semantic_masks"] = self.get_semantic_mask_dict(
                imgid, sample["img"].shape)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def get_semantic_mask_dict(self, image_name, this_shape):

        archive_path = "semantic_masks_v7labs_lungs/" + image_name
        semantic_masks = {}
        if archive_path in self.semantic_masks_v7labs_lungs_namelist:
            with zipfile.ZipFile(self.semantic_masks_v7labs_lungs_path).open(archive_path) as file:
                mask = imageio.imread(file.read())

                mask = (mask == 255).astype(np.float)
                # Reshape so image resizing works
                mask = mask[None, :, :]

                semantic_masks["Lungs"] = mask

        return semantic_masks


class NLMTB_Dataset(BaseDataset):
    """National Library of Medicine Tuberculosis Datasets

    https://lhncbc.nlm.nih.gov/publication/pub9931
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4256233/

    Note that each dataset should be loaded separately by this class (they
    may be merged afterwards).  All images are of view PA.

    Jaeger S, Candemir S, Antani S, Wang YX, Lu PX, Thoma G. Two public chest
    X-ray datasets for computer-aided screening of pulmonary diseases. Quant
    Imaging Med Surg. 2014 Dec;4(6):475-7. doi:
    10.3978/j.issn.2223-4292.2014.11.20. PMID: 25525580; PMCID: PMC4256233.

    Download Links:
    Montgomery County
    https://academictorrents.com/details/ac786f74878a5775c81d490b23842fd4736bfe33
    http://openi.nlm.nih.gov/imgs/collections/NLM-MontgomeryCXRSet.zip

    Shenzhen
    https://academictorrents.com/details/462728e890bd37c05e9439c885df7afc36209cc8
    http://openi.nlm.nih.gov/imgs/collections/ChinaSet_AllFiles.zip
    """

    def __init__(self,
                 imgpath,
                 transform=None,
                 data_aug=None,
                 seed=0,
                 views=["PA"]
                 ):
        """
        Args:
            img_path (str): Path to `MontgomerySet` or `ChinaSet_AllFiles`
                folder
        """

        super(NLMTB_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        file_list = []
        source_list = []

        for fname in sorted(os.listdir(os.path.join(self.imgpath, "CXR_png"))):
            if fname.endswith(".png"):
                file_list.append(fname)

        self.csv = pd.DataFrame({"fname": file_list})

        # Label is the last digit on the simage filename
        self.csv["label"] = self.csv["fname"].apply(
            lambda x: int(x.split(".")[-2][-1]))
        # All the images are PA according to the article.
        self.csv["view"] = "PA"
        self.limit_to_selected_views(views)

        self.labels = self.csv["label"].values.reshape(-1, 1)
        self.pathologies = ["Tuberculosis"]

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        item = self.csv.iloc[idx]
        img_path = os.path.join(self.imgpath, "CXR_png", item["fname"])
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class SIIM_Pneumothorax_Dataset(BaseDataset):
    """SIIM Pneumothorax Dataset

    https://academictorrents.com/details/6ef7c6d039e85152c4d0f31d83fa70edc4aba088
    https://www.kaggle.com/c/siim-acr-pneumothorax-segmentation

    "The data is comprised of images in DICOM format and annotations in the
    form of image IDs and run-length-encoded (RLE) masks. Some of the images
    contain instances of pneumothorax (collapsed lung), which are indicated
    by encoded binary masks in the annotations. Some training images have
    multiple annotations. Images without pneumothorax have a mask value of -1."
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 transform=None,
                 data_aug=None,
                 seed=0,
                 unique_patients=False,
                 pathology_masks=True,
                 return_bbox=False,
                 nonzero_mask=False,
                 bbox_shift=10
                 ):
        super(SIIM_Pneumothorax_Dataset, self).__init__()
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        self.bbox_shift = bbox_shift
        self.nonzero_mask = nonzero_mask
        self.return_bbox = return_bbox

        # Load data
        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(
                datapath, "siim-pneumothorax-train-rle.csv.gz")
        else:
            self.csvpath = csvpath

        self.raw_csv = pd.read_csv(self.csvpath)
        self.raw_csv["has_masks"] = self.raw_csv[" EncodedPixels"] != "-1"
        if self.nonzero_mask:
            self.raw_csv = self.raw_csv.loc[self.raw_csv["has_masks"], :].reset_index(
                drop=True)

        self.csv = self.raw_csv.groupby("ImageId").first().reset_index()
        self.pathologies = ["Pneumothorax"]
        labels = [self.csv[" EncodedPixels"] != "-1"]
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # To figure out the paths
        # TODO: make faster
        if not ("siim_file_map" in _cache_dict):
            file_map = {}
            for root, directories, files in os.walk(self.imgpath, followlinks=False):
                for filename in files:
                    filePath = os.path.join(root, filename)
                    file_map[filename] = filePath
            _cache_dict["siim_file_map"] = file_map
        self.file_map = _cache_dict["siim_file_map"]

    def string(self):
        return self.__class__.__name__ + " num_patients=NA num_samples={} views=PA/AP data_aug={}".format(
            len(self), self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['ImageId'].iloc[idx]
        img_path = self.file_map[imgid + ".dcm"]
        sample["imgid"] = imgid

        try:
            import pydicom
        except ImportError as e:
            raise Exception("Please install pydicom to work with this dataset")
        img = pydicom.filereader.dcmread(img_path).pixel_array
        img = normalize(img, maxval=255, reshape=True)  # 2D
        sample["img"] = img
        H, W, _ = img.shape
        sample["img_size"] = [H, W]

        if self.pathology_masks:
            pathology_mask = self.get_pathology_mask_dict(
                imgid, sample["img"].shape[1])
            if len(pathology_mask) == 0:
                sample["mask"] = np.zeros(
                    (sample["img"].shape[0], sample["img"].shape[1], 1))
            else:
                sample["mask"] = pathology_mask[0].reshape(
                    (sample["img"].shape[0], sample["img"].shape[1], 1))

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        # create bbox for masks
        if self.return_bbox:
            # FIXME: don't consider multiple connected components
            mask_labels = np.where(sample["lab"] == 1)[0]
            sample["mask"], sample["bbox"], sample["bbox_label"] = detect_bbox_from_masks(
                sample["mask"], mask_labels, bbox_shift=self.bbox_shift)

        return sample

    def get_pathology_mask_dict(self, image_name, this_size):

        base_size = 1024
        images_with_masks = self.raw_csv[np.logical_and(self.raw_csv["ImageId"] == image_name,
                                                        self.raw_csv[" EncodedPixels"] != "-1")]
        path_mask = {}

        # From kaggle code
        def rle2mask(rle, width, height):
            mask = np.zeros(width * height)
            array = np.asarray([int(x) for x in rle.split()])
            starts = array[0::2]
            lengths = array[1::2]

            current_position = 0
            for index, start in enumerate(starts):
                current_position += start
                mask[current_position:current_position + lengths[index]] = 1
                current_position += lengths[index]

            return mask.reshape(width, height)

        if len(images_with_masks) > 0:
            # Using a for loop so it is consistent with the other code
            for patho in ["Pneumothorax"]:
                seg_mask = np.zeros([this_size, this_size])

                # don't add masks for labels we don't have
                if patho in self.pathologies:

                    for i in range(len(images_with_masks)):
                        row = images_with_masks.iloc[i]
                        mask = rle2mask(
                            row[" EncodedPixels"], base_size, base_size)
                        mask = mask.T
                        mask = skimage.transform.resize(
                            mask, (this_size, this_size), mode='constant', order=0)
                        mask = mask.round()  # make 0,1
                        seg_mask += mask

                seg_mask = (seg_mask > 0).astype(np.float32)
                # reshape so image resizing works
                seg_mask = seg_mask[:, :, None]

                path_mask[self.pathologies.index(patho)] = seg_mask

        return path_mask


class CANDID_PTX_Dataset(BaseDataset):
    def __init__(self,
                 imgpath,
                 csvpath,
                 pathologies=None,
                 views=["AP", "PA"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 pathology_masks=True,
                 return_bbox=False,
                 nonzero_mask=True,
                 unique_patients=False,
                 bbox_shift=10):
        np.random.seed(seed)  # Reset the seed so all runs are the same.
        super().__init__()
        default_pathologies = ["Pneumothorax"]
        if pathologies is None:
            self.pathologies = sorted(default_pathologies)
        else:
            self.pathologies = sorted(pathologies)
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.rawcsv = pd.read_csv(self.csvpath)
        self.transform = transform
        self.data_aug = data_aug
        self.nonzero_mask = nonzero_mask
        self.pathology_masks = pathology_masks
        self.return_bbox = return_bbox
        self.bbox_shift = bbox_shift

        if self.nonzero_mask:
            self.rawcsv = self.rawcsv[self.rawcsv["EncodedPixels"]
                                      != "-1"].reset_index(drop=True)
        self.rawcsv["Report"] = self.rawcsv["Report"].apply(
            self._split_report_into_segment)
        self.csv = self.rawcsv.groupby("SOPInstanceUID").first().reset_index()
        labels = [self.csv["EncodedPixels"] != "-1"]
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

    def string(self):
        return self.__class__.__name__ + " num_samples={} data_aug={}".format(len(self), self.data_aug)

    def __len__(self):
        return len(self.labels)

    def _split_report_into_segment(self, report):
        '''clean up raw reports into sentences
        '''
        if pd.isnull(report):
            return []
        else:
            report = report.replace("[ALPHANUMERICID]", "")
            report = report.replace("[date]", "")
            report = report.replace("[DATE]", "")
            report = report.replace("[AGE]", "")

            report = report.replace("[ADDRESS]", "")
            report = report.replace("[PERSONALNAME]", "")

            report = report.replace('\n', ' ')
            # splitter = re.compile("[0-9]+\.")
            splitter = re.compile("[0-9]+\.+[^0-9]")
            report = splitter.split(report)
            reports = [point.split(". ") for point in report]
            # reports = [point.split(".") for point in report]
            reports = [sent for point in reports for sent in point]
            study_sent = []
            for sent in reports:
                if len(sent) == 0:
                    continue

                sent = sent.replace("\ufffd\ufffd", " ")
                # tokenizer = RegexpTokenizer(r"\w+")
                # tokens = tokenizer.tokenize(sent.lower())

                tokens = tokens = nltk.wordpunct_tokenize(sent.lower())

                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)
                if len(included_tokens) > 4:  # only include relative long sentences
                    study_sent.append(" ".join(included_tokens))
            study_sent = [x for x in study_sent if "clinical data" not in x]
            study_sent = [x for x in study_sent if "comparison" not in x]
            study_sent = [x for x in study_sent if "medical question" not in x]
            study_sent = [x for x in study_sent if "transcribed" not in x]

            return study_sent

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv["SOPInstanceUID"].iloc[idx]
        sample["imgid"] = imgid
        img_path = os.path.join(self.imgpath, imgid)

        try:
            import pydicom
        except ImportError as e:
            raise Exception("Please install pydicom to work with this dataset")
        img = pydicom.filereader.dcmread(img_path).pixel_array
        img = normalize(img, maxval=img.max(), reshape=True)  # 2D
        sample["img"] = img
        H, W, _ = img.shape
        sample["img_size"] = [H, W]

        if self.pathology_masks:
            pathology_mask = self.get_pathology_mask_dict(
                imgid, sample["img"].shape[0], sample["img"].shape[1])
            if len(pathology_mask) == 0:
                sample["mask"] = np.zeros(
                    (sample["img"].shape[0], sample["img"].shape[1], 1))
            else:
                sample["mask"] = pathology_mask[0].reshape(
                    (sample["img"].shape[0], sample["img"].shape[1], 1))

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        # create bbox for masks
        if self.return_bbox:
            # FIXME: don't consider multiple connected components
            mask_labels = np.where(sample["lab"] == 1)[0]
            sample["mask"], sample["bbox"], sample["bbox_label"] = detect_bbox_from_masks(
                sample["mask"], mask_labels, bbox_shift=self.bbox_shift)

        return sample

    def get_pathology_mask_dict(self, image_name, width, height):
        # base_size = 1024
        images_with_masks = self.csv[np.logical_and(self.csv["SOPInstanceUID"] == image_name,
                                                    self.csv["EncodedPixels"] != "-1")]
        path_mask = {}

        # From kaggle code
        def rle2mask(rle, width, height):
            mask = np.zeros(width * height)
            array = np.asarray([int(x) for x in rle.split()])
            starts = array[0::2]
            lengths = array[1::2]

            current_position = 0
            for index, start in enumerate(starts):
                current_position += start
                mask[current_position:current_position + lengths[index]] = 1
                current_position += lengths[index]

            return mask.reshape(width, height)

        if len(images_with_masks) > 0:
            # Using a for loop so it is consistent with the other code
            for patho in ["Pneumothorax"]:
                mask = np.zeros([width, height])

                # don't add masks for labels we don't have
                if patho in self.pathologies:

                    for i in range(len(images_with_masks)):
                        row = images_with_masks.iloc[i]
                        mask = rle2mask(row["EncodedPixels"], width, height)
                        mask = mask.T
                        mask = skimage.transform.resize(
                            mask, (width, height), mode='constant', order=0)
                        mask = mask.round()  # make 0,1

                # reshape so image resizing works
                mask = mask[:, :, None]

                path_mask[self.pathologies.index(patho)] = mask

        return path_mask


class VinBrain_Dataset(BaseDataset):
    """VinBrain Dataset

    .. code-block:: python

        d_vin = xrv.datasets.VinBrain_Dataset(
            imgpath=".../train",
            csvpath=".../train.csv"
        )

    Nguyen, H. Q., Lam, K., Le, L. T., Pham, H. H., Tran, D. Q., Nguyen,
    D. B., Le, D. D., Pham, C. M., Tong, H. T. T., Dinh, D. H., Do, C. D.,
    Doan, L. T., Nguyen, C. N., Nguyen, B. T., Nguyen, Q. V., Hoang, A. D.,
    Phan, H. N., Nguyen, A. T., Ho, P. H., … Vu, V. (2020). VinDr-CXR: An
    open dataset of chest X-rays with radiologist’s annotations.
    http://arxiv.org/abs/2012.15029

    https://www.kaggle.com/c/vinbigdata-chest-xray-abnormalities-detection
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 mask_path=USE_INCLUDED_FILE,
                 views=None,
                 transform=None,
                 data_aug=None,
                 seed=0,
                 pathology_masks=False,
                 bbox_only=False,
                 ):
        super(VinBrain_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath

        if csvpath == USE_INCLUDED_FILE:
            self.csvpath = os.path.join(datapath, "vinbigdata-train.csv.gz")
        else:
            self.csvpath = csvpath

        self.mask_path = mask_path
        self.transform = transform
        self.data_aug = data_aug
        self.pathology_masks = pathology_masks
        self.views = views
        self.bbox_only = bbox_only

        self.pathologies = ['Aortic enlargement',
                            'Atelectasis',
                            'Calcification',
                            'Cardiomegaly',
                            'Consolidation',
                            'ILD',
                            'Infiltration',
                            'Lung Opacity',
                            'Nodule/Mass',
                            'Other lesion',
                            'Pleural effusion',
                            'Pleural thickening',
                            'Pneumothorax',
                            'Pulmonary fibrosis']

        # Load data
        self.check_paths_exist()
        self.rawcsv = pd.read_csv(self.csvpath)
        if self.bbox_only or self.pathology_masks:
            self.rawcsv = self.rawcsv[self.rawcsv["class_id"] != 14]

        self.csv = pd.DataFrame(self.rawcsv.groupby("image_id")[
                                "class_name"].apply(lambda x: "|".join(np.unique(x))))
        self.csv["has_masks"] = self.csv.class_name != "No finding"

        labels = []
        for pathology in self.pathologies:
            mask = self.csv["class_name"].str.lower(
            ).str.contains(pathology.lower())
            labels.append(mask.values)
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.csv = self.csv.reset_index()

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        imgid = self.csv['image_id'].iloc[idx]
        sample["imgid"] = imgid
        img_path = os.path.join(self.imgpath, imgid + ".dicom")

        try:
            import pydicom
        except ImportError as e:
            raise Exception(
                "Please install pydicom to work with this dataset")
        from pydicom.pixel_data_handlers.util import apply_modality_lut
        dicom_obj = pydicom.filereader.dcmread(img_path)
        img = apply_modality_lut(dicom_obj.pixel_array, dicom_obj)
        img = pydicom.pixel_data_handlers.apply_windowing(img, dicom_obj)

        # Photometric Interpretation to see if the image needs to be inverted
        mode = dicom_obj[0x28, 0x04].value
        bitdepth = dicom_obj[0x28, 0x101].value

        # hack!
        if img.max() < 256:
            bitdepth = 8

        if mode == "MONOCHROME1":
            img = -1 * img + 2**float(bitdepth)
        elif mode == "MONOCHROME2":
            pass
        else:
            raise Exception("Unknown Photometric Interpretation mode")

        img = normalize(img, maxval=2**float(bitdepth), reshape=True)
        H, W, _ = img.shape
        sample["img_size"] = [H, W]
        sample["img"] = img

        if self.bbox_only:
            sub_df = self.rawcsv[self.rawcsv["image_id"] == imgid]
            sample["bbox"] = sub_df[[
                "x_min", "y_min", "x_max", "y_max"]].values
            sample["bbox_label"] = sub_df["class_id"].values
        elif self.pathology_masks:
            # mask_file = os.path.join(self.mask_path, imgid + ".pkl")
            # with open(mask_file, "rb") as f:
            #     mask_dict = pickle.load(f)
            # sample["img"] = mask_dict["img"]
            # sample["mask"] = mask_dict["mask"]
            sub_df = self.rawcsv[self.rawcsv["image_id"] == imgid]
            unique_mask_labels = np.unique(sub_df["class_id"])
            mask = np.zeros((H, W, len(unique_mask_labels)))
            for i, label in enumerate(unique_mask_labels):
                sub_df_label = sub_df[sub_df["class_id"] == label]
                for j in range(len(sub_df_label)):
                    row = sub_df_label.iloc[j]
                    x_min, y_min, x_max, y_max = row[[
                        "x_min", "y_min", "x_max", "y_max"]].astype(int)
                    mask[y_min:y_max, x_min:x_max, i] = 1
            sample["mask"] = mask

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class COVID_Rural_Dataset(BaseDataset):
    """ 
    COVID Rural Dataset

    https://github.com/haimingt/opacity_segmentation_covid_chest_X_ray

    Citation: 
    Deep learning segmentation model for automated detection of the opacity regions in the 
    chest X-rays of the Covid-19 positive patients and the application for disease severity

    """

    def __init__(self,
                 imgpath,  # path to images
                 annpath,  # path to annotations
                 transform=None,
                 data_aug=None,
                 seed=0,
                 views=["PA", "AP"],
                 nonzero_mask=True
                 ):
        super(COVID_Rural_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.annpath = annpath
        self.transform = transform
        self.data_aug = data_aug

        self.pathologies = ["COVID19"]

        all_img_paths = []
        all_seg_masks = []
        all_subject_ids = []
        all_labels = []
        for json_file in os.listdir(self.annpath):
            with open(os.path.join(self.annpath, json_file)) as f:
                ann = json.load(f)
                height, width = ann["imageHeight"], ann["imageWidth"]
                mask = self.create_mask(ann["shapes"], height, width)
                subject_id = "-".join(json_file.split("-")[:4])
                if nonzero_mask:
                    if mask.sum() > 0:
                        all_subject_ids.append(subject_id)
                        all_img_paths.append(os.path.join(
                            self.imgpath, ann["imagePath"]))
                        all_seg_masks.append(mask)
                        all_labels.append(1)
                else:
                    all_subject_ids.append(subject_id)
                    all_img_paths.append(os.path.join(
                        self.imgpath, ann["imagePath"]))
                    all_seg_masks.append(mask)
                    if mask.sum() > 0:
                        all_labels.append(1)
                    else:
                        all_labels.append(0)

        self.img_paths = all_img_paths
        self.masks = all_seg_masks
        self.subject_ids = all_subject_ids
        self.labels = np.array(all_labels, dtype=np.float32).reshape(-1, 1)

    def string(self):
        return self.__class__.__name__ + " num_patients={} num_samples={}".format(len(set(self.subject_ids)), len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        sample["imgid"] = self.subject_ids[idx]

        img_path = self.img_paths[idx]
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)
        sample["img"] = normalize(img, maxval=255, reshape=True)

        seg_mask = self.masks[idx]
        sample["mask"] = seg_mask[..., np.newaxis]

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample

    def create_mask(self, polygon_dict, height, width):
        ''' Create mask for polygons '''

        mask = np.zeros((height, width), dtype=np.uint8)

        for shape in polygon_dict:
            if shape["label"] not in ["right_lung", "left_lung"]:
                polygons = shape["points"]

                polygons = np.array(polygons)
                arr = np.zeros((len(polygons), 2))
                for i, p in enumerate(polygons):
                    arr[i, 0] = int(round(p[0]))
                    arr[i, 1] = int(round(p[1]))

                # Draw the element wit a color depending on the Class
                cv2.drawContours(
                    mask, [arr.astype(np.int32)], -1, (1, 255, 255), -1, cv2.LINE_AA)

        return mask


class StonyBrookCOVID_Dataset(BaseDataset):
    """Stonybrook Radiographic Assessment of Lung Opacity Score Dataset

    https://doi.org/10.5281/zenodo.4633999

    Citation will be set soon.
    """

    def __init__(self,
                 imgpath,  # path to CXR_images_scored
                 csvpath,  # path to ralo-dataset-metadata.csv
                 transform=None,
                 data_aug=None,
                 views=["AP"],
                 seed=0
                 ):
        super(StonyBrookCOVID_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgpath = imgpath
        self.transform = transform
        self.data_aug = data_aug

        # Load data
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath, skiprows=1)
        self.MAXVAL = 255  # Range [0 255]

        self.pathologies = ["Geographic Extent", "Lung Opacity"]

        self.csv["Geographic Extent"] = (
            self.csv["Total GEOGRAPHIC"] + self.csv["Total GEOGRAPHIC.1"]) / 2
        self.csv["Lung Opacity"] = (
            self.csv["Total OPACITY"] + self.csv["Total OPACITY.1"]) / 2

        labels = []
        labels.append(self.csv["Geographic Extent"])
        labels.append(self.csv["Lung Opacity"])

        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # add consistent csv values

        # offset_day_int

        date_col = self.csv["Exam_DateTime"].str.split("_", expand=True)[0]
        dt = pd.to_datetime(date_col, format="%Y%m%d")
        self.csv["offset_day_int"] = dt.astype(np.int64) // 10**9 // 86400

        # patientid
        self.csv["patientid"] = self.csv["Subject_ID"].astype(str)

        # all the images are AP according to the article.
        self.csv["view"] = "AP"
        self.limit_to_selected_views(views)

    def string(self):
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]

        img_path = os.path.join(self.imgpath, str(idx) + ".jpg")
        img = imread(img_path, cv2.IMREAD_GRAYSCALE)

        sample["img"] = normalize(img, maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class ObjectCXR_Dataset(BaseDataset):
    """ObjectCXR Dataset

    "We provide a large dataset of chest X-rays with strong annotations of
    foreign objects, and the competition for automatic detection of foreign
    objects. Specifically, 5000 frontal chest X-ray images with foreign
    objects presented and 5000 frontal chest X-ray images without foreign
    objects are provided. All the chest X-ray images were filmed in township
    hospitals in China and collected through our telemedicine platform.
    Foreign objects within the lung field of each chest X-ray are annotated
    with bounding boxes, ellipses or masks depending on the shape of the
    objects."

    Challenge dataset from MIDL2020

    https://jfhealthcare.github.io/object-CXR/

    https://academictorrents.com/details/fdc91f11d7010f7259a05403fc9d00079a09f5d5
    """

    def __init__(self,
                 imgzippath,
                 csvpath,
                 transform=None,
                 data_aug=None,
                 seed=0
                 ):
        super(ObjectCXR_Dataset, self).__init__()

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        self.imgzippath = imgzippath
        self.csvpath = csvpath
        self.transform = transform
        self.data_aug = data_aug
        self.views = []
        self.pathologies = ['Foreign Object']

        # Load data
        self.csv = pd.read_csv(self.csvpath)

        labels = []
        labels.append(~self.csv["annotation"].isnull())
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        self.csv = self.csv.reset_index()

        self.csv["has_masks"] = ~self.csv["annotation"].isnull()

        self.imgzip = zipfile.ZipFile(self.imgzippath)

    def string(self):
        return self.__class__.__name__ + " num_samples={} views={} data_aug={}".format(len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        imgid = self.csv.iloc[idx]["image_name"]

        with zipfile.ZipFile(self.imgzippath).open("train/" + imgid) as file:
            sample["img"] = imread(file.read(), cv2.IMREAD_GRAYSCALE)

        sample["img"] = normalize(sample["img"], maxval=255, reshape=True)

        sample = apply_transforms(sample, self.transform)
        sample = apply_transforms(sample, self.data_aug)

        return sample


class ToPILImage(object):
    def __init__(self):
        self.to_pil = transforms.ToPILImage(mode="F")

    def __call__(self, x):
        return self.to_pil(x[0])


class XRayResizer(object):
    """Resize an image to a specific size"""

    def __init__(self, size: int, engine="skimage"):
        self.size = size
        self.engine = engine
        if 'cv2' in sys.modules:
            print("Setting XRayResizer engine to cv2 could increase performance.")

    def __call__(self, img: np.ndarray) -> np.ndarray:
        if self.engine == "skimage":
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return skimage.transform.resize(img, (1, self.size, self.size), mode='constant', preserve_range=True).astype(np.float32)
        elif self.engine == "cv2":
            import cv2  # pip install opencv-python
            return cv2.resize(img[0, :, :],
                              (self.size, self.size),
                              interpolation=cv2.INTER_AREA
                              ).reshape(1, self.size, self.size).astype(np.float32)
        else:
            raise Exception(
                "Unknown engine, Must be skimage (default) or cv2.")


class BRAX_Dataset(BaseDataset):
    """BRAX, a Brazilian labeled chest X-ray dataset

    The Brazilian labeled chest x-ray dataset (BRAX) is an automatically labeled dataset 
    designed to assist researchers in the validation of machine learning models. 
    The dataset contains 24,959 chest radiography studies from patients presenting to 
    a large general Brazilian hospital. A total of 40,967 images are available in the BRAX dataset. 
    All images have been verified by trained radiologists and de-identified to protect patient privacy. 
    Fourteen labels were derived from free-text radiology reports written in 
    Brazilian Portuguese using Natural Language Processing. 
    More details can be seen in the paper: https://www.nature.com/articles/s41597-022-01608-8 

    Dataset release website:
    https://physionet.org/content/brax/1.1.0/
    """

    def __init__(self,
                 imgpath,
                 csvpath=USE_INCLUDED_FILE,
                 views=["PA", "AP"],
                 transform=None,
                 data_aug=None,
                 seed=0,
                 unique_patients=False,
                 medclip_format=False,
                 pathologies=None,
                 sentence_label=None,
                 data_pct=1.,
                 prompt_sentence_label=None,
                 sent_info_dict=None
                 ):

        np.random.seed(seed)  # Reset the seed so all runs are the same.
        super().__init__()
        default_pathologies = ["No Finding",
                               "Enlarged Cardiomediastinum",
                               "Cardiomegaly",
                               "Lung Opacity",
                               "Lung Lesion",
                               "Edema",
                               "Consolidation",
                               "Pneumonia",
                               "Atelectasis",
                               "Pneumothorax",
                               "Pleural Effusion",
                               "Pleural Other",
                               "Fracture",
                               "Support Devices"]
        if pathologies is None:
            self.pathologies = default_pathologies
        else:
            self.pathologies = pathologies
        self.pathologies = sorted(self.pathologies)
        self.imgpath = imgpath
        self.csvpath = csvpath
        self.csv = pd.read_csv(self.csvpath)
        self.views = views
        self.medclip_format = medclip_format
        self.transform = transform
        self.data_aug = data_aug
        self.sentence_label = sentence_label
        self.prompt_sentence_label = prompt_sentence_label
        self.sent_info_dict = sent_info_dict

        # Load data
        self.check_paths_exist()

        # Remove images with view position other than specified
        self.csv["view"] = self.csv['ViewPosition']
        self.limit_to_selected_views(views)

        if unique_patients:
            self.csv = self.csv.groupby("Patient ID").first()
        self.csv = self.csv.reset_index()

        self.csv.fillna(0, inplace=True)

        # sample a subset of dataset
        if data_pct < 1.0:
            self.csv = self.csv.sample(
                frac=data_pct, random_state=seed).reset_index(drop=True)

        # Get our classes.
        healthy = self.csv["No Finding"] == 1
        labels = []
        for pathology in self.pathologies:
            if pathology in self.csv.columns:
                if pathology not in ["No Finding", "Support Devices"]:
                    self.csv.loc[healthy, pathology] = 0
                mask = self.csv[pathology]

            labels.append(mask.values)
        # FIXME: check labels
        self.labels = np.asarray(labels).T
        self.labels = self.labels.astype(np.float32)

        # sanity check:
        normal_idx = self.pathologies.index("No Finding")
        abnormal_idx = [i for i in range(
            len(self.pathologies)) if i != normal_idx]
        abnormal_labels = self.labels[:, abnormal_idx]
        self.labels[:, normal_idx] = 1 - np.any(abnormal_labels != 0, axis=1)
        # print(pd.DataFrame(self.labels, columns=self.pathologies).describe().T)

        if self.medclip_format:
            self.labels = _process_medclip_labels(self.labels)

        # patientid
        self.csv["patientid"] = self.csv["PatientID"].astype(str)
        # age
        # self.csv['age_years'] = self.csv['PatientAge'].astype(int) * 1.0
        # sex
        self.csv['sex_male'] = self.csv['PatientSex'] == 'M'
        self.csv['sex_female'] = self.csv['PatientSex'] == 'F'

        self.num_patients = len(self.csv["patientid"].unique())

    def string(self):
        return self.__class__.__name__ + " num_patients={} num_samples={} views={} data_aug={}".format(
            self.num_patients, len(self), self.views, self.data_aug)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = {}
        sample["idx"] = idx
        sample["lab"] = self.labels[idx]
        imgpath = os.path.join(self.imgpath, self.csv.iloc[idx]["PngPath"])
        img = imread(imgpath, cv2.IMREAD_GRAYSCALE)
        sample["img"] = normalize(img, maxval=255, reshape=True)
        if isinstance(self.transform, DataAugmentationDINO):
            sample = apply_dino_transforms(sample, self.transform)
        elif isinstance(self.transform, A.Compose):
            sample = apply_transforms(sample, self.transform)
        else:
            raise RuntimeError(f"{self.transform.__class__} doesn't belong to any"
                               " transform class (DataAugmentationDINO, A.Compose). Please check it.")
        if self.medclip_format:
            sampled_report = sample_sent_prompts(
                sample, self.pathologies, self.sentence_label, self.prompt_sentence_label,
                num_reports=2)
            sample["report"] = sampled_report
            sample["report_lab"] = self.sent_info_dict[sampled_report[0]]

        return sample


class XRayCenterCrop(object):
    """Perform a center crop on the long dimension of the input image"""

    def crop_center(self, img: np.ndarray) -> np.ndarray:
        _, y, x = img.shape
        crop_size = np.min([y, x])
        startx = x // 2 - (crop_size // 2)
        starty = y // 2 - (crop_size // 2)
        return img[:, starty:starty + crop_size, startx:startx + crop_size]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        return self.crop_center(img)


class CovariateDataset(BaseDataset):
    """A covariate shift between two data distributions arises when some
    extraneous variable confounds with the variables of interest in the first
    dataset differently than in the second [Moreno-Torres et al., 2012].
    Covariate shifts between the training and test distribution in a machine
    learning setting can lead to models which generalize poorly, and this
    phenomenon is commonly observed in CXR models trained on a small dataset
    and deployed on another one [Zhao et al., 2019; DeGrave et al., 2020]. We
    provide tools to simulate covariate shifts in these datasets so
    researchers can evaluate the susceptibility of their models to these
    shifts, or explore mitigation strategies.

    .. code-block:: python

        d = xrv.datasets.CovariateDataset(
            d1 = # dataset1 with a specific condition.
            d1_target = # target label to predict.
            d2 = # dataset2 with a specific condition.
            d2_target = #target label to predict.
            mode="train", # train, valid, or test.
            ratio=0.75
        )

    .. image:: _static/CovariateDataset-Diagram.png

    The class xrv.datasets.CovariateDataset takes two datasets and two arrays
    representing the labels. It returns samples for the output classes with a
    specified ratio of examples from each dataset, thereby introducing a
    correlation between any dataset-specific nuisance features and the output
    label. This simulates a covariate shift. The test split can be set up
    with a different ratio than the training split; this setup has been shown
    to both decrease generalization performance and exacerbate incorrect
    feature attribution [Viviano et al., 2020]. See Figure 4 for a
    visualization of the effect the ratio parameter has on the mean class
    difference when correlating the view (each dataset) with the target
    label. The effect seen with low ratios is due to the majority of the
    positive labels being drawn from the first dataset, where in the high
    ratios, the majority of the positive labels are drawn from the second
    dataset. With any ratio, the number of samples returned will be the same
    in order to provide controlled experiments. The dataset has 3 modes,
    train sampled using the provided ratio and the valid and test dataset are
    sampled using 1−ratio.

    An example of the mean class difference drawn from the COVID-19 dataset
    at different covariate ratios. Here, the first COVID-19 dataset consisted
    of only AP images, whereas the second dataset consisted of only PA
    images. The third row shows, for each ratio, the difference in the class
    means, demonstrating the effect of sampling images from the two views on
    the perceived class difference. The fourth row shows the difference
    between each ratio’s difference image, and the difference image with a
    ratio of 0.5 (balanced sampling from all views).

    .. image:: _static/covariate.png

    Citation:

    Viviano, J. D., Simpson, B., Dutil, F., Bengio, Y., & Cohen, J. P. (2020).
    Saliency is a Possible Red Herring When Diagnosing Poor Generalization.
    International Conference on Learning Representations (ICLR).
    https://arxiv.org/abs/1910.00199
    """

    def __init__(self,
                 d1, d1_target,
                 d2, d2_target,
                 ratio=0.5,
                 mode="train",
                 seed=0,
                 nsamples=None,
                 splits=[0.5, 0.25, 0.25],
                 verbose=False
                 ):
        super(CovariateDataset, self).__init__()

        self.splits = np.array(splits)
        self.d1 = d1
        self.d1_target = d1_target
        self.d2 = d2
        self.d2_target = d2_target

        assert mode in ['train', 'valid', 'test']
        assert np.sum(self.splits) == 1.0

        np.random.seed(seed)  # Reset the seed so all runs are the same.

        all_imageids = np.concatenate([np.arange(len(self.d1)),
                                       np.arange(len(self.d2))]).astype(int)

        all_idx = np.arange(len(all_imageids)).astype(int)

        all_labels = np.concatenate([d1_target,
                                     d2_target]).astype(int)

        all_site = np.concatenate([np.zeros(len(self.d1)),
                                   np.ones(len(self.d2))]).astype(int)

        idx_sick = all_labels == 1
        n_per_category = np.min([sum(idx_sick[all_site == 0]),
                                 sum(idx_sick[all_site == 1]),
                                 sum(~idx_sick[all_site == 0]),
                                 sum(~idx_sick[all_site == 1])])

        all_csv = pd.concat([d1.csv, d2.csv])
        all_csv['site'] = all_site
        all_csv['label'] = all_labels

        if verbose:
            print("n_per_category={}".format(n_per_category))

        all_0_neg = all_idx[np.where((all_site == 0) & (all_labels == 0))]
        all_0_neg = np.random.choice(all_0_neg, n_per_category, replace=False)
        all_0_pos = all_idx[np.where((all_site == 0) & (all_labels == 1))]
        all_0_pos = np.random.choice(all_0_pos, n_per_category, replace=False)
        all_1_neg = all_idx[np.where((all_site == 1) & (all_labels == 0))]
        all_1_neg = np.random.choice(all_1_neg, n_per_category, replace=False)
        all_1_pos = all_idx[np.where((all_site == 1) & (all_labels == 1))]
        all_1_pos = np.random.choice(all_1_pos, n_per_category, replace=False)

        # TRAIN
        train_0_neg = np.random.choice(
            all_0_neg, int(n_per_category * ratio * splits[0] * 2), replace=False)
        train_0_pos = np.random.choice(
            all_0_pos, int(n_per_category * (1 - ratio) * splits[0] * 2), replace=False)
        train_1_neg = np.random.choice(
            all_1_neg, int(n_per_category * (1 - ratio) * splits[0] * 2), replace=False)
        train_1_pos = np.random.choice(
            all_1_pos, int(n_per_category * ratio * splits[0] * 2), replace=False)

        # REDUCE POST-TRAIN
        all_0_neg = np.setdiff1d(all_0_neg, train_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, train_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, train_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, train_1_pos)

        if verbose:
            print("TRAIN (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                ratio,
                len(train_0_neg) + len(train_1_neg),
                len(train_0_pos) + len(train_1_pos),
                len(train_0_pos),
                len(train_0_neg),
                len(train_1_pos),
                len(train_1_neg)))

        # VALID
        valid_0_neg = np.random.choice(
            all_0_neg, int(n_per_category * (1 - ratio) * splits[1] * 2), replace=False)
        valid_0_pos = np.random.choice(
            all_0_pos, int(n_per_category * ratio * splits[1] * 2), replace=False)
        valid_1_neg = np.random.choice(
            all_1_neg, int(n_per_category * ratio * splits[1] * 2), replace=False)
        valid_1_pos = np.random.choice(
            all_1_pos, int(n_per_category * (1 - ratio) * splits[1] * 2), replace=False)

        # REDUCE POST-VALID
        all_0_neg = np.setdiff1d(all_0_neg, valid_0_neg)
        all_0_pos = np.setdiff1d(all_0_pos, valid_0_pos)
        all_1_neg = np.setdiff1d(all_1_neg, valid_1_neg)
        all_1_pos = np.setdiff1d(all_1_pos, valid_1_pos)

        if verbose:
            print("VALID (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                1 - ratio,
                len(valid_0_neg) + len(valid_1_neg),
                len(valid_0_pos) + len(valid_1_pos),
                len(valid_0_pos),
                len(valid_0_neg),
                len(valid_1_pos),
                len(valid_1_neg)))

        # TEST
        test_0_neg = all_0_neg
        test_0_pos = all_0_pos
        test_1_neg = all_1_neg
        test_1_pos = all_1_pos

        if verbose:
            print("TEST (ratio={:.2}): neg={}, pos={}, d1_pos/neg={}/{}, d2_pos/neg={}/{}".format(
                1 - ratio,
                len(test_0_neg) + len(test_1_neg),
                len(test_0_pos) + len(test_1_pos),
                len(test_0_pos),
                len(test_0_neg),
                len(test_1_pos),
                len(test_1_neg)))

        def _reduce_nsamples(nsamples, a, b, c, d):
            if nsamples:
                a = a[:int(np.floor(nsamples / 4))]
                b = b[:int(np.ceil(nsamples / 4))]
                c = c[:int(np.ceil(nsamples / 4))]
                d = d[:int(np.floor(nsamples / 4))]

            return (a, b, c, d)

        if mode == "train":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, train_0_neg, train_0_pos, train_1_neg, train_1_pos)
        elif mode == "valid":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, valid_0_neg, valid_0_pos, valid_1_neg, valid_1_pos)
        elif mode == "test":
            (a, b, c, d) = _reduce_nsamples(
                nsamples, test_0_neg, test_0_pos, test_1_neg, test_1_pos)
        else:
            raise Exception("unknown mode")

        self.select_idx = np.concatenate([a, b, c, d])
        self.imageids = all_imageids[self.select_idx]
        self.pathologies = ["Custom"]
        self.labels = all_labels[self.select_idx].reshape(-1, 1)
        self.site = all_site[self.select_idx]
        self.csv = all_csv.iloc[self.select_idx]

    def __repr__(self):
        pprint.pprint(self.totals())
        return self.__class__.__name__ + " num_samples={}".format(len(self))

    def __len__(self):
        return len(self.imageids)

    def __getitem__(self, idx):

        if self.site[idx] == 0:
            dataset = self.d1
        else:
            dataset = self.d2

        sample = dataset[self.imageids[idx]]

        # Replace the labels with the specific label we focus on
        sample["lab-old"] = sample["lab"]
        sample["lab"] = self.labels[idx]
        sample["site"] = self.site[idx]

        return sample
