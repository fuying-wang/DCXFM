from typing import Tuple
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


class DataAugmentationDINO(object):
    ''' This class is used to create multi-crops commonly used in SSL.'''

    def __init__(self,
                 global_crops_scale: Tuple,
                 local_crops_scale: Tuple,
                 global_crops_number: int,
                 local_crops_number: int,
                 global_crops_size: int,
                 local_crops_size: int,
                 resize_size: int,
                 mean: float,
                 std: float):

        # first global crop
        self.global_crops_number = global_crops_number
        self.global_transfo = []
        for _ in range(global_crops_number):
            self.global_transfo.append(A.Compose([
                A.LongestMaxSize(max_size=resize_size,
                                 interpolation=cv2.INTER_AREA),
                A.ColorJitter(hue=0.0),
                A.GaussNoise(),
                A.Affine(mode=cv2.BORDER_CONSTANT, cval=0,
                         translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
                A.PadIfNeeded(min_height=resize_size, min_width=resize_size,
                              border_mode=cv2.BORDER_CONSTANT),
                A.RandomResizedCrop(
                    global_crops_size, global_crops_size,
                    scale=global_crops_scale,
                    interpolation=cv2.INTER_AREA),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]))
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfos = []
        for _ in range(local_crops_number):
            self.local_transfos.append(A.Compose([
                A.LongestMaxSize(max_size=resize_size,
                                 interpolation=cv2.INTER_AREA),
                A.ColorJitter(hue=0.0),
                A.GaussNoise(),
                A.Affine(mode=cv2.BORDER_CONSTANT, cval=0,
                         translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
                A.PadIfNeeded(min_height=resize_size, min_width=resize_size,
                              border_mode=cv2.BORDER_CONSTANT),
                # here is the only difference
                A.RandomResizedCrop(
                    local_crops_size, local_crops_size,
                    scale=local_crops_scale,
                    interpolation=cv2.INTER_AREA),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ]))

    def __call__(self, image):
        output = dict()

        global_crops = []
        for i in range(self.global_crops_number):
            global_crops.append(self.global_transfo[i](image=image)["image"])
        output["global_crops"] = torch.stack(global_crops)

        if self.local_crops_number > 0:
            local_crops = []
            for i in range(self.local_crops_number):
                local_crops.append(
                    self.local_transfos[i](image=image)["image"])
            output["local_crops"] = torch.stack(local_crops)

        return output


def get_dino_transforms(is_train: bool = False,
                        global_crops_scale: Tuple = (0.6, 1.),
                        local_crops_scale: Tuple = (0.3, 0.5),
                        global_crops_number: int = 2,
                        local_crops_number: int = 4,
                        resize_size: int = 512,
                        global_crops_size: int = 512,
                        local_crops_size: int = 224,
                        mean=0,
                        std=1.):
    if is_train:
        transform = DataAugmentationDINO(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
            resize_size=resize_size,
            global_crops_number=global_crops_number,
            local_crops_number=local_crops_number,
            mean=mean,
            std=std
        )

    else:
        transform = DataAugmentationDINO(
            global_crops_scale=global_crops_scale,
            local_crops_scale=local_crops_scale,
            global_crops_size=global_crops_size,
            local_crops_size=local_crops_size,
            resize_size=resize_size,
            global_crops_number=global_crops_number,
            local_crops_number=local_crops_number,
            mean=mean,
            std=std
        )

    return transform


def get_transforms(is_train=True, imagesize=1024, mean=0, std=1.):
    '''
    Usage: train medclip
    '''
    if is_train:
        transform = A.Compose([
            A.LongestMaxSize(max_size=imagesize, interpolation=cv2.INTER_AREA),
            A.ColorJitter(hue=0.0),
            A.GaussNoise(),
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0,
                     translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            A.PadIfNeeded(min_height=imagesize, min_width=imagesize,
                          border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.LongestMaxSize(max_size=imagesize, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=imagesize, min_width=imagesize,
                          border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    return transform


def get_bbox_transforms(is_train=True, IMAGE_INPUT_SIZE=1024, mean=0, std=1):
    '''
    Usage: MedSAM inference
    '''
    if is_train:
        # use albumentations for Compose and transforms
        # augmentations are applied with prob=0.5
        # since Affine translates and rotates the image, we also have to do the same with the bounding boxes, hence the bbox_params arugment
        transform = A.Compose(
            [
                # we want the long edge of the image to be resized to IMAGE_INPUT_SIZE, and the short edge of the image to be padded to IMAGE_INPUT_SIZE on both sides,
                # such that the aspect ratio of the images are kept, while getting images of uniform size (IMAGE_INPUT_SIZE x IMAGE_INPUT_SIZE)
                # LongestMaxSize: resizes the longer edge to IMAGE_INPUT_SIZE while maintaining the aspect ratio
                # INTER_AREA works best for shrinking images
                A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE,
                                 interpolation=cv2.INTER_AREA),
                A.ColorJitter(hue=0.0),
                A.GaussNoise(),
                # randomly (by default prob=0.5) translate and rotate image
                # mode and cval specify that black pixels are used to fill in newly created pixels
                # translate between -2% and 2% of the image height/width, rotate between -2 and 2 degrees
                A.Affine(mode=cv2.BORDER_CONSTANT, cval=0,
                         translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
                # PadIfNeeded: pads both sides of the shorter edge with 0's (black pixels)
                A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE,
                              min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=mean, std=std),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
        )
    else:
        transform = A.Compose(
            [
                A.LongestMaxSize(max_size=IMAGE_INPUT_SIZE,
                                 interpolation=cv2.INTER_AREA),
                A.PadIfNeeded(min_height=IMAGE_INPUT_SIZE,
                              min_width=IMAGE_INPUT_SIZE, border_mode=cv2.BORDER_CONSTANT),
                A.Normalize(mean=mean, std=std),
                ToTensorV2(),
            ], bbox_params=A.BboxParams(format="pascal_voc", label_fields=['class_labels'])
        )

    return transform


def get_medsam_transforms(is_train: bool = False, imagesize: int = 1024):
    '''
    Usage: Finetune MedSAM
    '''
    if is_train:
        transform = A.Compose([
            A.LongestMaxSize(max_size=imagesize, interpolation=cv2.INTER_AREA),
            A.Affine(mode=cv2.BORDER_CONSTANT, cval=0,
                     translate_percent=(-0.02, 0.02), rotate=(-2, 2)),
            A.PadIfNeeded(min_height=imagesize, min_width=imagesize,
                          border_mode=cv2.BORDER_CONSTANT),
            # The mean and std are used to make sure the consistency with original MedSAM
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ])
    else:
        transform = A.Compose([
            A.LongestMaxSize(max_size=imagesize, interpolation=cv2.INTER_AREA),
            A.PadIfNeeded(min_height=imagesize, min_width=imagesize,
                          border_mode=cv2.BORDER_CONSTANT),
            A.Normalize(mean=0.0, std=1.0),
            ToTensorV2(),
        ])

    return transform
