import os
from typing import List
import ipdb
from torch.utils.data import DataLoader
from lightning import LightningDataModule
from dcxfm.datasets.medclip_datasets import MedCLIP_Pretrain_Dataset
from dcxfm.datasets.transforms import get_dino_transforms, get_transforms
from dcxfm.datasets.medclip_datasets import ImageTextContrastiveCollator, ZeroShotImageDataset, ZeroShotImageCollator
from dcxfm.utils.constants import CHEXPERT_COMPETITION_TASKS


class SDMPDataModule(LightningDataModule):
    def __init__(self, dataset_dir: str, dataset_list: List,
                 bert_type: str, batch_size: int, num_workers: int, imagesize: int,
                 global_crops_size: int, local_crops_size: int,
                 train_data_pct: float,
                 global_crops_number: int, local_crops_number: int):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_list = dataset_list
        self.bert_type = bert_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        self.imagesize = imagesize
        self.train_data_pct = train_data_pct
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number

    def train_dataloader(self):
        transform = get_dino_transforms(
            resize_size=self.imagesize,
            global_crops_size=self.global_crops_size,
            local_crops_size=self.local_crops_size,
            global_crops_number=self.global_crops_number,
            local_crops_number=self.local_crops_number,
            is_train=True)

        train_dataset = MedCLIP_Pretrain_Dataset(
            split="train",
            bert_type=self.bert_type,
            dataset_dir=self.dataset_dir,
            dataset_list=self.dataset_list,
            sentence_label_csvpath=os.path.join(
                self.dataset_dir, "mask/reports/labeled_reports_mimic-cxr_iu-xray_padchest.csv"),
            data_pct=self.train_data_pct,
            transform=transform
        )

        train_collate_fn = ImageTextContrastiveCollator()
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            collate_fn=train_collate_fn,
            shuffle=True,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        return train_dataloader

    def val_dataloader(self):
        transform = get_dino_transforms(
            resize_size=self.imagesize,
            global_crops_size=self.global_crops_size,
            local_crops_size=self.local_crops_size,
            global_crops_number=self.global_crops_number,
            local_crops_number=self.local_crops_number,
            is_train=False)

        val_dataset = MedCLIP_Pretrain_Dataset(
            split="val",
            bert_type=self.bert_type,
            dataset_dir=self.dataset_dir,
            dataset_list=self.dataset_list,
            sentence_label_csvpath=os.path.join(
                self.dataset_dir, "mask/reports/labeled_reports_mimic-cxr_iu-xray_padchest.csv"),
            data_pct=self.train_data_pct,
            transform=transform
        )

        val_collate_fn = ImageTextContrastiveCollator()
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            collate_fn=val_collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        return val_dataloader

    def test_dataloader(self):
        # Here the test only includes the zero-shot classification.
        # FIXME: More evaluation tasks need to be filled ...
        transform = get_transforms(is_train=False, imagesize=self.imagesize)
        test_data = ZeroShotImageDataset(
            imgdir_list=[os.path.join(self.dataset_dir, "mimic_data"),
                         os.path.join(self.dataset_dir)],
            csv_list=[os.path.join(self.dataset_dir, 'preprocessed_csv/MIMIC-CXR/mimic-cxr-5x200-val.csv'),
                      os.path.join(
                          self.dataset_dir, 'preprocessed_csv/CheXpert/chexpert_5x200.csv')
                      ],
            class_names=CHEXPERT_COMPETITION_TASKS,
            imgtransform=transform
        )

        # pos_prompts = generate_class_prompts(
        #     CHEXPERT_COMPETITION_TASKS, mode="pos")
        # neg_prompts = generate_class_prompts(
        #     CHEXPERT_COMPETITION_TASKS, mode="neg")
        test_collate_fn = ZeroShotImageCollator(
            mode="multiclass",
            # bert_type=self.bert_type,
            # pos_prompts=pos_prompts,
            # neg_prompts=neg_prompts
        )
        testloader = DataLoader(
            test_data,
            batch_size=self.batch_size,
            collate_fn=test_collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=self.num_workers,
        )

        return testloader


if __name__ == "__main__":
    dm = SDMPDataModule(
        dataset_dir="/disk1/fywang/CXR_dataset",
        dataset_list=["mimic-cxr"],
        bert_type="microsoft/BiomedVLP-CXR-BERT-general",
        batch_size=4,
        num_workers=4,
        imagesize=512,
        train_data_pct=0.01,
        global_crops_number=1,
        local_crops_number=0
    )

    dataloader = dm.train_dataloader()

    for batch in dataloader:
        break

    ipdb.set_trace()