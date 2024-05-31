import os
import torch
import pandas as pd
import numpy as np
from typing import List
from PIL import Image
from collections import defaultdict
from transformers import AutoTokenizer
from torch.utils.data import Dataset
# from skimage.io import imread
import cv2
from cv2 import imread
from dcxfm.datasets.cxr_datasets import MIMIC_Dataset, CheX_Dataset, BRAX_Dataset, Openi_Dataset, \
    NIH_Dataset, PC_Dataset, normalize, apply_transforms, BaseDataset, _process_medclip_labels
from dcxfm.utils.constants import CHEXPERT_COMPETITION_TASKS, CHEXPERT_TASKS, CHEXPERT_UNCERTAIN_MAPPINGS
import ipdb


class SDMP_Pretrain_Dataset(BaseDataset):
    '''
    The main dataset class for MedCLIP pretraining.
    '''

    def __init__(self,
                 split: str,
                 bert_type: str = "microsoft/BiomedVLP-CXR-BERT-general",
                 dataset_dir: str = "/home/r15user2/Documents/CXRSeg/cxr_data",
                 dataset_list: List[str] = ["mimic-cxr",
                                            "chexpert", "brax", "openi", "nih", "pc"],
                 text_type: str = "label",
                 sentence_label_csvpath: str = "",
                 data_pct: float = 1.,
                 transform: object = None):

        # define pathologies as the chexpert tasks.
        # because we only have chexbert to extract labels from reports.
        self.pathologies = CHEXPERT_TASKS
        self.pathologies = sorted(self.pathologies)

        super().__init__()

        assert split in ["train", "val"], "split must be either train or val"
        self.tokenizer = AutoTokenizer.from_pretrained(
            bert_type, trust_remote_code=True)
        self.sentence_label_csvpath = sentence_label_csvpath
        self.sentence_label = pd.read_csv(self.sentence_label_csvpath)
        # convert csv into a dictionary
        self._preprocess_sentence_label()
        self._build_prompt_sentence()
        custom_mapping = {}
        self.custom_mapping = custom_mapping

        self.dataset_dir = dataset_dir

        all_datasets = []
        if "mimic-cxr" in dataset_list:
            if split == "train":
                mimic_dataset = MIMIC_Dataset(
                    imgpath=os.path.join(
                        self.dataset_dir, "mimic_data/2.0.0/files"),
                    csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/MIMIC-CXR/mimic-cxr-train-meta.csv"),
                    pathologies=self.pathologies,
                    medclip_format=True,
                    sent_info_dict=self.sent_info_dict,
                    data_pct=data_pct,
                    text_type=text_type,
                    transform=transform
                )
            elif split == "val":
                mimic_dataset = MIMIC_Dataset(
                    imgpath=os.path.join(
                        self.dataset_dir, "mimic_data/2.0.0/files"),
                    csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/MIMIC-CXR/mimic-cxr-val-test-meta.csv"),
                    pathologies=self.pathologies,
                    medclip_format=True,
                    sent_info_dict=self.sent_info_dict,
                    data_pct=data_pct,
                    text_type=text_type,
                    transform=transform
                )
            all_datasets.append(mimic_dataset)

        # TODO: fix other datasets with data_pct
        if "chexpert" in dataset_list:
            if split == "train":
                chex_dataset = CheX_Dataset(
                    imgpath=os.path.join(self.dataset_dir, "CheXpert-v1.0"),
                    csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/CheXpert/chexpert_train.csv"),
                    pathologies=self.pathologies,
                    medclip_format=True,
                    sentence_label=self.sentence_label,
                    prompt_sentence_label=self.prompt_sentence_label,
                    data_pct=data_pct,
                    sent_info_dict=self.sent_info_dict,
                    transform=transform
                )
            elif split == "val":
                chex_dataset = CheX_Dataset(
                    imgpath=os.path.join(self.dataset_dir, "CheXpert-v1.0"),
                    csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/CheXpert/chexpert_val.csv"),
                    pathologies=self.pathologies,
                    medclip_format=True,
                    sentence_label=self.sentence_label,
                    prompt_sentence_label=self.prompt_sentence_label,
                    sent_info_dict=self.sent_info_dict,
                    data_pct=data_pct,
                    transform=transform
                )
            all_datasets.append(chex_dataset)
        if "brax" in dataset_list:
            # brax doesn't have valid set
            if split == "train":
                brax_dataset = BRAX_Dataset(
                    imgpath=os.path.join(self.dataset_dir, "brax/1.1.0"),
                    csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/BRAX/brax_train.csv"),
                    pathologies=self.pathologies,
                    medclip_format=True,
                    sentence_label=self.sentence_label,
                    prompt_sentence_label=self.prompt_sentence_label,
                    sent_info_dict=self.sent_info_dict,
                    data_pct=data_pct,
                    transform=transform
                )
            elif split == "val":
                brax_dataset = BRAX_Dataset(
                    imgpath=os.path.join(self.dataset_dir, "brax/1.1.0"),
                    csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/BRAX/brax_val.csv"),
                    pathologies=self.pathologies,
                    medclip_format=True,
                    sentence_label=self.sentence_label,
                    prompt_sentence_label=self.prompt_sentence_label,
                    sent_info_dict=self.sent_info_dict,
                    data_pct=data_pct,
                    transform=transform
                )
            all_datasets.append(brax_dataset)
        if "openi" in dataset_list:
            if split == "train":
                openi_dataset = Openi_Dataset(
                    imgpath=os.path.join(
                        self.dataset_dir, "chest-xrays-indiana-university/images/images_normalized"),
                    projection_csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/OpenI/iu_train.csv"),
                    report_csvpath=os.path.join(
                        self.dataset_dir, "chest-xrays-indiana-university/indiana_reports.csv"),
                    medclip_format=True,
                    sent_info_dict=self.sent_info_dict,
                    transform=transform
                )
            elif split == "val":
                openi_dataset = Openi_Dataset(
                    imgpath=os.path.join(
                        self.dataset_dir, "chest-xrays-indiana-university/images/images_normalized"),
                    projection_csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/OpenI/iu_val.csv"),
                    report_csvpath=os.path.join(
                        self.dataset_dir, "chest-xrays-indiana-university/indiana_reports.csv"),
                    medclip_format=True,
                    sent_info_dict=self.sent_info_dict,
                    transform=transform
                )
            all_datasets.append(openi_dataset)
        if "nih" in dataset_list:
            if split == "train":
                nih_dataset = NIH_Dataset(
                    imgpath=os.path.join(self.dataset_dir, "NIH/images"),
                    csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/NIH/NIH_train.csv"),
                    pathologies=self.pathologies,
                    medclip_format=True,
                    sentence_label=self.sentence_label,
                    prompt_sentence_label=self.prompt_sentence_label,
                    sent_info_dict=self.sent_info_dict,
                    data_pct=data_pct,
                    transform=transform
                )
            elif split == "val":
                nih_dataset = NIH_Dataset(
                    imgpath=os.path.join(self.dataset_dir, "NIH/images"),
                    csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/NIH/NIH_val.csv"),
                    pathologies=self.pathologies,
                    medclip_format=True,
                    sentence_label=self.sentence_label,
                    prompt_sentence_label=self.prompt_sentence_label,
                    sent_info_dict=self.sent_info_dict,
                    data_pct=data_pct,
                    transform=transform
                )
            all_datasets.append(nih_dataset)
        if "padchest" in dataset_list:
            if split == "train":
                padchest_dataset = PC_Dataset(
                    imgpath=os.path.join(
                        self.dataset_dir, "PadChest/BIMCV-PadChest-FULL/images"),
                    csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/PadChest/padchest_train.csv"),
                    pathologies=self.pathologies,
                    medclip_format=True,
                    sent_info_dict=self.sent_info_dict,
                    data_pct=data_pct,
                    transform=transform
                )
            elif split == "val":
                padchest_dataset = PC_Dataset(
                    imgpath=os.path.join(
                        self.dataset_dir, "PadChest/BIMCV-PadChest-FULL/images"),
                    csvpath=os.path.join(
                        self.dataset_dir, "preprocessed_csv/PadChest/padchest_val.csv"),
                    pathologies=self.pathologies,
                    medclip_format=True,
                    sent_info_dict=self.sent_info_dict,
                    data_pct=data_pct,
                    transform=transform
                )
            all_datasets.append(padchest_dataset)

        self.datasets = all_datasets
        self.length = 0
        self.which_dataset = np.zeros(0)
        self.offset = np.zeros(0)
        currentoffset = 0
        total_len = sum([len(x) for x in self.datasets])
        self.labels = np.zeros([total_len, len(self.pathologies)])
        self.pathology_indices = []
        self.report_labels = np.zeros([total_len, len(self.pathologies)])
        self.num_patients = 0
        for i, dataset in enumerate(self.datasets):
            self.which_dataset = np.concatenate(
                [self.which_dataset, np.zeros(len(dataset)) + i])
            self.length += len(dataset)
            self.num_patients += dataset.num_patients
            self.offset = np.concatenate(
                [self.offset, np.zeros(len(dataset)) + currentoffset])
            currentoffset += len(dataset)

            indices = []
            for pathology in dataset.pathologies:
                if pathology in self.custom_mapping:
                    pathology = self.custom_mapping[pathology]
                p_index = self.pathologies.index(pathology)
                indices.append(p_index)
                self.labels[currentoffset - len(dataset):currentoffset, p_index] = \
                    dataset.labels[:, dataset.pathologies.index(pathology)]
            self.pathology_indices.append(indices)

        self.which_dataset = self.which_dataset.astype(int)

    def string(self):
        s = self.__class__.__name__ + \
            " num_patients={} num_samples={}\n".format(
                self.num_patients, len(self))
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

    def _preprocess_sentence_label(self):
        self.sentence_label = self.sentence_label.drop_duplicates(
            subset='Report Impression')
        self.sentence_label = self.sentence_label[self.sentence_label['Report Impression'].map(
            len) > 2].reset_index(drop=True)
        self.sentence_label['report'] = self.sentence_label['Report Impression']
        self.sentence_label = self.sentence_label.drop(
            'Report Impression', axis=1)
        self.sentence_label.fillna(0, inplace=True)
        # replace nan labels
        uncertain_mask = {k: -1 for k in CHEXPERT_COMPETITION_TASKS}
        self.sentence_label = self.sentence_label.replace(
            uncertain_mask, CHEXPERT_UNCERTAIN_MAPPINGS)

        labels = self.sentence_label[self.pathologies].values
        labels = _process_medclip_labels(labels)

        # tokenize all sentences
        # sentence_dataset = Dataset_hg.from_pandas(self.sentence_label)
        # self.tokenized_dataset = sentence_dataset.map(tokenize_function)
        # keys = self.tokenized_dataset['report']
        # values = zip(labels.tolist(), self.tokenized_dataset["input_ids"], self.tokenized_dataset["attention_mask"])
        # self.sent_info_dict = dict(zip(keys, values))

        keys = self.sentence_label['report']
        self.sent_info_dict = dict(zip(keys, labels.tolist()))

    def _build_prompt_sentence(self, n=500):
        # print('build prompt sentences.')
        sentence_label = self.sentence_label.copy()
        new_sent_list = []
        num_sent = []
        for task in CHEXPERT_TASKS:
            sub_sent_df = sentence_label.loc[sentence_label[task] == 1]
            if len(sub_sent_df) < n:
                new_sent_list.append(sub_sent_df)
            else:
                new_sent_list.append(sub_sent_df.sample(n))
            num_sent.append(min(len(sub_sent_df), n))

        new_sent_df = pd.concat(new_sent_list, axis=0)
        new_sent_df = new_sent_df.drop_duplicates()
        new_sent_df.reset_index(inplace=True, drop=True)
        new_sent_df[CHEXPERT_TASKS] = _process_medclip_labels(
            new_sent_df[CHEXPERT_TASKS].values)
        self.prompt_sentence_label = new_sent_df

    def tokenize_function(self, report, max_length=77):
        tokenized_text = self.tokenizer(
            report,
            add_special_tokens=True,
            padding='max_length',
            max_length=max_length,
            truncation=True,
            return_tensors="pt")
        return tokenized_text

    def __getitem__(self, idx):
        dataset_idx = int(self.which_dataset[idx])
        item = self.datasets[dataset_idx][idx - int(self.offset[idx])]
        item["dataset_idx"] = dataset_idx
        item["lab"] = self.labels[idx]
        item["source"] = self.which_dataset[idx]

        # only keep the first sentence
        # FIXME: we have several strategies here
        # First: only keep the first sentence
        # Concat: concatenate all sentences
        # Multiple: keep all sentences

        tokenized_output = self.tokenize_function(item["report"])
        item["input_ids"] = tokenized_output['input_ids']
        item["attention_mask"] = tokenized_output['attention_mask']

        report_labels = np.zeros(len(self.pathologies))
        report_labels[self.pathology_indices[dataset_idx]] = item["report_lab"]
        item["report_lab"] = report_labels

        return item


class ImageTextContrastiveCollator:
    def __init__(self,
                 use_eda=False):
        '''Args:
        use_EDA: easy data augmentation from text augment
        '''
        pass
    
    def __call__(self, batch):
        inputs = defaultdict(list)
        # report_aug_list = []
        for data in batch:
            # inputs['pixel_values'].append(data["img"])
            inputs['img_global'].append(data["img_global"])
            if "img_local" in data:
                inputs['img_local'].append(data["img_local"])
            inputs['report'].append(data["report"])
            inputs['input_ids'].append(data["input_ids"].long())
            inputs['attention_mask'].append(data["attention_mask"].long())
            inputs['img_labels'].append(data["lab"])
            inputs['text_labels'].append(data["report_lab"])

        # inputs['pixel_values'] = torch.stack(inputs['pixel_values'])
        inputs['img_global'] = torch.stack(inputs['img_global'])
        inputs['img_labels'] = torch.tensor(
            np.stack(inputs['img_labels']).astype(float))
        inputs['text_labels'] = torch.tensor(
            np.stack(inputs['text_labels']).astype(float))
        if len(inputs['img_local']) > 0:
            inputs['img_local'] = torch.stack(inputs['img_local'])

        return inputs


class ZeroShotImageDataset(Dataset):
    def __init__(self,
                 imgdir_list=[""],
                 csv_list=['chexpert-5x200-meta.csv'],
                 class_names=None,
                 imgtransform=None,
                 path_col="Path",
                 ) -> None:
        '''support data list in mimic-5x200, chexpert-5x200, rsna-balanced-test, covid-test
        args:
            imgtransform: a torchvision transform
            cls_prompts: a dict of prompt sentences, cls:[sent1, sent2, ..],
        '''
        super().__init__()

        self.transform = imgtransform
        self.class_names = class_names

        df_list = []
        for imgdir, filename in zip(imgdir_list, csv_list):
            df = pd.read_csv(filename)
            df["Path"] = df[path_col].apply(lambda x: os.path.join(imgdir, x))
            df_list.append(df)
        self.df = pd.concat(df_list, axis=0).reset_index(drop=True)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        img = imread(row.Path, cv2.IMREAD_GRAYSCALE)
        img = normalize(img, maxval=255, reshape=True)
        sample = dict()
        sample["img"] = img
        sample = apply_transforms(sample, self.transform)
        sample["lab"] = pd.DataFrame(row[self.class_names]).transpose()
        return sample

    def __len__(self):
        return len(self.df)


def process_class_prompts(cls_prompts, tokenizer):
    cls_prompt_inputs = defaultdict()
    for k, v in cls_prompts.items():
        text_inputs = tokenizer(v,
                                add_special_tokens=True,
                                padding='max_length',
                                max_length=77,
                                truncation=True,
                                return_tensors="pt")
        cls_prompt_inputs[k] = text_inputs
    return cls_prompt_inputs


class ZeroShotImageCollator:
    def __init__(self, mode,
                 ):
        assert mode in ['multiclass', 'multilabel', 'binary']
        self.mode = mode

    def __call__(self, batch):
        inputs = defaultdict(list)
        for data in batch:
            inputs['pixel_values'].append(data["img"])
            inputs['labels'].append(data["lab"])

        inputs['labels'] = pd.concat(inputs['labels']).astype(int).values
        if self.mode in ['multiclass', 'binary']:
            inputs['labels'] = torch.tensor(
                inputs['labels'].argmax(1), dtype=int)
        else:
            inputs['labels'] = torch.tensor(inputs['labels'], dtype=float)

        inputs['pixel_values'] = torch.stack(inputs['pixel_values'])

        return {
            'pixel_values': inputs['pixel_values'],
            'labels': inputs['labels']
        }
        # if self.neg_prompts is not None:
        #     return {
        #         'pixel_values': inputs['pixel_values'],
        #         'pos_prompt_inputs': self.pos_prompts,
        #         'neg_prompt_inputs': self.neg_prompts,
        #         'labels': inputs['labels'],
        #     }
        # else:
        #     return {
        #         'pixel_values': inputs['pixel_values'],
        #         'pos_prompt_inputs': self.pos_prompts,
        #         'labels': inputs['labels'],
        #     }


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from dcxfm.datasets.transforms import get_dino_transforms
    # transform = get_transforms(is_train=True, imagesize=512)
    transform = get_dino_transforms(
        is_train=True, global_crops_number=1, local_crops_number=0)
    train_dataset = MedCLIP_Pretrain_Dataset(
        split="val",
        bert_type="microsoft/BiomedVLP-CXR-BERT-general",
        dataset_dir="/disk1/fywang/CXR_dataset",
        dataset_list=["mimic-cxr"],
        sentence_label_csvpath="/disk1/fywang/CXR_dataset/mask/reports/labeled_reports_mimic-cxr_iu-xray_padchest.csv",
        transform=transform
    )

    train_collate_fn = ImageTextContrastiveCollator()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=4,
        collate_fn=train_collate_fn,
        shuffle=True,
        pin_memory=True,
        num_workers=1,
    )

    for batch in train_dataloader:
        print(batch['pixel_values'].shape)
        print(batch['input_ids'].shape)
        print(batch['attention_mask'].shape)
        print(batch['img_labels'].shape)
        print(batch['text_labels'].shape)
        break
    # sample = train_dataset[0]

    '''
    The following code snippet is used to test zero-shot dataloader.
    '''

    # val_data = ZeroShotImageDataset(
    #     imgdir_list=["/home/r15user2/Documents/CXRSeg/cxr_data/mimic_data",
    #                  "/home/r15user2/Documents/CXRSeg/cxr_data"],
    #     csv_list=['/home/r15user2/Documents/CXRSeg/cxr_data/mask/preprocessed_csv/mimic-cxr-5x200-val.csv',
    #               '/home/r15user2/Documents/CXRSeg/cxr_data/mask/preprocessed_csv/chexpert_5x200.csv'],
    #     class_names=CHEXPERT_COMPETITION_TASKS,
    #     imgtransform=transform
    # )

    # from cxrseg.utils.prompts import generate_class_prompts
    # prompts = generate_class_prompts(CHEXPERT_COMPETITION_TASKS)
    # val_collate_fn = ZeroShotImageCollator(
    #     mode="multiclass",
    #     bert_type="microsoft/BiomedVLP-CXR-BERT-specialized",
    #     cls_prompts=prompts
    # )
    # eval_dataloader = DataLoader(
    #     val_data,
    #     batch_size=32,
    #     collate_fn=val_collate_fn,
    #     shuffle=False,
    #     pin_memory=True,
    #     num_workers=1,
    # )

    # print(len(eval_dataloader))
    # for batch in eval_dataloader:
    #     print(batch['pixel_values'].shape)
    #     print(batch['labels'].shape)
    #     print(batch['prompt_inputs']['Atelectasis']['input_ids'].shape)
    #     print(batch['prompt_inputs']['Atelectasis']['attention_mask'].shape)
    #     break
    # ipdb.set_trace()
