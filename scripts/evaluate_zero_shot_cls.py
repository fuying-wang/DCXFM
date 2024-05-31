import os
import ipdb
import random
import numpy as np
import pandas as pd
from argparse import ArgumentParser, Namespace
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, matthews_corrcoef, f1_score, accuracy_score, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from functools import reduce
from typing import Dict
from cxrseg.modeling.our_medclip.silc_module import SILCModule
from cxrseg.modeling.our_medclip.gloria_module import GLoRIAModule
from cxrseg.datasets.medclip_datasets import ZeroShotImageDataset, ZeroShotImageCollator
from cxrseg.datasets.transforms import get_transforms
from cxrseg.utils.constants import CHEXPERT_COMPETITION_TASKS, NIH_TASKS, \
    PADCHEST_SEEN_CLASSES, PADCHEST_UNSEEN_CLASSES
from cxrseg.utils.prompts import generate_class_prompts
from cxrseg.utils.evaluation_utils import bootstrap_metric, create_ci_record


# !TODO: add a validation set for threshold chosing.
''''
CUDA_VISIBLE_DEVICES=5 python evaluate_zero_shot_cls.py --model_name mgca_vit --prompt_style medclip_cnn \
    --dataset_list padchest nih --batch_size 128 --use_negative_prompt

CUDA_VISIBLE_DEVICES=0 python evaluate_zero_shot_cls.py --model_name our_medclip_s2 --prompt_style xplainer \
--use_negative_prompt \
--dataset_list mimic_5x200 chexpert_5x200 chexpert padchest --batch_size 128 \
--ckpt_path /disk1/fywang/CXRSEG/output/2024-05-04_16-38-32/model_0000199.pth
'''

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_zero_shot_cls_dataloader(dataset_dir: str,
                                    dataset_name: str,
                                    batch_size: int,
                                    # tokenizer: str,
                                    use_negative_prompt: bool = True,
                                    prompt_style: str = "xplainer",
                                    num_workers: int = 4,
                                    imagesize: int = 384,
                                    mean: float = 0.,
                                    std: float = 1.):

    transform = get_transforms(
        is_train=False, imagesize=imagesize, mean=mean, std=std)
    if dataset_name == "mimic_5x200":
        test_data = ZeroShotImageDataset(
            imgdir_list=[os.path.join(dataset_dir, "mimic_data")],
            csv_list=[os.path.join(
                dataset_dir, 'preprocessed_csv/MIMIC-CXR/mimic-cxr-5x200-val.csv')],
            class_names=CHEXPERT_COMPETITION_TASKS,
            imgtransform=transform
        )
        pos_prompts = generate_class_prompts(
            prompt_file=os.path.join(
                dataset_dir, "preprocessed_csv/CheXpert/chexpert_prompts.json"),
            class_names=CHEXPERT_COMPETITION_TASKS,
            mode="pos",
            prompt_style=prompt_style)
        if use_negative_prompt:
            neg_prompts = generate_class_prompts(
                prompt_file=os.path.join(
                    dataset_dir, "preprocessed_csv/CheXpert/chexpert_prompts.json"),
                class_names=CHEXPERT_COMPETITION_TASKS,
                mode="neg",
                prompt_style=prompt_style)
        test_collate_fn = ZeroShotImageCollator(
            mode="multiclass",
            # tokenizer=tokenizer,
            # pos_prompts=pos_prompts,
            # neg_prompts=neg_prompts
        )
    elif dataset_name == "chexpert_5x200":
        test_data = ZeroShotImageDataset(
            imgdir_list=[os.path.join(dataset_dir)],
            csv_list=[os.path.join(
                dataset_dir, 'preprocessed_csv/CheXpert/chexpert_5x200.csv')],
            class_names=CHEXPERT_COMPETITION_TASKS,
            imgtransform=transform
        )
        pos_prompts = generate_class_prompts(
            prompt_file=os.path.join(
                dataset_dir, "preprocessed_csv/CheXpert/chexpert_prompts.json"),
            class_names=CHEXPERT_COMPETITION_TASKS,
            mode="pos",
            prompt_style=prompt_style)
        if use_negative_prompt:
            neg_prompts = generate_class_prompts(
                prompt_file=os.path.join(
                    dataset_dir, "preprocessed_csv/CheXpert/chexpert_prompts.json"),
                class_names=CHEXPERT_COMPETITION_TASKS,
                mode="neg",
                prompt_style=prompt_style)
        test_collate_fn = ZeroShotImageCollator(
            mode="multiclass",
            # tokenizer=tokenizer,
            # pos_prompts=pos_prompts,
            # neg_prompts=neg_prompts
        )
    elif dataset_name == "chexpert":
        test_data = ZeroShotImageDataset(
            imgdir_list=[os.path.join(dataset_dir, "CheXpert")],
            csv_list=[os.path.join(
                dataset_dir, 'preprocessed_csv/CheXpert/chexpert_test.csv')],
            # csv_list=[os.path.join(
            #     dataset_dir, 'CheXpert/test_labels.csv')],
            class_names=CHEXPERT_COMPETITION_TASKS,
            imgtransform=transform
        )

        val_data = ZeroShotImageDataset(
            imgdir_list=[os.path.join(dataset_dir)],
            csv_list=[os.path.join(
                dataset_dir, 'preprocessed_csv/CheXpert/chexpert_val.csv')],
            class_names=CHEXPERT_COMPETITION_TASKS,
            imgtransform=transform
        )

        val_collate_fn = ZeroShotImageCollator(
            mode="multilabel"
        )

        pos_prompts = generate_class_prompts(
            prompt_file=os.path.join(
                dataset_dir, "preprocessed_csv/CheXpert/chexpert_prompts.json"),
            class_names=CHEXPERT_COMPETITION_TASKS,
            mode="pos",
            prompt_style=prompt_style)
        if use_negative_prompt:
            neg_prompts = generate_class_prompts(
                prompt_file=os.path.join(
                    dataset_dir, "preprocessed_csv/CheXpert/chexpert_prompts.json"),
                class_names=CHEXPERT_COMPETITION_TASKS,
                mode="neg",
                prompt_style=prompt_style)
        test_collate_fn = ZeroShotImageCollator(
            mode="multilabel",
            # tokenizer=tokenizer,
            # pos_prompts=pos_prompts,
            # neg_prompts=neg_prompts
        )
    elif dataset_name == "nih":
        val_data = ZeroShotImageDataset(
            imgdir_list=[os.path.join(dataset_dir, "NIH/images")],
            csv_list=[os.path.join(
                dataset_dir, 'preprocessed_csv/NIH/NIH_val_zero_shot.csv')],
            class_names=NIH_TASKS,
            imgtransform=transform,
        )
        val_collate_fn = ZeroShotImageCollator(
            mode="multilabel"
        )

        test_data = ZeroShotImageDataset(
            imgdir_list=[os.path.join(dataset_dir, "NIH/images")],
            csv_list=[os.path.join(
                dataset_dir, 'preprocessed_csv/NIH/NIH_zero_shot.csv')],
            class_names=NIH_TASKS,
            imgtransform=transform
        )
        pos_prompts = generate_class_prompts(
            prompt_file=os.path.join(
                dataset_dir, "preprocessed_csv/NIH/NIH_prompts.json"),
            class_names=NIH_TASKS,
            mode="pos",
            prompt_style=prompt_style)
        if use_negative_prompt:
            neg_prompts = generate_class_prompts(
                prompt_file=os.path.join(
                    dataset_dir, "preprocessed_csv/NIH/NIH_prompts.json"),
                class_names=NIH_TASKS,
                mode="neg",
                prompt_style=prompt_style)
        test_collate_fn = ZeroShotImageCollator(
            mode="multilabel"
        )

    # elif dataset_name == "openi":
    #     val_data = ZeroShotImageDataset(
    #         imgdir_list=[os.path.join(
    #             dataset_dir, "chest-xrays-indiana-university/images/images_normalized")],
    #         csv_list=[os.path.join(
    #             dataset_dir, 'preprocessed_csv/OpenI/iu_zero_shot.csv')],
    #         class_names=OPENI_TASKS,
    #         imgtransform=transform
    #     )
    #     pos_prompts = generate_class_prompts(OPENI_TASKS, mode="pos")
    #     if use_negative_prompt:
    #         neg_prompts = generate_class_prompts(OPENI_TASKS, mode="neg")
    #     val_collate_fn = ZeroShotImageCollator(
    #         mode="multilabel",
    #         tokenizer=tokenizer,
    #         pos_prompts=pos_prompts,
    #         neg_prompts=neg_prompts
    #     )
    elif dataset_name == "padchest":
        test_data = ZeroShotImageDataset(
            imgdir_list=[os.path.join(
                dataset_dir, "PadChest/BIMCV-PadChest-FULL/images")],
            csv_list=[os.path.join(
                dataset_dir, 'preprocessed_csv/PadChest/padchest_test.csv')],
            class_names=PADCHEST_SEEN_CLASSES + PADCHEST_UNSEEN_CLASSES,
            imgtransform=transform
        )
        pos_prompts = generate_class_prompts(
            prompt_file=os.path.join(
                dataset_dir, "preprocessed_csv/PadChest/padchest_prompts.json"),
            class_names=PADCHEST_SEEN_CLASSES + PADCHEST_UNSEEN_CLASSES,
            mode="pos",
            prompt_style=prompt_style)
        if use_negative_prompt:
            neg_prompts = generate_class_prompts(
                prompt_file=os.path.join(
                    dataset_dir, "preprocessed_csv/PadChest/padchest_prompts.json"),
                class_names=PADCHEST_SEEN_CLASSES + PADCHEST_UNSEEN_CLASSES,
                mode="neg",
                prompt_style=prompt_style)
        test_collate_fn = ZeroShotImageCollator(
            mode="multilabel",
            # tokenizer=tokenizer,
            # pos_prompts=pos_prompts,
            # neg_prompts=neg_prompts
        )
    print(test_data)
    testloader = DataLoader(
        test_data,
        batch_size=batch_size,
        collate_fn=test_collate_fn,
        shuffle=False,
        pin_memory=True,
        num_workers=num_workers,
    )

    if dataset_name in ["nih", "chexpert"]:
        valloader = DataLoader(
            val_data,
            batch_size=batch_size,
            collate_fn=val_collate_fn,
            shuffle=False,
            pin_memory=True,
            num_workers=num_workers,
        )
        return (testloader, valloader), pos_prompts, neg_prompts
    else:
        return testloader, pos_prompts, neg_prompts


def compute_Accs_threshold(gt, pred, threshold, n_class):
    gt_np = gt.copy()
    pred_np = pred.copy()

    Accs = []
    for i in range(n_class):
        pred_np[:, i][pred_np[:, i] >= threshold[i]] = 1
        pred_np[:, i][pred_np[:, i] < threshold[i]] = 0
        Accs.append(accuracy_score(gt_np[:, i], pred_np[:, i]))

    return Accs


def compute_AUCs(gt, pred, n_class):
    AUROCs = []
    gt_np = gt.copy()
    pred_np = pred.copy()
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))

    return AUROCs


def compute_F1s_threshold(gt, pred, threshold, n_class):
    gt_np = gt.copy()
    pred_np = pred.copy()

    F1s = []
    for i in range(n_class):
        pred_np[:, i][pred_np[:, i] >= threshold[i]] = 1
        pred_np[:, i][pred_np[:, i] < threshold[i]] = 0
        F1s.append(f1_score(gt_np[:, i], pred_np[:, i], average='macro'))

    return F1s


def compute_mccs_threshold(gt, pred, threshold, n_class=14):
    gt_np = gt.copy()
    pred_np = pred.copy()
    mccs = []
    for i in range(n_class):
        pred_np[:, i][pred_np[:, i] >= threshold[i]] = 1
        pred_np[:, i][pred_np[:, i] < threshold[i]] = 0
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))

    return mccs


def get_best_thresholds(gt, pred, n_class):
    ''' Select best thresholds for each class
        Matthews correlation coefficient is used as the metric to select the best threshold
    '''
    # get a best threshold for all classes
    gt_np = gt
    pred_np = pred
    select_best_thresholds = []

    for i in tqdm(range(n_class), desc=f"Select best threshold"):
        y_true = gt_np[:, i]
        _, _, probabilities = roc_curve(y_true, pred_np[:, i])
        probabilities = probabilities[1:]
        probabilities.sort()

        metrics_list = []
        for p in probabilities:
            pred_np_ = pred_np.copy()
            y_pred = np.where(pred_np_[:, i] < p, 0, 1)
            mcc = matthews_corrcoef(y_true, y_pred)
            metrics_list.append(mcc)

        best_index = np.argmax(metrics_list)
        best_p = probabilities[best_index]
        select_best_thresholds.append(best_p)

    return select_best_thresholds


def compute_multilabel_metrics(dataset_name, save_dir, gt, pred, class_names, val_gt=None, val_pred=None):

    if dataset_name == "padchest":
        AUROCs = compute_AUCs(gt, pred, n_class=len(class_names))
        auc_df = pd.DataFrame(np.array([AUROCs]), columns=class_names).T
        mean_auc = auc_df.mean(axis=0)
        auc_df.loc["mean"] = mean_auc
        auc_df.to_csv(os.path.join(save_dir, "ori_metrics.csv"), index=True)

        num_samples = len(gt)
        all_aucs = []

        idx = 0
        while True:
            sample_ids = np.random.choice(
                num_samples, size=num_samples, replace=True)
            gt_sample = gt[sample_ids].copy()
            pred_sample = pred[sample_ids].copy()
            # In some case, only one class appears in GT.
            try:
                AUROCs = compute_AUCs(
                    gt_sample, pred_sample, n_class=len(class_names))
                idx += 1
                print(f"Bootstrap AUROCs: idx [{idx} / 1000]")
            except:
                pass
            all_aucs.append(AUROCs)

            if idx == 1000:
                break

        boot_auc_df = pd.DataFrame(np.array(all_aucs), columns=class_names)
        boot_auc_df.to_csv(os.path.join(
            save_dir, f"boot_auc.csv"), index=False)
        auc_records = []
        for task in boot_auc_df.columns:
            auc_records.append(create_ci_record(
                boot_auc_df[task], task, "AUROC"))
        summary_auc_df = pd.DataFrame.from_records(
            auc_records).sort_values(by='name')
        summary_df = summary_auc_df.copy()
        mean_metrics = summary_df.iloc[:, 1:].mean(axis=0)
        summary_df.loc[len(summary_df)] = ["mean"] + mean_metrics.tolist()
        summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)
    else:
        threshold = get_best_thresholds(
            val_gt, val_pred, n_class=len(class_names))
        AUROCs = compute_AUCs(gt, pred, n_class=len(class_names))
        mccs = compute_mccs_threshold(
            gt, pred, threshold, n_class=len(class_names))
        F1s = compute_F1s_threshold(
            gt, pred, threshold, n_class=len(class_names))
        Accs = compute_Accs_threshold(
            gt, pred, threshold, n_class=len(class_names))

        ori_metrics_df = pd.DataFrame(np.array([AUROCs, mccs, F1s, Accs]),
                                      columns=class_names,
                                      index=["AUROC", "MCC", "F1", "ACC"]).T
        mean_metrics = ori_metrics_df.mean(axis=0)
        ori_metrics_df.loc["mean"] = mean_metrics
        ori_metrics_df.to_csv(os.path.join(
            save_dir, "ori_metrics.csv"), index=True)

        num_samples = len(gt)
        all_aucs, all_mccs, all_f1s, all_accs = [], [], [], []
        for idx in tqdm(range(1000), desc="bootstrap"):
            sample_ids = np.random.choice(
                num_samples, size=num_samples, replace=True)
            gt_sample = gt[sample_ids].copy()
            pred_sample = pred[sample_ids].copy()
            AUROCs = compute_AUCs(gt_sample, pred_sample,
                                  n_class=len(class_names))
            mccs = compute_mccs_threshold(
                gt_sample, pred_sample, threshold, n_class=len(class_names))
            # mccs, threshold = compute_mccs(gt, pred, n_class=len(class_names))
            F1s = compute_F1s_threshold(
                gt, pred, threshold, n_class=len(class_names))
            Accs = compute_Accs_threshold(
                gt, pred, threshold, n_class=len(class_names))

            all_aucs.append(AUROCs)
            all_mccs.append(mccs)
            all_f1s.append(F1s)
            all_accs.append(Accs)

        boot_auc_df = pd.DataFrame(np.array(all_aucs), columns=class_names)
        boot_mcc_df = pd.DataFrame(np.array(all_mccs), columns=class_names)
        boot_f1_df = pd.DataFrame(np.array(all_f1s), columns=class_names)
        boot_acc_df = pd.DataFrame(np.array(all_accs), columns=class_names)

        os.makedirs(save_dir, exist_ok=True)
        boot_auc_df.to_csv(os.path.join(
            save_dir, f"boot_auc.csv"), index=False)
        boot_mcc_df.to_csv(os.path.join(
            save_dir, f"boot_mcc.csv"), index=False)
        boot_f1_df.to_csv(os.path.join(save_dir, f"boot_f1.csv"), index=False)
        boot_acc_df.to_csv(os.path.join(
            save_dir, f"boot_acc.csv"), index=False)

        # get 95% confidence intervals
        auc_records = []
        for task in boot_auc_df.columns:
            auc_records.append(create_ci_record(
                boot_auc_df[task], task, "AUROC"))
        summary_auc_df = pd.DataFrame.from_records(
            auc_records).sort_values(by='name')

        mcc_records = []
        for task in boot_mcc_df.columns:
            mcc_records.append(create_ci_record(
                boot_mcc_df[task], task, "MCC"))
        summary_mcc_df = pd.DataFrame.from_records(
            mcc_records).sort_values(by='name')

        f1_records = []
        for task in boot_f1_df.columns:
            f1_records.append(create_ci_record(boot_f1_df[task], task, "F1"))
        summary_f1_df = pd.DataFrame.from_records(
            f1_records).sort_values(by='name')

        acc_records = []
        for task in boot_acc_df.columns:
            acc_records.append(create_ci_record(
                boot_acc_df[task], task, "ACC"))
        summary_acc_df = pd.DataFrame.from_records(
            acc_records).sort_values(by='name')

        summary_df = reduce(lambda left, right: pd.merge(
            left, right, on=['name'], how='inner'), [summary_auc_df, summary_mcc_df, summary_f1_df, summary_acc_df])
        mean_metrics = summary_df.iloc[:, 1:].mean(axis=0)
        summary_df.loc[len(summary_df)] = ["mean"] + mean_metrics.tolist()
        summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)


def compute_multiclass_metrics(dataset_name, save_dir, gt, pred, class_names):
    pred_labels = np.argmax(pred, axis=1)
    num_samples = len(gt)
    all_accs = []
    for idx in tqdm(range(1000), desc="bootstrap"):
        sample_ids = np.random.choice(
            num_samples, size=num_samples, replace=True)
        gt_sample = gt[sample_ids]
        pred_sample = pred_labels[sample_ids]
        acc = accuracy_score(gt_sample, pred_sample)
        all_accs.append(acc)

    boot_acc_df = pd.DataFrame(np.array(all_accs), columns=["ACC"])
    boot_acc_df.to_csv(os.path.join(save_dir, f"boot_acc.csv"), index=False)
    acc_records = []
    for task in boot_acc_df.columns:
        acc_records.append(create_ci_record(boot_acc_df[task], task, "ACC"))
    summary_acc_df = pd.DataFrame.from_records(
        acc_records).sort_values(by='name')
    summary_df = summary_acc_df.copy()
    summary_df.to_csv(os.path.join(save_dir, "summary.csv"), index=False)


@torch.no_grad()
def zero_shot_cls_evaluation(model, dataloader: DataLoader,
                             pos_prompts: Dict, neg_prompts: Dict,
                             dataset_name: str,
                             prompt_style: str,
                             save_dir: str,
                             model_name: str, mode: str = "multiclass",
                             use_negative: bool = False):
    # Extract text embeddings
    if model_name in ["kad_resnet_224", "kad_resnet_512"]:
        kad_model, image_encoder, text_encoder = model
        tokenizer = kad_model.tokenizer
        max_length = 77
        class_names = list(pos_prompts.keys())
        # convert into lower case
        class_names = [x.lower() for x in class_names]
        for i, c_name in enumerate(class_names):
            if c_name == "mass":
                class_names[i] = "lung mass"
            elif c_name == "nodule":
                class_names[i] = "lung nodule"
            elif c_name == "pleural thicken":
                class_names[i] = "pleural thickening"
        text_token = tokenizer(
            class_names,
            add_special_tokens=True,
            max_length=max_length,
            pad_to_max_length=True,
            return_tensors='pt').to(device)
        text_features = text_encoder.encode_text(text_token)  # 193, 768

        if dataset_name in ["nih", "chexpert"]:
            testloader, valloader = dataloader
            dataloader = testloader
            all_val_pred = []
            all_val_labels = []
            for idx, batch in tqdm(enumerate(valloader), total=len(valloader), desc="Compute validation set"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                all_val_labels.append(labels.detach().cpu().numpy())

                image_features, image_features_pool = image_encoder(
                    pixel_values)
                pred_class = kad_model(image_features, text_features)

                pred_class = torch.softmax(pred_class, dim=-1)  # bz, 193, 2
                pred = pred_class[:, :, 1].detach().cpu().numpy()
                all_val_pred.append(pred)

            val_gt = np.concatenate(all_val_labels, axis=0)
            val_pred = np.concatenate(all_val_pred, axis=0)
        else:
            val_gt = None
            val_pred = None

        all_pred = []
        all_labels = []
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Compute test set"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            all_labels.append(labels.detach().cpu().numpy())

            image_features, image_features_pool = image_encoder(
                pixel_values)
            pred_class = kad_model(image_features, text_features)
            pred_class = torch.softmax(pred_class, dim=-1)  # bz, 193, 2
            pred = pred_class[:, :, 1].detach().cpu().numpy()
            all_pred.append(pred)

        gt = np.concatenate(all_labels, axis=0)
        pred = np.concatenate(all_pred, axis=0)

    elif model_name == "medklip":
        class_names = list(pos_prompts.keys())
        from cxrseg.third_party.medklip.load_pretrained_medklip import original_class
        indices = []
        for c_name in class_names:
            c_name = c_name.lower()
            if c_name == "lung opacity":
                c_name = "opacity"
            elif c_name == "pleural effusion":
                c_name = "effusion"
            elif c_name == "infiltration":
                c_name = "infiltrate"
            elif c_name == "pleural thickening":
                c_name = "thicken"
            elif c_name == "fibrosis":
                c_name = "tail_abnorm_obs"

            indices.append(original_class.index(c_name.lower()))

        if dataset_name in ["nih", "chexpert"]:
            testloader, valloader = dataloader
            dataloader = testloader
            all_val_pred = []
            all_val_labels = []
            for idx, batch in tqdm(enumerate(valloader), total=len(valloader), desc="Compute validation set"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                all_val_labels.append(labels.detach().cpu().numpy())

                pred_class, _ = model(pixel_values, None, is_train=False)
                pred_class = F.softmax(pred_class[:, indices], dim=-1)
                pred = pred_class[..., 1].detach().cpu().numpy()
                all_val_pred.append(pred)

            val_gt = np.concatenate(all_val_labels, axis=0)
            val_pred = np.concatenate(all_val_pred, axis=0)
        else:
            val_gt = None
            val_pred = None

        all_pred = []
        all_labels = []
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Compute test set"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            all_labels.append(labels.detach().cpu().numpy())

            pred_class, _ = model(pixel_values, None, is_train=False)
            pred_class = F.softmax(pred_class[:, indices], dim=-1)
            pred = pred_class[..., 1].detach().cpu().numpy()
            all_pred.append(pred)

        gt = np.concatenate(all_labels, axis=0)
        pred = np.concatenate(all_pred, axis=0)

    elif model_name in ["biovil", "biovil_t"]:
        # FIXME: add negative prompts
        class_names = list(pos_prompts.keys())
        tokenized_pos_texts = tokenize_prompts(model.tokenizer, pos_prompts)
        all_pos_input_ids = []
        all_pos_attention_mask = []
        for pathology in class_names:
            all_pos_input_ids.append(
                tokenized_pos_texts[pathology]["input_ids"])
            all_pos_attention_mask.append(
                tokenized_pos_texts[pathology]["attention_mask"])

        all_pos_input_ids = torch.stack(all_pos_input_ids, dim=0)
        all_pos_attention_mask = torch.stack(all_pos_attention_mask, dim=0)

        if use_negative:
            tokenized_neg_texts = tokenize_prompts(
                model.tokenizer, neg_prompts)
            all_neg_input_ids = []
            all_neg_attention_mask = []
            for pathology in class_names:
                all_neg_input_ids.append(
                    tokenized_neg_texts[pathology]["input_ids"])
                all_neg_attention_mask.append(
                    tokenized_neg_texts[pathology]["attention_mask"])

            all_neg_input_ids = torch.stack(all_neg_input_ids, dim=0)
            all_neg_attention_mask = torch.stack(all_neg_attention_mask, dim=0)

        if dataset_name in ["nih", "chexpert"]:
            testloader, valloader = dataloader
            dataloader = testloader
            all_val_pred = []
            all_val_labels = []

            for idx, batch in tqdm(enumerate(valloader), total=len(valloader), desc="Compute validation set"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                all_val_labels.append(labels.detach().cpu().numpy())

                all_pos_scores = []
                for i in range(len(class_names)):
                    cur_pos_scores = model.get_similarity_score_from_raw_data(
                        pixel_values, all_pos_input_ids[i], all_pos_attention_mask[i])
                    if use_negative:
                        cur_neg_scores = model.get_similarity_score_from_raw_data(
                            pixel_values, all_neg_input_ids[i], all_neg_attention_mask[i])

                        pos_neg_prob = F.softmax(
                            torch.stack([cur_pos_scores, cur_neg_scores], dim=-1) / 0.2, dim=-1)
                        similarities = torch.log(
                            pos_neg_prob[..., 0]).mean(dim=1).exp()
                        all_pos_scores.append(similarities.detach())
                    else:
                        all_pos_scores.append(cur_pos_scores.mean(dim=1))
                all_pos_scores = torch.stack(all_pos_scores, dim=1)
                all_val_pred.append(all_pos_scores.detach().cpu().numpy())

            val_gt = np.concatenate(all_val_labels, axis=0)
            val_pred = np.concatenate(all_val_pred, axis=0)
        else:
            val_gt = None
            val_pred = None

        all_pred = []
        all_labels = []
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Compute test set"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            all_labels.append(labels.detach().cpu().numpy())

            all_pos_scores = []
            for i in range(len(class_names)):
                cur_pos_scores = model.get_similarity_score_from_raw_data(pixel_values,
                                                                          all_pos_input_ids[i],
                                                                          all_pos_attention_mask[i])
                if use_negative:
                    cur_neg_scores = model.get_similarity_score_from_raw_data(
                        pixel_values, all_neg_input_ids[i], all_neg_attention_mask[i])

                    pos_neg_prob = F.softmax(
                        torch.stack([cur_pos_scores, cur_neg_scores], dim=-1) / 0.2, dim=-1)
                    similarities = torch.log(
                        pos_neg_prob[..., 0]).mean(dim=1).exp()
                    all_pos_scores.append(similarities.detach())
                else:
                    all_pos_scores.append(cur_pos_scores.mean(dim=1))

            all_pos_scores = torch.stack(all_pos_scores, dim=1)
            all_pred.append(all_pos_scores.detach().cpu().numpy())

        gt = np.concatenate(all_labels, axis=0)
        pred = np.concatenate(all_pred, axis=0)

    elif model_name == "chexzero":
        class_names = list(pos_prompts.keys())
        tokenized_pos_texts = tokenize_chexzero_prompts(pos_prompts)
        tokenized_neg_texts = tokenize_chexzero_prompts(neg_prompts)
        pos_report_embs = []
        neg_report_embs = []
        for pathology in class_names:
            texts = tokenized_pos_texts[pathology]
            class_embeddings = model.encode_text(texts)
            class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
            pos_report_embs.append(class_embeddings)

            if use_negative:
                texts = tokenized_neg_texts[pathology]
                class_embeddings = model.encode_text(texts)
                class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
                neg_report_embs.append(class_embeddings)

        pos_report_embs = torch.stack(pos_report_embs, dim=0)
        if use_negative:
            neg_report_embs = torch.stack(neg_report_embs, dim=0)

        if dataset_name in ["nih", "chexpert"]:
            testloader, valloader = dataloader
            dataloader = testloader
            all_val_pred = []
            all_val_labels = []

            for idx, batch in tqdm(enumerate(valloader), total=len(valloader), desc="Compute validation set"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                all_val_labels.append(labels.detach().cpu().numpy())

                image_features = model.encode_image(pixel_values)[0]
                image_features = F.normalize(image_features, p=2, dim=-1)
                all_pos_scores = torch.einsum(
                    "bd,cpd->bcp", image_features, pos_report_embs)
                if use_negative:
                    all_neg_scores = torch.einsum(
                        "bd,cpd->bcp", image_features, neg_report_embs)
                    pos_neg_prob = F.softmax(
                        torch.stack([all_pos_scores, all_neg_scores], dim=-1) / 0.2, dim=-1)
                    similarities = torch.log(
                        pos_neg_prob[..., 0]).mean(dim=2).exp()
                else:
                    similarities = all_pos_scores.mean(dim=2)
                all_val_pred.append(similarities.detach().cpu().numpy())

            val_gt = np.concatenate(all_val_labels, axis=0)
            val_pred = np.concatenate(all_val_pred, axis=0)
        else:
            val_gt = None
            val_pred = None

        all_pred = []
        all_labels = []
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Compute test set"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            all_labels.append(labels.detach().cpu().numpy())

            image_features = model.encode_image(pixel_values)[0]
            image_features = F.normalize(image_features, p=2, dim=-1)

            all_pos_scores = torch.einsum(
                "bd,cpd->bcp", image_features, pos_report_embs)
            if use_negative:
                all_neg_scores = torch.einsum(
                    "bd,cpd->bcp", image_features, neg_report_embs)
                pos_neg_prob = F.softmax(
                    torch.stack([all_pos_scores, all_neg_scores], dim=-1) / 0.2, dim=-1)
                similarities = torch.log(
                    pos_neg_prob[..., 0]).mean(dim=2).exp()
            else:
                similarities = all_pos_scores.mean(dim=2)

            all_pred.append(similarities.detach().cpu().numpy())

        gt = np.concatenate(all_labels, axis=0)
        pred = np.concatenate(all_pred, axis=0)

    elif model_name in ["random", "gloria", "gloria_chexpert"]:
        class_names = list(pos_prompts.keys())
        tokenized_pos_texts = tokenize_prompts(model.tokenizer, pos_prompts)
        tokenized_neg_texts = tokenize_prompts(model.tokenizer, neg_prompts)
        pos_word_embs, pos_report_embs, pos_sents = [], [], []
        neg_word_embs, neg_report_embs, neg_sents = [], [], []
        for pathology in class_names:
            texts = tokenized_pos_texts[pathology]
            word_embs, report_embs, sents = model.text_encoder(
                texts["input_ids"], texts["attention_mask"]
            )
            report_embs = F.normalize(report_embs, p=2, dim=-1)
            word_embs = F.normalize(word_embs, p=2, dim=-1)
            pos_report_embs.append(report_embs)
            pos_word_embs.append(word_embs)
            pos_sents.append(sents)

            if use_negative:
                texts = tokenized_neg_texts[pathology]
                word_embs, report_embs, sents = model.text_encoder(
                    texts["input_ids"], texts["attention_mask"]
                )
                report_embs = F.normalize(report_embs, p=2, dim=-1)
                word_embs = F.normalize(word_embs, p=2, dim=-1)
                neg_report_embs.append(report_embs)
                neg_word_embs.append(word_embs)
                neg_sents.append(sents)

        pos_report_embs = torch.stack(pos_report_embs, dim=0)
        pos_word_embs = torch.stack(pos_word_embs, dim=0)
        if use_negative:
            neg_report_embs = torch.stack(neg_report_embs, dim=0)
            neg_word_embs = torch.stack(neg_word_embs, dim=0)

        if dataset_name in ["nih", "chexpert"]:
            testloader, valloader = dataloader
            dataloader = testloader
            all_val_pred = []
            all_val_labels = []

            for idx, batch in tqdm(enumerate(valloader), total=len(valloader), desc="Compute validation set"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                all_val_labels.append(labels.detach().cpu().numpy())

                img_emb_l, img_emb_g = model.image_encoder_forward(
                    pixel_values)
                img_emb_l = F.normalize(img_emb_l, p=2, dim=-1)
                img_emb_g = F.normalize(img_emb_g, p=2, dim=-1)

                # cls_val_pos_pred, cls_val_neg_pred = [], []
                cls_val_pred = []
                for i in range(len(class_names)):
                    cap_lens = [
                        len([w for w in sent if not w.startswith("[")]) + 1 for sent in pos_sents[i]]
                    global_similarities = model.get_global_similarities(
                        img_emb_g, pos_report_embs[i])
                    local_similarities = model.get_local_similarities(
                        img_emb_l, pos_word_embs[i], cap_lens)
                    pos_similarities = (
                        global_similarities + local_similarities) / 2
                    if use_negative:
                        cap_lens = [
                            len([w for w in sent if not w.startswith("[")]) + 1 for sent in neg_sents[i]]
                        global_similarities = model.get_global_similarities(
                            img_emb_g, neg_report_embs[i])
                        local_similarities = model.get_local_similarities(
                            img_emb_l, neg_word_embs[i], cap_lens)
                        neg_similarities = (
                            global_similarities + local_similarities) / 2
                        pos_neg_prob = F.softmax(
                            torch.stack([pos_similarities, neg_similarities], dim=-1) / 0.2, dim=-1)
                        similarities = torch.log(
                            pos_neg_prob[..., 0]).mean(dim=1).exp()
                        cls_val_pred.append(
                            similarities.detach().cpu().numpy())
                    else:
                        similarities = pos_similarities.mean(dim=1)
                        cls_val_pred.append(
                            similarities.detach().cpu().numpy())

                cls_val_pred = np.stack(cls_val_pred, axis=1)
                cls_val_pred = (
                    cls_val_pred - cls_val_pred.mean(axis=0)) / (cls_val_pred.std(axis=0))
                all_val_pred.append(cls_val_pred)

            val_gt = np.concatenate(all_val_labels, axis=0)
            val_pred = np.concatenate(all_val_pred, axis=0)
        else:
            val_gt = None
            val_pred = None

        all_pred = []
        all_labels = []
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Compute test set"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            all_labels.append(labels.detach().cpu().numpy())

            img_emb_l, img_emb_g = model.image_encoder_forward(pixel_values)
            img_emb_l = F.normalize(img_emb_l, p=2, dim=-1)
            img_emb_g = F.normalize(img_emb_g, p=2, dim=-1)

            cls_pred = []
            for i in range(len(class_names)):
                cap_lens = [
                    len([w for w in sent if not w.startswith("[")]) + 1 for sent in pos_sents[i]]
                global_similarities = model.get_global_similarities(
                    img_emb_g, pos_report_embs[i])
                local_similarities = model.get_local_similarities(
                    img_emb_l, pos_word_embs[i], cap_lens)
                pos_similarities = (global_similarities +
                                    local_similarities) / 2
                if use_negative:
                    cap_lens = [
                        len([w for w in sent if not w.startswith("[")]) + 1 for sent in neg_sents[i]]
                    global_similarities = model.get_global_similarities(
                        img_emb_g, neg_report_embs[i])
                    local_similarities = model.get_local_similarities(
                        img_emb_l, neg_word_embs[i], cap_lens)
                    neg_similarities = (
                        global_similarities + local_similarities) / 2
                    pos_neg_prob = F.softmax(
                        torch.stack([pos_similarities, neg_similarities], dim=-1) / 0.2, dim=-1)
                    similarities = torch.log(
                        pos_neg_prob[..., 0]).mean(dim=1).exp()
                    cls_pred.append(similarities.detach().cpu().numpy())
                else:
                    similarities = pos_similarities.mean(dim=1)
                    cls_pred.append(similarities.detach().cpu().numpy())

            cls_pred = np.stack(cls_pred, axis=1)
            cls_pred = (cls_pred - cls_pred.mean(axis=0)) / \
                (cls_pred.std(axis=0))
            all_pred.append(cls_pred)

        gt = np.concatenate(all_labels, axis=0)
        pred = np.concatenate(all_pred, axis=0)

    elif model_name == "afloc":
        class_names = list(pos_prompts.keys())
        # tokenized_pos_texts = tokenize_prompts(model.tokenizer, pos_prompts)
        # tokenized_neg_texts = tokenize_prompts(model.tokenizer, neg_prompts)
        pos_report_embs = []
        neg_report_embs = []
        for pathology in class_names:
            # texts = tokenized_pos_texts[pathology]
            text_prompts = pos_prompts[pathology]
            texts = model.process_text(text_prompts, device)
            res = model.text_encoder_forward(
                texts["caption_ids"].to(device),
                texts["attention_mask"].to(device),
                texts["token_type_ids"].to(device))
            report_embs = res["report_embeddings"]
            report_embs = F.normalize(report_embs, p=2, dim=-1)
            pos_report_embs.append(report_embs)

            if use_negative:
                text_prompts = neg_prompts[pathology]
                texts = model.process_text(text_prompts, device)
                res = model.text_encoder_forward(
                    texts["caption_ids"].to(device),
                    texts["attention_mask"].to(device),
                    texts["token_type_ids"].to(device))
                report_embs = res["report_embeddings"]
                report_embs = F.normalize(report_embs, p=2, dim=-1)
                neg_report_embs.append(report_embs)

        pos_report_embs = torch.stack(pos_report_embs, dim=0)

        if use_negative:
            neg_report_embs = torch.stack(neg_report_embs, dim=0)

        if dataset_name in ["nih", "chexpert"]:
            testloader, valloader = dataloader
            dataloader = testloader
            all_val_pred = []
            all_val_labels = []

            for idx, batch in tqdm(enumerate(valloader), total=len(valloader), desc="Compute validation set"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                all_val_labels.append(labels.detach().cpu().numpy())

                _, _, _, img_emb_g = model.image_encoder_forward(
                    pixel_values)
                img_emb_g = F.normalize(img_emb_g, p=2, dim=-1)

                cls_val_pred = []
                for i in range(len(class_names)):
                    pos_similarities = img_emb_g @ pos_report_embs[i].T
                    if use_negative:
                        neg_similarities = img_emb_g @ neg_report_embs[i].T
                        pos_neg_prob = F.softmax(
                            torch.stack([pos_similarities, neg_similarities], dim=-1) / 0.2, dim=-1)
                        similarities = torch.log(
                            pos_neg_prob[..., 0]).mean(dim=1).exp()
                        cls_val_pred.append(
                            similarities.detach().cpu().numpy())
                    else:
                        similarities = pos_similarities.mean(dim=1)
                        cls_val_pred.append(
                            similarities.detach().cpu().numpy())

                cls_val_pred = np.stack(cls_val_pred, axis=1)
                cls_val_pred = (
                    cls_val_pred - cls_val_pred.mean(axis=0)) / (cls_val_pred.std(axis=0))
                all_val_pred.append(cls_val_pred)

            val_gt = np.concatenate(all_val_labels, axis=0)
            val_pred = np.concatenate(all_val_pred, axis=0)
        else:
            val_gt = None
            val_pred = None

        all_pred = []
        all_labels = []
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Compute test set"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            all_labels.append(labels.detach().cpu().numpy())

            _, _, _, img_emb_g = model.image_encoder_forward(
                pixel_values)
            img_emb_g = F.normalize(img_emb_g, p=2, dim=-1)

            cls_pred = []
            for i in range(len(class_names)):
                pos_similarities = img_emb_g @ pos_report_embs[i].T
                if use_negative:
                    neg_similarities = img_emb_g @ neg_report_embs[i].T
                    pos_neg_prob = F.softmax(
                        torch.stack([pos_similarities, neg_similarities], dim=-1) / 0.2, dim=-1)
                    similarities = torch.log(
                        pos_neg_prob[..., 0]).mean(dim=1).exp()
                    cls_pred.append(similarities.detach().cpu().numpy())
                else:
                    similarities = pos_similarities.mean(dim=1)
                    cls_pred.append(similarities.detach().cpu().numpy())

            cls_pred = np.stack(cls_pred, axis=1)
            cls_pred = (cls_pred - cls_pred.mean(axis=0)) / \
                (cls_pred.std(axis=0))
            all_pred.append(cls_pred)

        gt = np.concatenate(all_labels, axis=0)
        pred = np.concatenate(all_pred, axis=0)

    elif model_name == "convirt":
        class_names = list(pos_prompts.keys())
        tokenized_pos_texts = tokenize_prompts(model.tokenizer, pos_prompts)
        tokenized_neg_texts = tokenize_prompts(model.tokenizer, neg_prompts)
        pos_report_embs = []
        neg_report_embs = []
        for pathology in class_names:
            texts = tokenized_pos_texts[pathology]
            word_embs, report_embs, sents = model.text_encoder(
                texts["input_ids"], texts["attention_mask"]
            )
            report_embs = F.normalize(report_embs, p=2, dim=-1)
            word_embs = F.normalize(word_embs, p=2, dim=-1)
            pos_report_embs.append(report_embs)

            if use_negative:
                texts = tokenized_neg_texts[pathology]
                word_embs, report_embs, sents = model.text_encoder(
                    texts["input_ids"], texts["attention_mask"]
                )
                report_embs = F.normalize(report_embs, p=2, dim=-1)
                word_embs = F.normalize(word_embs, p=2, dim=-1)
                neg_report_embs.append(report_embs)

        pos_report_embs = torch.stack(pos_report_embs, dim=0)
        if use_negative:
            neg_report_embs = torch.stack(neg_report_embs, dim=0)

        if dataset_name in ["nih", "chexpert"]:
            testloader, valloader = dataloader
            dataloader = testloader
            all_val_pred = []
            all_val_labels = []

            for idx, batch in tqdm(enumerate(valloader), total=len(valloader), desc="Compute validation set"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                all_val_labels.append(labels.detach().cpu().numpy())

                _, img_emb_g = model.image_encoder_forward(
                    pixel_values)
                img_emb_g = F.normalize(img_emb_g, p=2, dim=-1)

                cls_val_pred = []
                for i in range(len(class_names)):
                    pos_similarities = model.get_global_similarities(
                        img_emb_g, pos_report_embs[i])
                    if use_negative:
                        neg_similarities = model.get_global_similarities(
                            img_emb_g, neg_report_embs[i])
                        pos_neg_prob = F.softmax(
                            torch.stack([pos_similarities, neg_similarities], dim=-1) / 0.2, dim=-1)
                        similarities = torch.log(
                            pos_neg_prob[..., 0]).mean(dim=1).exp()
                        cls_val_pred.append(
                            similarities.detach().cpu().numpy())
                    else:
                        similarities = pos_similarities.mean(dim=1)
                        cls_val_pred.append(
                            similarities.detach().cpu().numpy())

                cls_val_pred = np.stack(cls_val_pred, axis=1)
                cls_val_pred = (
                    cls_val_pred - cls_val_pred.mean(axis=0)) / (cls_val_pred.std(axis=0))
                all_val_pred.append(cls_val_pred)

            val_gt = np.concatenate(all_val_labels, axis=0)
            val_pred = np.concatenate(all_val_pred, axis=0)
        else:
            val_gt = None
            val_pred = None

        all_pred = []
        all_labels = []
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Compute test set"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            all_labels.append(labels.detach().cpu().numpy())

            _, img_emb_g = model.image_encoder_forward(pixel_values)
            img_emb_g = F.normalize(img_emb_g, p=2, dim=-1)

            cls_pred = []
            for i in range(len(class_names)):
                pos_similarities = model.get_global_similarities(
                    img_emb_g, pos_report_embs[i])

                if use_negative:
                    neg_similarities = model.get_global_similarities(
                        img_emb_g, neg_report_embs[i])
                    pos_neg_prob = F.softmax(
                        torch.stack([pos_similarities, neg_similarities], dim=-1) / 0.2, dim=-1)
                    similarities = torch.log(
                        pos_neg_prob[..., 0]).mean(dim=1).exp()
                    cls_pred.append(similarities.detach().cpu().numpy())
                else:
                    similarities = pos_similarities.mean(dim=1)
                    cls_pred.append(similarities.detach().cpu().numpy())

            cls_pred = np.stack(cls_pred, axis=1)
            cls_pred = (cls_pred - cls_pred.mean(axis=0)) / \
                (cls_pred.std(axis=0))
            all_pred.append(cls_pred)

        gt = np.concatenate(all_labels, axis=0)
        pred = np.concatenate(all_pred, axis=0)

    elif model_name in ["mgca_cnn", "mgca_vit"]:
        class_names = list(pos_prompts.keys())
        tokenized_pos_texts = tokenize_prompts(model.tokenizer, pos_prompts)
        tokenized_neg_texts = tokenize_prompts(model.tokenizer, neg_prompts)
        pos_report_embs = []
        neg_report_embs = []
        for pathology in class_names:
            texts = tokenized_pos_texts[pathology]
            _, report_embs = model.encode_text(
                texts["input_ids"], texts["attention_mask"]
            )
            report_embs = F.normalize(report_embs, p=2, dim=-1)
            pos_report_embs.append(report_embs)

            if use_negative:
                texts = tokenized_neg_texts[pathology]
                _, report_embs = model.encode_text(
                    texts["input_ids"], texts["attention_mask"]
                )
                report_embs = F.normalize(report_embs, p=2, dim=-1)
                neg_report_embs.append(report_embs)

        pos_report_embs = torch.stack(pos_report_embs, dim=0)
        if use_negative:
            neg_report_embs = torch.stack(neg_report_embs, dim=0)

        if dataset_name in ["nih", "chexpert"]:
            testloader, valloader = dataloader
            dataloader = testloader
            all_val_pred = []
            all_val_labels = []

            for idx, batch in tqdm(enumerate(valloader), total=len(valloader), desc="Compute validation set"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                all_val_labels.append(labels.detach().cpu().numpy())

                img_emb_g, _ = model.encode_image(
                    pixel_values)
                img_emb_g = F.normalize(img_emb_g, p=2, dim=-1)

                cls_val_pred = []
                for i in range(len(class_names)):
                    pos_similarities = img_emb_g @ pos_report_embs[i].T
                    if use_negative:
                        neg_similarities = img_emb_g @ neg_report_embs[i].T
                        pos_neg_prob = F.softmax(
                            torch.stack([pos_similarities, neg_similarities], dim=-1) / 0.2, dim=-1)
                        similarities = torch.log(
                            pos_neg_prob[..., 0]).mean(dim=1).exp()
                        cls_val_pred.append(
                            similarities.detach().cpu().numpy())
                    else:
                        similarities = pos_similarities.mean(dim=1)
                        cls_val_pred.append(
                            similarities.detach().cpu().numpy())

                cls_val_pred = np.stack(cls_val_pred, axis=1)
                # cls_val_pred = (
                #     cls_val_pred - cls_val_pred.mean(axis=0)) / (cls_val_pred.std(axis=0))
                all_val_pred.append(cls_val_pred)

            val_gt = np.concatenate(all_val_labels, axis=0)
            val_pred = np.concatenate(all_val_pred, axis=0)
        else:
            val_gt = None
            val_pred = None

        all_pred = []
        all_labels = []
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Compute test set"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            all_labels.append(labels.detach().cpu().numpy())

            img_emb_g, _ = model.encode_image(pixel_values)
            img_emb_g = F.normalize(img_emb_g, p=2, dim=-1)

            cls_pred = []
            for i in range(len(class_names)):
                pos_similarities = img_emb_g @ pos_report_embs[i].T
                if use_negative:
                    neg_similarities = img_emb_g @ neg_report_embs[i].T
                    pos_neg_prob = F.softmax(
                        torch.stack([pos_similarities, neg_similarities], dim=-1) / 0.2, dim=-1)
                    similarities = torch.log(
                        pos_neg_prob[..., 0]).mean(dim=1).exp()
                    cls_pred.append(similarities.detach().cpu().numpy())
                else:
                    similarities = pos_similarities.mean(dim=1)
                    cls_pred.append(similarities.detach().cpu().numpy())

            cls_pred = np.stack(cls_pred, axis=1)
            # cls_pred = (cls_pred - cls_pred.mean(axis=0)) / \
            #     (cls_pred.std(axis=0))
            all_pred.append(cls_pred)

        gt = np.concatenate(all_labels, axis=0)
        pred = np.concatenate(all_pred, axis=0)

    elif model_name in ["medclip_cnn", "medclip_vit"]:
        class_names = list(pos_prompts.keys())
        # tokenized_pos_texts = tokenize_chexzero_prompts(pos_prompts)
        tokenized_pos_texts = tokenize_prompts(model.tokenizer, pos_prompts)
        zeroshot_weights = []
        for pathology in class_names:
            texts = tokenized_pos_texts[pathology]
            class_embeddings = model.encode_text(
                texts["input_ids"], texts["attention_mask"]
            )
            class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
            zeroshot_weights.append(class_embeddings)
        zeroshot_weights = torch.stack(zeroshot_weights, dim=0)

        if dataset_name in ["nih", "chexpert"]:
            testloader, valloader = dataloader
            dataloader = testloader
            all_val_pred = []
            all_val_labels = []

            for idx, batch in tqdm(enumerate(valloader), total=len(valloader), desc="Compute validation set"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                all_val_labels.append(labels.detach().cpu().numpy())

                image_features = model.encode_image(pixel_values)
                image_features = F.normalize(image_features, p=2, dim=-1)
                all_pos_scores = torch.einsum(
                    "bd,cpd->bcp", image_features, zeroshot_weights)
                all_pos_scores = all_pos_scores.mean(dim=2)
                all_val_pred.append(all_pos_scores.detach().cpu().numpy())

            val_gt = np.concatenate(all_val_labels, axis=0)
            val_pred = np.concatenate(all_val_pred, axis=0)
        else:
            val_gt = None
            val_pred = None

        all_pred = []
        all_labels = []
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Compute test set"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            all_labels.append(labels.detach().cpu().numpy())

            image_features = model.encode_image(pixel_values)
            image_features = F.normalize(image_features, p=2, dim=-1)
            all_pos_scores = torch.einsum(
                "bd,cpd->bcp", image_features, zeroshot_weights)
            all_pos_scores = all_pos_scores.mean(dim=2)

            all_pred.append(all_pos_scores.detach().cpu().numpy())

        gt = np.concatenate(all_labels, axis=0)
        pred = np.concatenate(all_pred, axis=0)

    elif model_name in ["our_medclip", "our_medclip_s2"]:
        class_names = list(pos_prompts.keys())
        tokenized_pos_texts = tokenize_prompts(model.tokenizer, pos_prompts)
        tokenized_neg_texts = tokenize_prompts(model.tokenizer, neg_prompts)
        pos_report_embs = []
        neg_report_embs = []
        for pathology in class_names:
            texts = tokenized_pos_texts[pathology]
            _, class_embeddings, _ = model.encode_text_student(
                texts["input_ids"], texts["attention_mask"]
            )
            class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
            pos_report_embs.append(class_embeddings)

            if use_negative:
                texts = tokenized_neg_texts[pathology]
                _, class_embeddings, _ = model.encode_text_student(
                    texts["input_ids"], texts["attention_mask"]
                )
                class_embeddings = F.normalize(class_embeddings, p=2, dim=-1)
                neg_report_embs.append(class_embeddings)

        pos_report_embs = torch.stack(pos_report_embs, dim=0)

        if use_negative:
            neg_report_embs = torch.stack(neg_report_embs, dim=0)

        if dataset_name in ["nih", "chexpert"]:
            testloader, valloader = dataloader
            dataloader = testloader
            all_val_pred = []
            all_val_labels = []

            for idx, batch in tqdm(enumerate(valloader), total=len(valloader), desc="Compute validation set"):
                pixel_values = batch['pixel_values'].to(device)
                labels = batch['labels'].to(device)
                all_val_labels.append(labels.detach().cpu().numpy())

                _, patch_feats = model.encode_image_student(
                    pixel_values, dense=True)
                patch_embeds = model.img_projection_student(patch_feats)
                image_features = torch.mean(patch_embeds, dim=1)
                image_features = F.normalize(image_features, p=2, dim=-1)

                cls_val_pred = []
                for i in range(len(class_names)):
                    pos_similarities = image_features @ pos_report_embs[i].T
                    if use_negative:
                        neg_similarities = image_features @ neg_report_embs[i].T
                        pos_neg_prob = F.softmax(
                            torch.stack([pos_similarities, neg_similarities], dim=-1) / 0.2, dim=-1)
                        similarities = torch.log(
                            pos_neg_prob[..., 0]).mean(dim=1).exp()
                        cls_val_pred.append(
                            similarities.detach().cpu().numpy())
                    else:
                        similarities = pos_similarities.mean(dim=1)
                        cls_val_pred.append(
                            similarities.detach().cpu().numpy())

                # all_pos_scores = torch.einsum(
                #     "bd,cpd->bcp", image_features, zeroshot_weights)
                # all_pos_scores = all_pos_scores.mean(dim=2)
                # all_val_pred.append(all_pos_scores.detach().cpu().numpy())

                cls_val_pred = np.stack(cls_val_pred, axis=1)
                # cls_val_pred = (
                #     cls_val_pred - cls_val_pred.mean(axis=0)) / (cls_val_pred.std(axis=0))
                all_val_pred.append(cls_val_pred)

            val_gt = np.concatenate(all_val_labels, axis=0)
            val_pred = np.concatenate(all_val_pred, axis=0)
        else:
            val_gt = None
            val_pred = None

        all_pred = []
        all_labels = []
        for idx, batch in tqdm(enumerate(dataloader), total=len(dataloader), desc="Compute test set"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            all_labels.append(labels.detach().cpu().numpy())

            _, patch_feats = model.encode_image_student(
                pixel_values, dense=True)
            patch_embeds = model.img_projection_student(patch_feats)
            image_features = torch.mean(patch_embeds, dim=1)
            image_features = F.normalize(image_features, p=2, dim=-1)

            cls_pred = []
            for i in range(len(class_names)):
                pos_similarities = image_features @ pos_report_embs[i].T
                if use_negative:
                    neg_similarities = image_features @ neg_report_embs[i].T
                    pos_neg_prob = F.softmax(
                        torch.stack([pos_similarities, neg_similarities], dim=-1) / 0.2, dim=-1)
                    similarities = torch.log(
                        pos_neg_prob[..., 0]).mean(dim=1).exp()
                    cls_pred.append(
                        similarities.detach().cpu().numpy())
                else:
                    similarities = pos_similarities.mean(dim=1)
                    cls_pred.append(
                        similarities.detach().cpu().numpy())

            cls_pred = np.stack(cls_pred, axis=1)
            # cls_pred = (cls_pred - cls_pred.mean(axis=0)) / \
            #     (cls_pred.std(axis=0))
            all_pred.append(cls_pred)

        gt = np.concatenate(all_labels, axis=0)
        pred = np.concatenate(all_pred, axis=0)

    else:
        raise NotImplementedError

    # create the folder to save results
    if use_negative:
        save_dir = os.path.join(
            save_dir, f"{dataset_name}_{model_name}_{prompt_style}_use_negative")
    else:
        save_dir = os.path.join(
            save_dir, f"{dataset_name}_{model_name}_{prompt_style}")
    os.makedirs(save_dir, exist_ok=True)

    if mode == "multilabel":
        compute_multilabel_metrics(dataset_name, save_dir,
                                   gt, pred, class_names,
                                   val_gt=val_gt, val_pred=val_pred)
    elif mode == "multiclass":
        compute_multiclass_metrics(dataset_name, save_dir,
                                   gt, pred, class_names)


@torch.no_grad()
def tokenize_prompts(tokenizer, pos_prompts, num_templates=4):
    '''
    Tokenize class prompts
    '''
    class_names = list(pos_prompts.keys())
    tokenized_pos_text = dict()
    for pathology in class_names:
        pos_texts = pos_prompts[pathology]
        if len(pos_texts) > num_templates:
            pos_texts = random.sample(pos_texts, num_templates)
        elif len(pos_texts) < num_templates:
            pad_num = num_templates - len(pos_texts)
            pos_texts += [pos_texts[-1]] * pad_num
        assert len(pos_texts) == num_templates
        pos_texts = tokenizer(pos_texts,
                              return_tensors="pt",
                              padding=True,
                              max_length=77)

        pos_texts["input_ids"] = pos_texts["input_ids"].cuda()
        pos_texts["attention_mask"] = pos_texts["attention_mask"].cuda()
        tokenized_pos_text[pathology] = pos_texts
    return tokenized_pos_text


@torch.no_grad()
def tokenize_chexzero_prompts(pos_prompts, num_templates=4):
    '''
    Tokenize class prompts
    '''
    class_names = list(pos_prompts.keys())
    tokenized_pos_text = dict()
    for pathology in class_names:
        pos_texts = pos_prompts[pathology]
        if len(pos_texts) > num_templates:
            pos_texts = random.sample(pos_texts, num_templates)
        elif len(pos_texts) < num_templates:
            pad_num = num_templates - len(pos_texts)
            pos_texts += [pos_texts[-1]] * pad_num
        assert len(pos_texts) == num_templates
        from cxrseg.third_party.CheXzero.clip import tokenize
        pos_texts = tokenize(pos_texts)

        tokenized_pos_text[pathology] = pos_texts.cuda()

    return tokenized_pos_text


def get_model(hparams: Namespace):
    # TODO: fix herer with abs path
    if hparams.model_name == "biovil_t":
        # Load biovil
        from cxrseg.third_party.biovil.text import get_bert_inference
        from cxrseg.third_party.biovil.text.utils import BertEncoderType
        from cxrseg.third_party.biovil.image import get_image_inference
        from cxrseg.third_party.biovil.image.utils import ImageModelType
        from cxrseg.third_party.biovil.vlp import ImageTextInferenceEngine

        text_inference = get_bert_inference(BertEncoderType.BIOVIL_T_BERT)
        image_inference = get_image_inference(ImageModelType.BIOVIL_T)
        image_text_inference = ImageTextInferenceEngine(
            image_inference_engine=image_inference,
            text_inference_engine=text_inference,
        )
        image_text_inference.to(device)
        return image_text_inference
    elif hparams.model_name == "biovil":
        # Load biovil
        from cxrseg.third_party.biovil.text import get_bert_inference
        from cxrseg.third_party.biovil.text.utils import BertEncoderType
        from cxrseg.third_party.biovil.image import get_image_inference
        from cxrseg.third_party.biovil.image.utils import ImageModelType
        from cxrseg.third_party.biovil.vlp import ImageTextInferenceEngine

        text_inference = get_bert_inference(BertEncoderType.CXR_BERT)
        image_inference = get_image_inference(ImageModelType.BIOVIL)
        image_text_inference = ImageTextInferenceEngine(
            image_inference_engine=image_inference,
            text_inference_engine=text_inference,
        )
        image_text_inference.to(device)
        return image_text_inference
    elif hparams.model_name == "medclip_vit":
        from cxrseg.third_party.medclip import MedCLIPVisionModelViT, MedCLIPModel
        medclip = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
        medclip.from_pretrained(
            input_dir="../pretrained/medclip/medclip-vit")
        # medclip.logit_scale.data = torch.tensor(0.).to(device)
        medclip = medclip.to(device)
        medclip.eval()
        return medclip
    elif hparams.model_name == "medclip_cnn":
        from cxrseg.third_party.medclip import MedCLIPVisionModel, MedCLIPModel
        medclip = MedCLIPModel(vision_cls=MedCLIPVisionModel)
        medclip.from_pretrained(
            input_dir="../pretrained/medclip/medclip-resnet")
        # medclip.logit_scale.data = torch.tensor(0.).to(device)
        medclip = medclip.to(device)
        medclip.eval()
        return medclip
    elif hparams.model_name in ["gloria", "convirt"]:
        print(
            f"Load checkpoint of {hparams.model_name} from {hparams.ckpt_path}")
        medclip = GLoRIAModule.load_from_checkpoint(hparams.ckpt_path, map_location=device,
                                                    dataset_dir=hparams.dataset_dir, strict=False)
        medclip.eval()
        return medclip
    elif hparams.model_name == "random":
        print(f"Load random model")
        medclip = GLoRIAModule(dataset_dir=hparams.dataset_dir).to(device)
        medclip.eval()
        return medclip
    elif hparams.model_name == "our_medclip":
        print(
            f"Load checkpoint of {hparams.model_name} from {hparams.ckpt_path}")

        # Load the checkpoint
        ckpt = torch.load(hparams.ckpt_path, map_location=device)
        hyper_parameters = ckpt["hyper_parameters"]
        silc_module = SILCModule(**hyper_parameters).to(device)

        # only load three modules
        img_encoder_ckpt = dict()
        text_encoder_ckpt = dict()
        img_projection_ckpt = dict()
        text_projection_ckpt = dict()
        for k, v in ckpt["state_dict"].items():
            if "img_encoder_student" in k:
                img_encoder_ckpt[k.replace("img_encoder_student.", "")] = v
            elif "text_encoder_student" in k:
                text_encoder_ckpt[k.replace("text_encoder_student.", "")] = v
            elif "img_projection_student" in k:
                img_projection_ckpt[k.replace(
                    "img_projection_student.", "")] = v
            elif "text_projection_student" in k:
                text_projection_ckpt[k.replace(
                    "text_projection_student.", "")] = v

        silc_module.img_encoder_student.load_state_dict(img_encoder_ckpt)
        silc_module.text_encoder_student.load_state_dict(text_encoder_ckpt)
        silc_module.img_projection_student.load_state_dict(img_projection_ckpt)
        silc_module.text_projection_student.load_state_dict(
            text_projection_ckpt)
        silc_module.eval()

        return silc_module

    elif hparams.model_name == "medklip":
        from cxrseg.third_party.medklip.load_pretrained_medklip import load_pretrained_medklip
        medclip = load_pretrained_medklip(
            model_path="../pretrained/MedKLIP", device=device)
        medclip.eval()
        return medclip
    elif hparams.model_name == "kad_resnet_224":
        bert_model_name = "xmcmic/Med-KEBERT"
        from transformers import AutoTokenizer
        from cxrseg.third_party.kad.A3_CLIP.models.clip_tqn import ModelRes, ModelRes512, CLP_clinical, TQN_Model
        image_encoder = ModelRes(res_base_model='resnet50').to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            bert_model_name, do_lower_case=True)
        text_encoder = CLP_clinical(
            bert_model_name=bert_model_name).to(device=device)

        bert_pretrained = "../pretrained/KAD_Models/Knowledge_Encoder/epoch_latest.pt"
        checkpoint = torch.load(bert_pretrained, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        text_encoder.load_state_dict(state_dict, strict=False)
        print('Load pretrained bert success from: ', bert_pretrained)
        for param in text_encoder.parameters():
            param.requires_grad = False

        model = TQN_Model().to(device)
        model.tokenizer = tokenizer

        checkpoint_path = "../pretrained/KAD_Models/KAD_224/best_valid.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        image_state_dict = checkpoint['image_encoder']
        image_encoder.load_state_dict(image_state_dict)
        text_state_dict = checkpoint['text_encoder']
        text_encoder.load_state_dict(text_state_dict, strict=False)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        print('load checkpoint from %s' % checkpoint_path)

        model.eval()
        image_encoder.eval()
        text_encoder.eval()

        return (model, image_encoder, text_encoder)

    elif hparams.model_name == "kad_resnet_512":
        bert_model_name = "xmcmic/Med-KEBERT"
        from transformers import AutoTokenizer
        from cxrseg.third_party.kad.A3_CLIP.models.clip_tqn import ModelRes, ModelRes512, CLP_clinical, TQN_Model
        image_encoder = ModelRes512(res_base_model='resnet50').to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            bert_model_name, do_lower_case=True)
        text_encoder = CLP_clinical(
            bert_model_name=bert_model_name).to(device=device)

        bert_pretrained = "../pretrained/KAD_Models/Knowledge_Encoder/epoch_latest.pt"
        checkpoint = torch.load(bert_pretrained, map_location='cpu')
        state_dict = checkpoint["state_dict"]
        text_encoder.load_state_dict(state_dict, strict=False)
        print('Load pretrained bert success from: ', bert_pretrained)
        for param in text_encoder.parameters():
            param.requires_grad = False

        model = TQN_Model().to(device)
        model.tokenizer = tokenizer

        checkpoint_path = "../pretrained/KAD_Models/KAD_512/best_valid.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        image_state_dict = checkpoint['image_encoder']
        image_encoder.load_state_dict(image_state_dict)
        text_state_dict = checkpoint['text_encoder']
        text_encoder.load_state_dict(text_state_dict, strict=False)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        print('load checkpoint from %s' % checkpoint_path)

        model.eval()
        image_encoder.eval()
        text_encoder.eval()

        return (model, image_encoder, text_encoder)

    elif hparams.model_name == "kad_resnet_1024":
        bert_model_name = "xmcmic/Med-KEBERT"
        from transformers import AutoTokenizer
        from cxrseg.third_party.kad.A3_CLIP.models.clip_tqn import ModelRes, ModelRes512, CLP_clinical, TQN_Model
        image_encoder = ModelRes512(res_base_model='resnet50').to(device)
        tokenizer = AutoTokenizer.from_pretrained(
            bert_model_name, do_lower_case=True)
        text_encoder = CLP_clinical(
            bert_model_name=bert_model_name).to(device=device)
        model = TQN_Model().to(device)
        model.tokenizer = tokenizer

        checkpoint_path = "../pretrained/KAD_Models/KAD_1024/best_valid.pt"
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        image_state_dict = checkpoint['image_encoder']
        image_encoder.load_state_dict(image_state_dict)
        text_state_dict = checkpoint['text_encoder']
        text_encoder.load_state_dict(text_state_dict, strict=False)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        print('load checkpoint from %s' % checkpoint_path)

        model.eval()
        image_encoder.eval()
        text_encoder.eval()

        return (model, image_encoder, text_encoder)

    elif hparams.model_name == "gloria_chexpert":
        from cxrseg.third_party.gloria.load_original_gloria import load_gloria
        model = load_gloria()
        model.eval()
        return model

    elif hparams.model_name == "mgca_cnn":
        from cxrseg.third_party.mgca.mgca_module import MGCA
        checkpoint_path = "../pretrained/MGCA/resnet_50.ckpt"
        mgca = MGCA.load_from_checkpoint(checkpoint_path).to(device)
        mgca.eval()
        return mgca

    elif hparams.model_name == "mgca_vit":
        from cxrseg.third_party.mgca.mgca_module import MGCA
        checkpoint_path = "../pretrained/MGCA/vit_base.ckpt"
        mgca = MGCA.load_from_checkpoint(checkpoint_path).to(device)
        mgca.eval()
        return mgca

    elif hparams.model_name == "chexzero":
        from cxrseg.third_party.CheXzero.load_pretrained_model import load_pretrained_chexzero
        checkpoint_path = "../pretrained/CheXzero/best_64_0.0001_original_16000_0.861.pt"
        model = load_pretrained_chexzero(
            model_path=checkpoint_path, pretrained=True, context_length=77).to(device)
        model.eval()
        return model

    elif hparams.model_name == "afloc":
        checkpoint_path = "../pretrained/AFLoc/release.ckpt"
        from cxrseg.third_party.AFLoc.afloc.builder import load_model
        model = load_model(ckpt_path=checkpoint_path, device=device)
        model.eval()
        return model

    elif hparams.model_name == "our_medclip_s2":
        ckpt = torch.load(hparams.ckpt_path, map_location=device)
        model = SILCModule(dataset_dir=hparams.dataset_dir,
                           vision_model_name="resnet50").to(device)
        silc_ckpt = dict()
        for k, v in ckpt["model"].items():
            if "sem_seg_head.predictor.silc_model" in k:
                new_k = k.replace("sem_seg_head.predictor.silc_model.", "")
                silc_ckpt[new_k] = v
        model.load_state_dict(silc_ckpt)
        model.eval()
        return model


def main(hparams: Namespace):
    model = get_model(hparams)
    # sum(p.numel() for p in model[2].parameters()) / 1e6

    model_name = hparams.model_name
    if model_name == "biovil":
        mean, std = 0, 1
        imagesize = 480
        bert_type = "microsoft/BiomedVLP-CXR-BERT-specialized"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            bert_type, trust_remote_code=True)
        model.tokenizer = tokenizer
    elif model_name == "biovil_t":
        mean, std = 0, 1
        imagesize = 448
        bert_type = "microsoft/BiomedVLP-BioViL-T"
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            bert_type, trust_remote_code=True)
        model.tokenizer = tokenizer
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
        # bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        imagesize = 224
    elif model_name == "gloria_chexpert":
        mean, std = 0.5, 0.5
        bert_type = "emilyalsentzer/Bio_ClinicalBERT"
        imagesize = 224
    elif model_name == "medklip":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        imagesize = 224
        # tokenizer = model[0].tokenizer
        tokenizer = None
    elif model_name == "kad_resnet_224":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        imagesize = 224
        tokenizer = model[0].tokenizer
    elif model_name == "kad_resnet_512":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        imagesize = 512
        tokenizer = model[0].tokenizer
    elif model_name == "kad_resnet_1024":
        mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
        imagesize = 1024
        tokenizer = model[0].tokenizer
    elif model_name in ["mgca_cnn", "mgca_vit"]:
        mean, std = 0.5, 0.5
        imagesize = 224
    elif model_name == "chexzero":
        mean, std = (0.398, 0.398, 0.398), (0.327, 0.327, 0.327)
        imagesize = 224
    elif model_name == "afloc":
        mean, std = 0.5, 0.5
        imagesize = 224

    for dataset_name in hparams.dataset_list:
        print(f"Start evaluating {dataset_name}...")
        dataloader, pos_prompts, neg_prompts = create_zero_shot_cls_dataloader(
            dataset_dir=hparams.dataset_dir,
            dataset_name=dataset_name,
            batch_size=hparams.batch_size,
            num_workers=hparams.num_workers,
            imagesize=imagesize,
            mean=mean,
            std=std,
            prompt_style=hparams.prompt_style
        )
        if dataset_name in ["mimic_5x200", "chexpert_5x200"]:
            mode = "multiclass"
        else:
            mode = "multilabel"

        zero_shot_cls_evaluation(model, dataloader, pos_prompts, neg_prompts,
                                 dataset_name=dataset_name,
                                 save_dir=hparams.save_dir,
                                 model_name=hparams.model_name,
                                 mode=mode,
                                 prompt_style=hparams.prompt_style,
                                 use_negative=hparams.use_negative_prompt)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--dataset_dir", type=str,
                        default="/disk1/fywang/CXR_dataset")
    # default="/data1/r20user2/CXR_dataset")
    parser.add_argument("--use_negative_prompt", action="store_true")
    # use another conda environment if evaluating medclip
    parser.add_argument("--model_name", type=str, default="our_medclip_s2",
                        choices=["medclip_vit", "medclip_cnn", "convirt", "gloria_chexpert",
                                 "gloria", "biovil", "biovil_t", "our_medclip", "our_medclip_s2",
                                 "medklip", "kad_resnet_224", "kad_resnet_512", "kad_resnet_1024",
                                 "mgca_cnn", "mgca_vit", "chexzero", "afloc", 
                                 "random"])
    parser.add_argument("--prompt_style", type=str, default="xplainer",
                        choices=["xplainer", "biovil", "chexzero", "gloria"])
    parser.add_argument("--ckpt_path", type=str,
                        # default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_04_13_12_11_22/epoch=14-step=31500.ckpt")
                        # gloria
                        # default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_03_18_23_27_36/epoch=12-step=13715.ckpt")
                        # convirt
                        default="/home/fywang/Documents/CXRSeg/logs/medclip/ckpts/MedCLIP_2024_03_21_22_25_20/epoch=12-step=27417.ckpt")
    parser.add_argument("--save_dir", type=str,
                        default="/home/fywang/Documents/CXRSeg/evaluation_results/zero_shot_cls")
    parser.add_argument("--dataset_list", nargs="+",
                        default=["mimic_5x200", "chexpert_5x200", "chexpert", "nih", "padchest"])
    args = parser.parse_args()
    from pprint import pprint
    pprint(vars(args))
    main(args)
