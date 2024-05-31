import numpy as np
import torch
from typing import List
import torch.nn.functional as F
from sklearn.metrics import classification_report, accuracy_score
from lightning import LightningModule
import ipdb
from dcxfm.utils.constants import CHEXPERT_COMPETITION_TASKS
from dcxfm.utils.lr_scheduler import linear_warmup_decay
from dcxfm.modeling.sdmp.datamodule import SDMPDataModule
from dcxfm.modeling.sdmp.ibot_modules import iBOTHead


class BaseLightningModule(LightningModule):
    ''' The base lightning model used for pretraining.'''

    def __init__(self,
                 vision_model_name: str = "microsoft/swin-base-patch4-window12-384",
                 text_model_name: str = "microsoft/BiomedVLP-CXR-BERT-general",
                 dataset_dir: str = "/data1/r20user2/CXR_dataset",
                 dataset_list: List = ["mimic-cxr"],
                 train_data_pct: float = 1.,
                 prompt_ensemble: bool = True,
                 imagesize: int = 512,
                 global_crops_number: int = 2,
                 local_crops_number: int = 4,
                 global_crops_size: int = 512,
                 local_crops_size: int = 224,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 lr: float = 2e-5,
                 weight_decay: float = 1e-4,
                 max_epochs: int = 100,
                 num_devices: int = 1,
                 accumulate_grad_batches: int = 1):
        super().__init__()

        self.vision_model_name = vision_model_name
        self.text_model_name = text_model_name
        self.dataset_dir = dataset_dir
        self.dataset_list = dataset_list
        self.train_data_pct = train_data_pct
        self.lr = lr
        self.weight_decay = weight_decay
        self.imagesize = imagesize
        self.prompt_ensemble = prompt_ensemble
        self.max_epochs = max_epochs
        self.warmup_epochs = int(0.2 * self.max_epochs)
        self.num_devices = num_devices
        self.accumulate_grad_batches = accumulate_grad_batches
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        self.setup_datamodule()

    def setup_datamodule(self):
        self.datamodule = SDMPDataModule(
            dataset_dir=self.dataset_dir,
            dataset_list=self.dataset_list,
            bert_type=self.text_model_name,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            imagesize=self.imagesize,
            train_data_pct=self.train_data_pct,
            global_crops_number=self.global_crops_number,
            local_crops_number=self.local_crops_number,
            global_crops_size=self.global_crops_size,
            local_crops_size=self.local_crops_size,
        )
        # this is approximately to 2086 (batch: 64, devices: 4)
        self.train_iters_per_epoch = len(
            self.datamodule.train_dataloader()) // (self.num_devices * self.accumulate_grad_batches)

    def _clip_loss(self, logits):
        # Note that temperature has been divided

        labels = torch.arange(logits.size(0)).type_as(logits).long()
        loss0 = F.cross_entropy(logits, labels)
        loss1 = F.cross_entropy(logits.t(), labels)
        loss = (loss0 + loss1) / 2

        return loss

    def _soft_clip_loss(self, logits_per_img, soft_label):
        '''take labels of images and sentences as a softlabel
        e.g., image_label = [1, 0, 1, -1], sentence_label = [0, 0, 1, -1]
        this pair has similarity as: 1 * 0 + 0 * 0 + 1 * 1 + -1 * -1 = 2.
        We will clamp the similarity into [-1,1], and take softmax as a soft-label.
        '''
        # when using InfoNCE-like loss
        image_loss = self._soft_xent_loss(
            logits_per_img, F.softmax(soft_label / 0.5, 1))
        caption_loss = self._soft_xent_loss(
            logits_per_img.T, F.softmax(soft_label.T / 0.5, 1))

        return (image_loss + caption_loss) / 2.

    def _soft_xent_loss(self, input, target):
        logprobs = torch.nn.functional.log_softmax(input, dim=1)
        return -(target * logprobs).sum() / input.shape[0]

    def training_step(self, batch, batch_idx):
        batch_size = batch['img_global'].size(0)
        loss_dict = self(
            input_ids=batch['input_ids'],
            img_global=batch['img_global'],
            img_local=batch['img_local'] if len(
                batch["img_local"]) > 0 else None,
            attention_mask=batch['attention_mask'],
            return_loss=True,
            img_labels=batch['img_labels'],
            text_labels=batch['text_labels'],
        )

        new_loss_dict = dict()
        for k, v in loss_dict.items():
            new_loss_dict[f"train_{k}"] = v

        self.log_dict(new_loss_dict, prog_bar=True, on_step=True, on_epoch=True,
                      batch_size=batch_size, sync_dist=True)

        return loss_dict["loss"]

    def validation_step(self, batch, batch_idx):
        batch_size = batch['img_global'].size(0)
        loss_dict = self(
            input_ids=batch['input_ids'],
            img_global=batch['img_global'],
            img_local=batch['img_local'] if len(
                batch["img_local"]) > 0 else None,
            attention_mask=batch['attention_mask'],
            return_loss=True,
            img_labels=batch['img_labels'],
            text_labels=batch['text_labels'],
        )
        new_loss_dict = dict()
        for k, v in loss_dict.items():
            new_loss_dict[f"val_{k}"] = v

        self.log_dict(new_loss_dict, prog_bar=True, on_step=False, on_epoch=True,
                      batch_size=batch_size, sync_dist=True)

        return loss_dict["loss"]

    def on_test_epoch_start(self) -> None:
        self.test_step_outputs = []

    def test_step(self, batch, batch_idx):
        pixel_values = batch['pixel_values']
        pos_prompt_inputs = batch['pos_prompt_inputs']
        neg_prompt_inputs = batch['neg_prompt_inputs']
        labels = batch['labels']
        outputs = self.zeroshot_forward(
            pixel_values, pos_prompt_inputs, neg_prompt_inputs)
        step_output = {
            "pos_disease_probs": outputs["pos_disease_probs"],
            "neg_disease_probs": outputs["neg_disease_probs"],
            "labels": labels
        }
        self.test_step_outputs.append(step_output)
        return outputs

    def on_test_epoch_end(self) -> None:
        '''
        Report metrics of zero-shot classification
        '''
        labels = []
        pos_disease_probs = dict()
        neg_disease_probs = dict()
        for class_name in CHEXPERT_COMPETITION_TASKS:
            pos_disease_probs[class_name] = []
            neg_disease_probs[class_name] = []

        for step_output in self.test_step_outputs:
            for class_name in CHEXPERT_COMPETITION_TASKS:
                pos_disease_probs[class_name].append(
                    step_output["pos_disease_probs"][class_name].cpu().detach().numpy())
                neg_disease_probs[class_name].append(
                    step_output["neg_disease_probs"][class_name].cpu().detach().numpy())
            labels.append(step_output["labels"].cpu().detach().numpy())

        labels = np.concatenate(labels, axis=0)
        for k, v in pos_disease_probs.items():
            pos_disease_probs[k] = np.concatenate(v, axis=0)
        for k, v in neg_disease_probs.items():
            neg_disease_probs[k] = np.concatenate(v, axis=0)

        pred_probs = np.zeros(
            (labels.shape[0], len(CHEXPERT_COMPETITION_TASKS)))
        for i, class_name in enumerate(CHEXPERT_COMPETITION_TASKS):
            pos_neg_scores = np.vstack(
                [pos_disease_probs[class_name], neg_disease_probs[class_name]]).T
            pred_probs[:, i] = pos_disease_probs[class_name]
        pred_labels = np.argmax(pred_probs, axis=1)

        acc = accuracy_score(labels, pred_labels)
        res = classification_report(
            labels, pred_labels, output_dict=True, zero_division=0)
        log_dict = dict()
        log_dict["test_acc"] = acc
        log_dict["test_macro_precision"] = res["macro avg"]["precision"]
        log_dict["test_macro_recall"] = res["macro avg"]["recall"]
        log_dict["test_macro_f1"] = res["macro avg"]["f1-score"]
        log_dict["test_weighted_precision"] = res["weighted avg"]["precision"]
        log_dict["test_weighted_recall"] = res["weighted avg"]["recall"]
        log_dict["test_weighted_f1"] = res["weighted avg"]["f1-score"]
        self.log_dict(log_dict, prog_bar=True, on_step=False, on_epoch=True,
                      batch_size=len(self.test_step_outputs), sync_dist=True)

    def zeroshot_forward(self, pixel_values, pos_prompt_inputs, neg_prompt_inputs):
        '''take image pixel values (after transform) and prompt_inputs
        (a dict of {'class1':{'input_ids':...,'attention_mask':,...}), 'class2':...}
        '''
        pos_disease_probs = dict()
        neg_disease_probs = dict()

        for class_name in pos_prompt_inputs:
            pos_inputs = {'pixel_values': pixel_values}
            for k in pos_prompt_inputs[class_name].keys():
                pos_inputs[k] = pos_prompt_inputs[class_name][k]

            neg_inputs = {'pixel_values': pixel_values}
            for k in neg_prompt_inputs[class_name].keys():
                neg_inputs[k] = neg_prompt_inputs[class_name][k]

            # TODO: set if we need the negative prompts
            pos_outputs = self(
                input_ids=pos_inputs['input_ids'],
                img_global=pos_inputs['pixel_values'],
                attention_mask=pos_inputs['attention_mask'],
                return_loss=False
            )
            pos_logits = pos_outputs['logits']
            neg_outputs = self(
                input_ids=neg_inputs['input_ids'],
                img_global=neg_inputs['pixel_values'],
                attention_mask=neg_inputs['attention_mask'],
                return_loss=False
            )
            neg_logits = neg_outputs['logits']

            probs = F.softmax(torch.stack(
                [pos_logits, neg_logits], dim=2) / 0.5, dim=2)
            pos_probs = probs[:, :, 0]
            neg_probs = probs[:, :, 1]
            joint_pos_probs = torch.log(pos_probs).mean(dim=1).exp()
            joint_neg_probs = torch.log(neg_probs).mean(dim=1).exp()

            pos_disease_probs[class_name] = joint_pos_probs
            neg_disease_probs[class_name] = joint_neg_probs

        outputs = {
            "pos_disease_probs": pos_disease_probs,
            "neg_disease_probs": neg_disease_probs
        }
        return outputs

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)

        warmup_steps = self.train_iters_per_epoch * self.warmup_epochs
        total_steps = self.train_iters_per_epoch * self.max_epochs

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                linear_warmup_decay(warmup_steps, total_steps, cosine=True),
            ),
            "interval": "step",
            "frequency": 1,
        }

        return [optimizer], [scheduler]
