import ipdb
from pprint import pprint
import os
from argparse import ArgumentParser, Namespace
import datetime
from dateutil import tz
import random
import numpy as np
import torch
from lightning import seed_everything, Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from lightning.pytorch.loggers import WandbLogger
# from cxrseg.modeling.our_medclip import MedCLIPModule
from dcxfm.modeling.sdmp import SDMPModule

'''
CUDA_VISIBLE_DEVICES=0,1,2,3 python pretrain_sdmp.py --num_devices 4 --use_i2t_loss \
    --loss_type soft_cont --use_local_loss --use_self_distil_loss \
    --train_data_pct 0.2 --lambda3 2
'''

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision('high')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT_DIR = os.path.join(BASE_DIR, "../")


def main(hparams: Namespace):

    # ------------------------
    # 1 INIT TRAINER
    # ------------------------
    now = datetime.datetime.now(tz.tzlocal())
    extension = now.strftime("%Y_%m_%d_%H_%M_%S")
    extension = f"SDMP_{extension}"
    ckpt_dir = os.path.join(
        REPO_ROOT_DIR, f"logs/sdmp/ckpts/{extension}")
    os.makedirs(ckpt_dir, exist_ok=True)
    callbacks = [
        LearningRateMonitor(logging_interval="step"),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir,
                        save_last=False, mode="min", save_top_k=4,
                        auto_insert_metric_name=True),
        EarlyStopping(monitor="val_loss", min_delta=0,
                      patience=5, verbose=False, mode="min")
    ]
    logger_dir = os.path.join(REPO_ROOT_DIR, "logs/sdmp")
    os.makedirs(logger_dir, exist_ok=True)
    wandb_logger = WandbLogger(
        project="sdmp", save_dir=logger_dir, name=extension)
    trainer = Trainer(
        max_epochs=hparams.max_epochs,
        accelerator="gpu",
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        # deterministic=True,
        devices=hparams.num_devices,
        strategy="ddp_find_unused_parameters_true",
        precision="bf16-mixed",
        callbacks=callbacks,
        logger=wandb_logger
    )

    # ------------------------
    # 2 INIT LIGHTNING MODEL and lightning datamodule
    # ------------------------
    hparams.exp_log_dir = os.path.join(
        REPO_ROOT_DIR, f"data/medclip/{extension}/exp_logs")

    if hparams.model_name == "sdmp":
        if hparams.ckpt_path:
            print(f"Loading model from {hparams.ckpt_path}")
            model = SDMPModule.load_from_checkpoint(
                hparams.ckpt_path, **hparams, strict=False)
        else:
            model = SDMPModule(**vars(hparams))
    else:
        raise NotImplementedError
    pprint(vars(hparams))

    datamodule = model.datamodule

    # ------------------------
    # 3 START TRAINING
    # ------------------------
    if not hparams.eval_only:
        trainer.fit(model, datamodule=datamodule)
    else:
        trainer.test(model, datamodule=datamodule)


if __name__ == '__main__':
    parser = ArgumentParser(description="Stage 1: pretrain MedCLIP.")
    parser.add_argument("--model_name", type=str, default="sdmp",
                        choices=["sdmp"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_data_pct", type=float, default=1.)
    parser.add_argument("--batch_size", type=int, default=24)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_devices", type=int, default=1)
    parser.add_argument("--max_epochs", type=int, default=50)
    parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    parser.add_argument("--imagesize", type=int, default=512)
    parser.add_argument("--dataset_dir", type=str,
                        default="/disk1/fywang/CXR_dataset")
    parser.add_argument("--dataset_list", nargs="+",
                        default=["mimic-cxr"])
    parser.add_argument("--ckpt_path", type=str,
                        default="")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--vision_model_name", type=str, default="resnet50")
    parser.add_argument("--text_model_name", type=str,
                        default="microsoft/BiomedVLP-CXR-BERT-general")
    parser.add_argument("--loss_type", type=str, default="nt_xent",
                        choices=["soft_cont", "nt_xent"])
    parser.add_argument("--use_patch_cont", action="store_true")
    parser.add_argument("--use_i2t_loss", action="store_true")
    parser.add_argument("--use_self_distil_loss", action="store_true")
    parser.add_argument("--use_local_loss", action="store_true")
    parser.add_argument("--lambda3", type=float, default=0.5)
    parser.add_argument("--local_loss_weight", type=float, default=1.,
                        help="Only used in GLoRIA, the weight for local loss.")
    hparams = parser.parse_args()

    seed_everything(hparams.seed)
    main(hparams)
