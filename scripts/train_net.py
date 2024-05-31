import os
import torch
import copy
import ipdb
import itertools
import weakref
import logging
from typing import Any, Dict, List, Set
from datetime import datetime
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.projects.deeplab import add_deeplab_config, build_lr_scheduler
from detectron2.utils.logger import setup_logger
from detectron2.utils.events import CommonMetricPrinter, JSONWriter
from detectron2.evaluation import verify_results, DatasetEvaluators, COCOEvaluator
from detectron2.engine.defaults import DefaultPredictor, create_ddp_model, SimpleTrainer, AMPTrainer

from cxrseg import (
    add_cat_seg_config,
    CATSeg,
    DETRPanopticDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    CXRSegDatasetMapper,
    CXRSegEvaluator)
from cxrseg.utils.events import WandbWriter, setup_wandb
from cxrseg.datasets import build_detection_train_loader, build_detection_test_loader


class Trainer(DefaultTrainer):
    '''
    Extension of the Trainer class adapted to CATSeg.
    '''

    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        """
        super(DefaultTrainer, self).__init__()
        logger = logging.getLogger("detectron2")
        # setup_logger is not called for d2
        if not logger.isEnabledFor(logging.INFO):
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        # find unused parameters
        model = create_ddp_model(
            model, broadcast_buffers=False, find_unused_parameters=True)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        # Semantic segmentation evaluator for CXR
        evaluator_list.append(
            CXRSegEvaluator(dataset_name=dataset_name,
                            distributed=True,
                            output_dir=output_folder)
        )

        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.INPUT.DATASET_MAPPER_NAME == "cxrseg_semantic":
            mapper = CXRSegDatasetMapper(cfg, True)
        else:
            mapper = None

        dataset = None
        return build_detection_train_loader(cfg, mapper=mapper, dataset=dataset)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        if cfg.INPUT.DATASET_MAPPER_NAME == "cxrseg_semantic":
            mapper = CXRSegDatasetMapper(cfg, False)
        else:
            mapper = None

        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.

        It is now implemented by:
        ::
            return [
                CommonMetricPrinter(self.max_iter),
                JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
                TensorboardXWriter(self.cfg.OUTPUT_DIR),
            ]

        """
        # Here the default print/log frequency of each writer is used.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CommonMetricPrinter(self.max_iter),
            JSONWriter(os.path.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            WandbWriter(),
        ]

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)

    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)
                hyperparams = copy.copy(defaults)
                # no backbone now
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * \
                        cfg.SOLVER.BACKBONE_MULTIPLIER
                if "silc_model" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * \
                        cfg.SOLVER.CLIP_MULTIPLIER
                # for deformable detr

                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    # print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(
                        *[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    time_now =  datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cfg.OUTPUT_DIR = os.path.join(
        cfg.OUTPUT_DIR, time_now)
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_cat_seg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    # Setup logger for "ovseg" module
    if not args.eval_only:
        setup_wandb(cfg, args, name=time_now)
    setup_logger(
        output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="ovseg"
    )
    return cfg


def main(args):
    cfg = setup(args)
    torch.set_float32_matmul_precision("high")
    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )

        res = Trainer.test(cfg, model)
        if comm.is_main_process():
            verify_results(cfg, res)
        return res
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == '__main__':
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
