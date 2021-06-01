# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.
This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.
In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".
Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)

from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import MetadataCatalog, DatasetCatalog

from detectron2.evaluation import PascalVOCDetectionEvaluator
from util.COCOEvaluator import COCOEvaluator # my modded version

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler # , buld_optimizer
from detectron2.utils.events import EventStorage

from util.helpers import save_sample, default_writers, load_csaw

logger = logging.getLogger("detectron2")

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np 

from util.loss_eval_hook import EarlyStoppingHook

from detectron2.data.samplers import InferenceSampler
from detectron2.data.common import DatasetFromList, MapDataset


from util.augmentor import DummyAlbuMapper
from util.helpers import build_optimizer

# TODO: Add a "no-checkpointer"-option for the lr search.
def do_train(cfg, model, resume=False, use_early_stopping=True, save_checkpoints=True):
    model.train()
    optimizer = build_optimizer(cfg, model) # TODO: This returns an SGD optimizer, maybe change to ADAM
    scheduler = build_lr_scheduler(cfg, optimizer)
    # either this or "reduce on plateau", reduce on plateau has less hyperparams but might be worse? idk not much research on 
    # lr scheduling pros and cons.
    #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.SOLVER.BASE_LR, max_lr=cfg.SOLVER.BASE_LR*2,step_size_up=5,mode="triangular2")

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training 
    data_loader = build_detection_train_loader(
      dataset=DatasetCatalog.get(cfg.DATASETS.TRAIN[0]), 
      mapper=DummyAlbuMapper(cfg, is_train=True),
      total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
      num_workers=cfg.DATALOADER.NUM_WORKERS
    )

    # CREATE EARLY STOPPING HOOK HERE
    early_stopping = EarlyStoppingHook(
          cfg,
          cfg.TEST.EVAL_PERIOD if use_early_stopping else 0,
          model,
          "best_model", # TODO: Change to configuration-dependent name e.g that encodes no. comp labels, etc.
          cfg.DATASETS.TEST[0],
          checkpointer,
          patience=1,
          save_checkpoints=save_checkpoints
        )
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        #if use_early_stopping:
        #  model.eval()
        #  stop_early = early_stopping.after_step(0, 1, storage) # simulate final iter to guarantee running first time
        #  model.train()
        #else:
        #  stop_early = False
        stop_early = False
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            # REGULARLY CHECK/APPLY HOOK HERE
            # if (
            #     cfg.TEST.EVAL_PERIOD > 0
            #     and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
            #     and iteration != max_iter - 1
            # ):
            if iteration % 100 == 0:
              model.eval()
              save_sample(cfg, model, data[0], "../../samples/sample.jpg")
              model.train()
            
            # TODO: Uncomment for early stopping
            #model.eval()
            #stop_early = early_stopping.after_step(iteration, max_iter, storage)
            #model.train()
            
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            logger.info("hi")
            
            if save_checkpoints:
              periodic_checkpointer.step(iteration)

            if stop_early and use_early_stopping:
              break
    return early_stopping._latest_loss # for learning rate evaluation

from util.datasets import CustomDataset
import detectron2.data.datasets.pascal_voc as pascal_voc 

# TODO: Abstract setup (config, metadata stuff)
# TODO: LR search

import numpy as np

# construct dataset base dictionaries of each split
# and return a CustomDataset object
import scripts.generate_obj_csaws as csaws

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
import random 

#TODO: Look into data loaders and add augmentations to them.