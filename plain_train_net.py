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

# BECAUSE IT WON'T IMPORT FOR SOME REASON
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from typing import Optional
def default_writers(output_dir: str, max_iter: Optional[int] = None):
    """
    Build a list of :class:`EventWriter` to be used.
    It now consists of a :class:`CommonMetricPrinter`,
    :class:`TensorboardXWriter` and :class:`JSONWriter`.
    Args:
        output_dir: directory to store JSON metrics and tensorboard events
        max_iter: the total number of iterations
    Returns:
        list[EventWriter]: a list of :class:`EventWriter` objects.
    """
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]

from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    return PascalVOCDetectionEvaluator(dataset_name)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np 

# TODO: ADD PATIENCE AND BEST MODEL, AND STOP IF PATIENCE EXCEEDED
class LossEvalHook:
  def __init__(self, eval_period, model, model_name, data_loader, checkpointer, patience=0):
    self._model = model
    self._model_name = model_name
    self._period = eval_period
    self._data_loader = data_loader
    self._patience = patience # patience is specified in evaluation period units (e.g 3 evaluation periods)
    self._cur_patience = 0
    self._checkpointer = checkpointer
    self._min_loss = float('inf') 
  
  def _do_loss_eval(self, cur_iter, storage):
    # Copying inference_on_dataset from evaluator.py
    total = len(self._data_loader)
    num_warmup = min(5, total - 1)
        
    start_time = time.perf_counter()
    total_compute_time = 0
    losses = []
    stop_early = False
    for idx, inputs in enumerate(self._data_loader):            
      if idx == num_warmup:
        start_time = time.perf_counter()
        total_compute_time = 0
      start_compute_time = time.perf_counter()
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      total_compute_time += time.perf_counter() - start_compute_time
      iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
      seconds_per_img = total_compute_time / iters_after_start
      if idx >= num_warmup * 2 or seconds_per_img > 5:
        total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
        eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
        log_every_n_seconds(
          logging.INFO,
          "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
            idx + 1, total, seconds_per_img, str(eta)
          ),
          n=5,
        )
      loss_batch = self._get_loss(inputs)
      losses.append(loss_batch)
    mean_loss = np.mean(losses)
    
    if mean_loss < self._min_loss:
      self._patience = 0
      self._min_loss = mean_loss
      self._checkpointer.save(self._model_name + "_" + str(cur_iter))
    else:
      self._cur_patience += 1
      if self._cur_patience > self._patience:
        stop_early = True
    storage.put_scalar('validation_loss', mean_loss)
    comm.synchronize()

    return (losses, stop_early)
          
  def _get_loss(self, data):
      # How loss is calculated on train_loop 
      metrics_dict = self._model(data)
      metrics_dict = {
          k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
          for k, v in metrics_dict.items()
      }
      total_losses_reduced = sum(loss for loss in metrics_dict.values())
      return total_losses_reduced
      
  def after_step(self, cur_iter, max_iter, storage):
    next_iter = cur_iter + 1
    is_final = next_iter == max_iter
    if is_final or (self._period > 0 and next_iter % self._period == 0):
        (_, stop_early) = self._do_loss_eval(cur_iter, storage)
        return stop_early
    return False

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    # scheduler = build_lr_scheduler(cfg, optimizer)
    # either this or "reduce on plateau", reduce on plateau has less hyperparams but might be worse? idk not much research on 
    # lr scheduling pros and cons.
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.1,step_size_up=5,mode="triangular2")

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
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    # CREATE EARLY STOPPING HOOK HERE
    early_stopping = LossEvalHook(
          cfg.TEST.EVAL_PERIOD,
          model,
          "best_model", # TODO: Change to configuration-dependent name e.g that encodes no. comp labels, etc.
          build_detection_test_loader(
            cfg,
            cfg.DATASETS.TEST[0],
            DatasetMapper(cfg,True)
          ),
          checkpointer,
          patience=0
        )
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        stop_early = early_stopping.after_step(0, 1, storage) # simulate final iter to guarantee running first time
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
            stop_early = early_stopping.after_step(iteration, max_iter, storage)
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

            if stop_early:
              break

from datasets import CustomDataset
import detectron2.data.datasets.pascal_voc as pascal_voc 

def setup(args):
    """
    Create configs and perform basic setups.
    """
    names = list(pascal_voc.CLASS_NAMES)
    splits = {
        "train": pascal_voc.load_voc_instances("VOC2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007", "train", names),
        "val": pascal_voc.load_voc_instances("VOC2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007", "val", names),
        "test": pascal_voc.load_voc_instances("VOC2007/voctest_06-nov-2007/VOCdevkit/VOC2007", "test", names),
      }
    ds = CustomDataset(names, "person", splits)
    (split_names, cfg, chosen_labels) = ds.subset("voc", 2, percentage=0.2)
    cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(split_names[0])) / cfg.SOLVER.IMS_PER_BATCH)) # = 1 epoch
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model, resume=args.resume)
    #res = do_test(cfg, model)
    #return res

import argparse 
import sys
def argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

# TODO: Abstract setup (config, metadata stuff)
# TODO: LR search

import numpy as np
# from: https://stackoverflow.com/questions/29346292/logarithmic-interpolation-in-python
def log_interp1d(xx, yy, kind='linear'):
    logx = np.log10(xx)
    logy = np.log10(yy)
    lin_interp = sp.interpolate.interp1d(logx, logy, kind=kind)
    log_interp = lambda zz: np.power(10.0, lin_interp(np.log10(zz)))
    return log_interp

def lr_search(splits, cfg, lr_min_pow=-5, lr_max_pow=-1, resolution=20, n_epochs=5):
  powers = np.linspace(lr_min_pow, lr_max_pow, resolution)
  lrs = 10 ** powers
  return lrs

if __name__ == "__main__":
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