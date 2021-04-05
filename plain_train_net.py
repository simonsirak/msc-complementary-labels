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

# TODO: ADD PATIENCE AND BEST MODEL, AND STOP IF PATIENCE EXCEEDED
class LossEvalHook(HookBase):
  def __init__(self, eval_period, model, data_loader):
    self._model = model
    self._period = eval_period
    self._data_loader = data_loader
  
  def _do_loss_eval(self):
    # Copying inference_on_dataset from evaluator.py
    total = len(self._data_loader)
    num_warmup = min(5, total - 1)
        
    start_time = time.perf_counter()
    total_compute_time = 0
    losses = []
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
    self.trainer.storage.put_scalar('validation_loss', mean_loss)
    comm.synchronize()

    return losses
          
  def _get_loss(self, data):
      # How loss is calculated on train_loop 
      metrics_dict = self._model(data)
      metrics_dict = {
          k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
          for k, v in metrics_dict.items()
      }
      total_losses_reduced = sum(loss for loss in metrics_dict.values())
      return total_losses_reduced
      
  def after_step(self):
      next_iter = self.trainer.iter + 1
      is_final = next_iter == self.trainer.max_iter
      if is_final or (self._period > 0 and next_iter % self._period == 0):
          self._do_loss_eval()
      self.trainer.storage.put_scalars(timetest=12)

def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

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
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg,True)
            )
        )
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
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

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            # REGULARLY CHECK/APPLY HOOK HERE
            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

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
    (split_names, cfg, chosen_labels) = ds.subset("voc", 2, percentage=0.5)
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
    return do_test(cfg, model)


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