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
    dataset_name = cfg.DATASETS.TEST[1]
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

from loss_eval_hook import LossEvalHook


from detectron2.data.samplers import InferenceSampler
from detectron2.data.common import DatasetFromList, MapDataset

# builds a loader that iterates over the dataset once,
# like build_detection_test_loader, but with arbitrary
# batch size.
def build_eval_loader(dataset, *, mapper, sampler=None, num_workers=0, batch_size=1):
  if isinstance(dataset, list):
    dataset = DatasetFromList(dataset, copy=False)
  if mapper is not None:
    dataset = MapDataset(dataset, mapper)
  if sampler is None:
    sampler = InferenceSampler(len(dataset))
  # Always use 1 image per worker during inference since this is the
  # standard when reporting inference time in papers.
  batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)
  data_loader = torch.utils.data.DataLoader(
    dataset,
    num_workers=num_workers,
    batch_sampler=batch_sampler,
    collate_fn=lambda batch: batch, # identity function
  )

  return data_loader

from augmentor import DummyAlbuMapper

# TODO: Add a "no-checkpointer"-option for the lr search.
def do_train(cfg, model, resume=False, use_early_stopping=True):
    model.train()
    optimizer = build_optimizer(cfg, model) # TODO: This returns an SGD optimizer, maybe change to ADAM
    # scheduler = build_lr_scheduler(cfg, optimizer)
    # either this or "reduce on plateau", reduce on plateau has less hyperparams but might be worse? idk not much research on 
    # lr scheduling pros and cons.
    scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=cfg.SOLVER.BASE_LR, max_lr=cfg.SOLVER.BASE_LR*2,step_size_up=5,mode="triangular2")

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

    # augmentations are those that are the strategies, those are the ones
    # that should be used. Transformations are deterministic operations 
    # used by the (potentially random) augmentations. 
    #
    # NOTE: I don't think I'll implement elastic transform, mostly needed for 
    # medical though since natural scale objects don't usually warp like that
    # in photos.
    data_loader = build_detection_train_loader(
      dataset=DatasetCatalog.get(cfg.DATASETS.TRAIN[0]), 
      mapper=DummyAlbuMapper(cfg, is_train=True),
      total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
      num_workers=cfg.DATALOADER.NUM_WORKERS
    )

    # CREATE EARLY STOPPING HOOK HERE
    early_stopping = LossEvalHook(
          cfg.TEST.EVAL_PERIOD,
          model,
          "best_model", # TODO: Change to configuration-dependent name e.g that encodes no. comp labels, etc.
          build_eval_loader( # test loader would use batch size 1 for benchmarking, very slow
            DatasetCatalog.get(cfg.DATASETS.TEST[0]), #TODO: idk WHY but the early stopping code goes past the size of the validation set... either the test set (which is larger) is used, or i accidentally constructed too many epochs.... OOOOOOOOOOOORRRRRRRRRRRRRRRRR the training data loader is literally infinite, i.e it loops forever! LMAO
            batch_size=cfg.SOLVER.IMS_PER_BATCH,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            mapper=DatasetMapper(cfg,True), #do_train=True means we are in training mode.
            #aspect_ratio_grouping=False
          ),
          checkpointer,
          patience=0
        )
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        #stop_early = early_stopping.after_step(0, 1, storage) # simulate final iter to guarantee running first time
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
              with torch.no_grad():
                outputs = model(data)
                #print(data[0]["image"])
                img = data[0]["image"]
                # print(data[0])
                # exit(0)
                from detectron2.data import detection_utils as utils
                orig = utils.read_image(data[0]['file_name'], format="RGB")
                #print(img.detach().cpu().numpy()[:,:,::-1])
                # permute C, H, W format to H, W, C format and flip C from BGR to RGB
                v = Visualizer(img.permute(1,2,0).numpy()[:,:,::-1], # = img[:,:,::-1] in numpy
                #v = Visualizer(orig[:,:,::-1], # = img[:,:,::-1] in numpy
                    metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
                    scale=1)
                #out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
                out = v.draw_dataset_dict(data[0])
                cv2.imshow('sample.jpg', out.get_image())
                cv2.waitKey()
                cv2.destroyWindow('sample.jpg')
                exit(0)
              model.train()
            stop_early = early_stopping.after_step(iteration, max_iter, storage)
            # Compared to "train_net.py", the test results are not dumped to EventStorage
            comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)

            if stop_early and use_early_stopping:
              break
    return early_stopping._latest_loss # for learning rate evaluation

from datasets import CustomDataset
import detectron2.data.datasets.pascal_voc as pascal_voc 

def setup(args):
    """
    Create configs and perform basic setups.
    """
    if args.dataset == "PascalVOC2007":
      names = list(pascal_voc.CLASS_NAMES)
      ds = CustomDataset(names, "person", splits)
      (split_names, cfg, chosen_labels) = ds.subset("voc", 2, percentage=0.2)
      cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(split_names[0])) / cfg.SOLVER.IMS_PER_BATCH)) # = 1 epoch
      #cfg.freeze()
      default_setup(
          cfg, args
      )  # if you don't like any of the default setup, write your own setup code
      return cfg
    else:
      raise NotImplementedError(
            f"Dataset {args.dataset} is not supported"
        )

# TODO: Abstract setup (config, metadata stuff)
# TODO: LR search

import numpy as np

def lr_search(cfg, lr_min_pow=-5, lr_max_pow=-2, resolution=20, n_epochs=5):
  powers = np.linspace(lr_min_pow, lr_max_pow, resolution)
  lrs = 10 ** powers
  best_val = float('inf')
  best_lr = 0
  for lr in lrs:
    # do setup 
    cfg.SOLVER.BASE_LR = float(lr)
    cfg.SOLVER.MAX_ITER = n_epochs * int(round(len(DatasetCatalog.get(cfg.DATASETS.TEST[0])) / cfg.SOLVER.IMS_PER_BATCH))
    model = build_model(cfg)
    # train 5 epochs
    val_loss = do_train(cfg, model, resume=False, use_early_stopping=False) # TODO: Use validation dataset, maybe by modding the config or adding option to do_train
    # calc val loss at the end
    if val_loss < best_val:
      best_val = val_loss
      best_lr = lr

  print("LR Search done: Best LR is", best_lr, "with validation loss", val_loss)
  return best_lr

# construct dataset base dictionaries of each split
# and return a CustomDataset object
def extract_dataset(dataset_name, main_label):
  if dataset_name == "PascalVOC2007":
    labels = list(pascal_voc.CLASS_NAMES)
    base_dataset = {
      "train": pascal_voc.load_voc_instances("VOC2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007", "train", labels),
      "val": pascal_voc.load_voc_instances("VOC2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007", "val", labels),
      "test": pascal_voc.load_voc_instances("VOC2007/voctest_06-nov-2007/VOCdevkit/VOC2007", "test", labels),
    }
    return CustomDataset(labels, main_label, base_dataset)
  else:
    raise NotImplementedError(f"Dataset {args.dataset} is not supported")

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
import random 
import cv2
def save_sample(img):
  predictor = DefaultPredictor(cfg)
  dataset_dicts = DatasetCatalog.get(dataset_name)
  for d in random.sample(dataset_dicts, 3):    
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
    v = Visualizer(im[:, :, ::-1],
                    metadata=MetadataCatalog.get(dataset_name), 
                    scale=0.5)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imshow(out.get_image()[:, :, ::-1])

#TODO: Look into data loaders and add augmentations to them.
def base_experiment(dataset):
  main_label = args.main_label

  # no complementaries
  ds, _ = dataset.subset(args.dataset + "_no_complementary_labels")

  # build config for it once in beginning (can do it once in 
  # the beginning since dataset will be the same for this 
  # experiment, no randomness involved)

  cfg = get_cfg()

  # These are the augmentations im thinking abt using. 
  # Basically shift/scale/rotate, some sort of brightness/contrast manip,
  # flip, cutting out a rectangle, and blurring the image. CLAHE and blur 
  # might contradict each other a bit which in their purpose but at the 
  # same time, one provides better brightness variation and one provides 
  # blur to focus on the overall picture. So in a sense they complement 
  # each other as well.
  cfg.INPUT.ALBUMENTATIONS = "./augs.json"

  # default is BGR for NO reason except it's nice with opencv. 
  # HOWEVER Visualizer wants RGB to conform with Matplotlib. 
  # BUT I want RGB for albumentations without copying and flipping the tensor manually.
  #cfg.INPUT.FORMAT = "RGB"
  cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"))
  cfg.DATASETS.TRAIN = (ds[0],) # training name
  cfg.DATASETS.TEST = (ds[1], ds[2]) # validation, test names
  cfg.DATALOADER.NUM_WORKERS = 8
  
  cfg.SOLVER.IMS_PER_BATCH = 2 # batch size is 2 images due to limitations  
  # lr_cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(split_names[0])) / cfg.SOLVER.IMS_PER_BATCH)) # = 1 epoch
  cfg.TEST.EVAL_PERIOD = 0 # only check validation loss at the end of the lr search
  
  #TODO: Also look into learning rate schedulers (i.e what type of decay/changes in base lr)
  
  # this will vary for the subset experiments. Also, 
  # detectron2 removes unannotated images by default
  # but only working on images with main label works 
  # around that problem. (the math expression results
  # in 1 epoch of training)
  cfg.SOLVER.MAX_ITER = 40 * int(round(len(DatasetCatalog.get(ds[0])) / cfg.SOLVER.IMS_PER_BATCH))
  
  #TODO: LR SCHEDULING (which scheduler, whether decay should be applied etc)

  cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # number of complementary labels + main label
  cfg.MODEL.RETINANET.NUM_CLASSES = 1  # number of complementary labels + main label

  cfg.TEST.EVAL_PERIOD = 0
  print("Entering lr search... ")
  lr = lr_search(cfg, resolution=2, n_epochs=1)
  print("lr search finished, optimal lr is", lr)
  cfg.SOLVER.BASE_LR = float(lr) # could instead be assigned to cfg in lr_search but whatevs
  for i in range(2): # repeat many times
    model = build_model(cfg)
    # set evaluation to occur every epoch instead of only in end
    cfg.SOLVER.MAX_ITER = 40 * int(round(len(DatasetCatalog.get(cfg.DATASETS.TRAIN[0])) / cfg.SOLVER.IMS_PER_BATCH))
    cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(ds[0])) / cfg.SOLVER.IMS_PER_BATCH))
    do_train(cfg, model)
    #do_test(cfg, model)
  pass

def leave_one_out(dataset, args):
  pass 

def main(args):
  # if args.create_all_datasets:
  # * create subsets for experiments
  # * do lr search for each of them (no intermediate val_loss)

  dataset_name = args.dataset 
  main_label = args.main_label
  base_dataset = extract_dataset(dataset_name, main_label)
  
  # default_setup literally only sets the cfg rng seed, the output directory, and whether cudnn.benchmark should be used.
  # I only load it because of the setup.
  base_cfg = get_cfg()
  base_cfg.SEED = args.seed

  # https://towardsdatascience.com/properly-setting-the-random-seed-in-machine-learning-experiments-7da298d1320b

  # set seed for all frameworks used (python hashing, python random, numpy, pytorch/detectron2)
  # also sets up logger and some cudnn benchmark thingy that idk. 
  # TODO: Make sure the seeding is redone before each experiment
  # (i.e once before base experiments, once before leave-one-out, 
  # once before varying labels etc). Only once tho, not for each 
  # repetition of each experiment!
  default_setup(base_cfg, args)
  
  print("Dataset loaded successfully, basic configuration completed.")
  if args.base:
    print("Entering base experiment...")
    base_experiment(base_dataset)
    print("Base experiment finished!")

  # cfg = setup(args)

  # model = build_model(cfg)
  # logger.info("Model:\n{}".format(model))
  # if args.eval_only:
  #     DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
  #         cfg.MODEL.WEIGHTS, resume=args.resume
  #     )
  #     return do_test(cfg, model)

  # distributed = comm.get_world_size() > 1
  # if distributed:
  #     model = DistributedDataParallel(
  #         model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
  #     )

  # do_train(cfg, model, resume=args.resume)
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
    parser.add_argument("--dataset", default="PascalVOC2007", help="dataset used for training and evaluation")
    parser.add_argument("--main-label", default="person", help="main label used for training and evaluation")
 
    parser.add_argument("--seed", type=int, default=random.randint(0,1000), help="seed used for randomization")
    
    parser.add_argument("--sample", action="store_true", default=False, help="perform sample inference on some images")

    # experiments to run in one sitting
    parser.add_argument("--base", action="store_true", default=False, help="perform base experiment")
    parser.add_argument("--leave-one-out", action="store_true", help="perform leave one out experiment")
    parser.add_argument("--vary-data", action="store_true", help="perform varying data experiments")
    parser.add_argument("--vary-labels", action="store_true", help="perform varying complementary labels experiments")

    parser.add_argument("--num-comp-labels", type=int, default=0, help="number of complementary labels for vary label experiments")
    parser.add_argument("--dataset-fraction", type=float, default=0.5, help="fraction of data to use for experiments; default half for all non-dataset related experiments")

    # TODO: Maybe an argument for datasets directory.

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

if __name__ == "__main__":
    args = argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )