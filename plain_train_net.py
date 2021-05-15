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
from COCOEvaluatorMODDED import COCOEvaluator # my modded version

from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler # , buld_optimizer
from detectron2.utils.events import EventStorage

from helpers import save_sample, default_writers

logger = logging.getLogger("detectron2")


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    return PascalVOCDetectionEvaluator(dataset_name)

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
from helpers import build_optimizer
# TODO: Add a "no-checkpointer"-option for the lr search.
def do_train(cfg, model, resume=False, use_early_stopping=True):
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
          patience=1
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
            if iteration % 100 == 0:
              model.eval()
              save_sample(cfg, model, data[0], "sample.jpg")
              model.train()
            
            # TODO: Uncomment for early stopping
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

# TODO: Abstract setup (config, metadata stuff)
# TODO: LR search

import numpy as np

def lr_search(cfg, lr_min_pow=-5, lr_max_pow=-2, resolution=20, n_epochs=5):
  powers = np.linspace(lr_min_pow, lr_max_pow, resolution)
  lrs = 10 ** powers
  best_val = float('inf')
  best_lr = 0
  losses = []
  for lr in lrs:
    # do setup 
    cfg.SOLVER.BASE_LR = float(lr)
    cfg.SOLVER.MAX_ITER = n_epochs * int(round(len(DatasetCatalog.get(cfg.DATASETS.TEST[0])) / cfg.SOLVER.IMS_PER_BATCH))
    model = build_model(cfg)
    # train 5 epochs
    val_loss = do_train(cfg, model, resume=False, use_early_stopping=True) # TODO: Use validation dataset, maybe by modding the config or adding option to do_train
    #losses.append() # TODO: Add logging of loss every n:th epoch during training.
    # calc val loss at the end
    if val_loss < best_val:
      best_val = val_loss
      best_lr = lr
    print("Tested LR", lr, "with validation loss", val_loss)
    

  print("LR Search done: Best LR is", best_lr, "with validation loss", best_val)
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
import random 

def evaluate(cfg, model):
  #cfg.MODEL.WEIGHTS = os.path.join("./output", "best_model.pth")  # path to the model we just trained
  #cfg.MODEL.WEIGHTS = os.path.join("./experimental_results/base_with_images_without_annotations/output", "best_model_7499.pth")  # path to the model we just trained
  #cfg.MODEL.WEIGHTS = os.path.join("./experimental_results/base_with_filter_annotations/output", "best_model_4814.pth")  # path to the model we just trained
  #cfg.MODEL.WEIGHTS = os.path.join("./experimental_results/new_base_with_annotations_using_1e-4lr_patience1/output", "best_model_2499.pth")  # path to the model we just trained
  # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set a custom testing threshold
  # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5   # set a custom testing threshold
  
  #model = build_model(cfg)
  #DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # load a file, usually from cfg.MODEL.WEIGHTS
  #model.eval()

  # build data loader, essentially equivalent to test loader but 
  # with arbitrary batch size because inference time is not a metric I want.
  data_loader = build_eval_loader( # test loader would use batch size 1 for benchmarking, very slow
    DatasetCatalog.get(cfg.DATASETS.TEST[1]), #TODO: idk WHY but the early stopping code goes past the size of the validation set... either the test set (which is larger) is used, or i accidentally constructed too many epochs.... OOOOOOOOOOOORRRRRRRRRRRRRRRRR the training data loader is literally infinite, i.e it loops forever! LMAO
    batch_size=cfg.SOLVER.IMS_PER_BATCH,
    num_workers=cfg.DATALOADER.NUM_WORKERS,
    mapper=DatasetMapper(cfg,False), #do_train=True means we are in training mode.
    #aspect_ratio_grouping=False
  )

  with torch.no_grad():
    def get_all_inputs_outputs():
      for data in data_loader:
        yield data, model(data)

    evaluator = COCOEvaluator(cfg.DATASETS.TEST[1], output_dir=cfg.OUTPUT_DIR, distributed=False, tasks=("bbox",))
    evaluator.reset()
    for inputs, outputs in get_all_inputs_outputs():
      evaluator.process(inputs, outputs)
    print("begin coco evaluation...")
    eval_results = evaluator.evaluate()
    print("finished coco evaluation!")
#TODO: Look into data loaders and add augmentations to them.
def base_experiment(dataset):
  main_label = args.main_label

  # no complementaries
  ds, _ = dataset.subset(args.dataset + "_no_complementary_labels", nb_comp_labels=3)

  cfg = get_cfg()

  cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"))
  
  cfg.INPUT.ALBUMENTATIONS = "./augs.json"
  cfg.INPUT.FORMAT = "BGR"

  # enables mixed precision, not super useful on my local GPU but might be free 
  # performance boost on remote!
  cfg.SOLVER.AMP.ENABLED = True

  cfg.DATASETS.TRAIN = (ds[0],) # training name
  cfg.DATASETS.TEST = (ds[1], ds[2]) # validation, test names
  cfg.DATALOADER.NUM_WORKERS = 8
  
  cfg.SOLVER.IMS_PER_BATCH = 2 # batch size is 2 images due to limitations  
  # lr_cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(split_names[0])) / cfg.SOLVER.IMS_PER_BATCH)) # = 1 epoch
  cfg.TEST.EVAL_PERIOD = 0 # only check validation loss at the end of the lr search
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # NOTE: Cannot do this and evaluate in base experiment as well, must rebuild model with 0.05.
  
  #TODO: Also look into learning rate schedulers (i.e what type of decay/changes in base lr)
  
  # this will vary for the subset experiments. Also, 
  # detectron2 removes unannotated images by default
  # but only working on images with main label works 
  # around that problem. (the math expression results
  # in 1 epoch of training)
  #cfg.SOLVER.MAX_ITER = 40 * int(round(len(DatasetCatalog.get(ds[0])) / cfg.SOLVER.IMS_PER_BATCH))
  #cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True 
  #TODO: LR SCHEDULING (which scheduler, whether decay should be applied etc)

  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label
  cfg.MODEL.RETINANET.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label

  cfg.TEST.EVAL_PERIOD = 0
  # print("Entering lr search... ")
  # lr = lr_search(cfg, resolution=2, n_epochs=1)
  # print("lr search finished, optimal lr is", lr)
  cfg.SOLVER.BASE_LR = float(1e-4) # could instead be assigned to cfg in lr_search but whatevs
  #cfg.SOLVER.STEPS = (1000, 2000)
  for i in range(1): # repeat many times
    model = build_model(cfg)
    # set evaluation to occur every epoch instead of only in end
    # cfg.SOLVER.MAX_ITER = 40 * int(round(len(DatasetCatalog.get(cfg.DATASETS.TRAIN[0])) / cfg.SOLVER.IMS_PER_BATCH))
    cfg.TEST.EVAL_PERIOD = 5000
    do_train(cfg, model)

    model.eval()
    evaluate(cfg, model)
    model.train()
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
  print(base_dataset.base_dict_func["train"][0]["file_name"])
  
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

  if args.lr:
    main_label = args.main_label

    # no complementaries
    ds, _ = base_dataset.subset(args.dataset, nb_comp_labels=0)

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"))
    
    cfg.INPUT.ALBUMENTATIONS = "./augs.json"
    cfg.INPUT.FORMAT = "BGR"

    # enables mixed precision, not super useful on my local GPU but might be free 
    # performance boost on remote!
    cfg.SOLVER.AMP.ENABLED = True

    cfg.DATASETS.TRAIN = (ds[0],) # training name
    cfg.DATASETS.TEST = (ds[1], ds[2]) # validation, test names
    cfg.DATALOADER.NUM_WORKERS = 8
    
    cfg.SOLVER.IMS_PER_BATCH = 2 # batch size is 2 images due to limitations  
    # lr_cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(split_names[0])) / cfg.SOLVER.IMS_PER_BATCH)) # = 1 epoch
    cfg.TEST.EVAL_PERIOD = 500 # only check validation loss at the end of the lr search
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # for evaluation
    
    #TODO: Also look into learning rate schedulers (i.e what type of decay/changes in base lr)
    
    # this will vary for the subset experiments. Also, 
    # detectron2 removes unannotated images by default
    # but only working on images with main label works 
    # around that problem. (the math expression results
    # in 1 epoch of training)
    cfg.SOLVER.MAX_ITER = 40 * int(round(len(DatasetCatalog.get(ds[0])) / cfg.SOLVER.IMS_PER_BATCH))
    #cfg.SOLVER.CLIP_GRADIENTS.ENABLED = True 
    #TODO: LR SCHEDULING (which scheduler, whether decay should be applied etc)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # number of complementary labels + main label
    cfg.MODEL.RETINANET.NUM_CLASSES = 1  # number of complementary labels + main label

    cfg.SOLVER.STEPS = (1000, 2000)
    best_lr = lr_search(cfg, resolution=10, n_epochs=4)

    print("best lr is", best_lr)
  if args.base:
    print("Entering base experiment...")
    base_experiment(base_dataset)
    print("Base experiment finished!")

  if args.eval:
    main_label = args.main_label
    # no complementaries
    ds, _ = base_dataset.subset(args.dataset + "_no_complementary_labels")

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"))
    print("WEIGHTS = ", cfg.MODEL.WEIGHTS)
    cfg.INPUT.ALBUMENTATIONS = "./augs.json"
    cfg.INPUT.FORMAT = "BGR"

    # enables mixed precision, not super useful on my local GPU but might be free 
    # performance boost on remote!
    cfg.SOLVER.AMP.ENABLED = True

    cfg.DATASETS.TRAIN = (ds[0],) # training name
    cfg.DATASETS.TEST = (ds[1], ds[2]) # validation, test names
    cfg.DATALOADER.NUM_WORKERS = 8
    
    cfg.SOLVER.IMS_PER_BATCH = 2 # batch size is 2 images due to limitations  
    # lr_cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(split_names[0])) / cfg.SOLVER.IMS_PER_BATCH)) # = 1 epoch
    cfg.TEST.EVAL_PERIOD = 0 # only check validation loss at the end of the lr search
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 # for visualization, use 0.5; for evaluation, use 0.05
    
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
    # print("Entering lr search... ")
    # lr = lr_search(cfg, resolution=2, n_epochs=1)
    # print("lr search finished, optimal lr is", lr)
    cfg.SOLVER.BASE_LR = float(1e-4) # could instead be assigned to cfg in lr_search but whatevs
    #cfg.MODEL.WEIGHTS = os.path.join("./experimental_results/testing/output", "best_model_999.pth")  # path to the model we just trained
    #print(cfg.MODEL.WEIGHTS)
    #cfg.MODEL.WEIGHTS = os.path.join("./output", "best_model.pth")  # path to the model we just trained
    #cfg.MODEL.WEIGHTS = os.path.join("./experimental_results/base_with_images_without_annotations/output", "best_model_7499.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join("./experimental_results/base_with_filter_annotations/output", "best_model_4814.pth")  # path to the model we just trained
    #cfg.MODEL.WEIGHTS = os.path.join("./experimental_results/new_base_with_annotations_using_1e-4lr_patience1/output", "best_model_2499.pth")  # path to the model we just trained
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set a custom testing threshold
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5   # set a custom testing threshold
    
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # load a file, usually from cfg.MODEL.WEIGHTS
    model.eval()

    dataset_dicts = DatasetCatalog.get(ds[2])
    data_loader = build_detection_train_loader(
      dataset=dataset_dicts, 
      mapper=DummyAlbuMapper(cfg, is_train=False),
      total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
      num_workers=cfg.DATALOADER.NUM_WORKERS
    )
    for data, iteration in zip(data_loader, range(len(dataset_dicts))):
      mapped_data = data[0]
      save_sample(cfg, model, mapped_data, "sample " + str(iteration), show=True)
      if iteration == 2:
        break

  if args.coco:
    main_label = args.main_label
    # no complementaries
    ds, _ = base_dataset.subset(args.dataset, identity=True)

    cfg = get_cfg()

    cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_C4.yaml"))
    print("WEIGHTS = ", cfg.MODEL.WEIGHTS)
    cfg.INPUT.ALBUMENTATIONS = "./augs.json"
    cfg.INPUT.FORMAT = "BGR"

    # enables mixed precision, not super useful on my local GPU but might be free 
    # performance boost on remote!
    cfg.SOLVER.AMP.ENABLED = True

    cfg.DATASETS.TRAIN = (ds[0],) # training name
    cfg.DATASETS.TEST = (ds[1], ds[2]) # validation, test names
    cfg.DATALOADER.NUM_WORKERS = 8
    
    cfg.SOLVER.IMS_PER_BATCH = 2 # batch size is 2 images due to limitations  
    # lr_cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(split_names[0])) / cfg.SOLVER.IMS_PER_BATCH)) # = 1 epoch
    cfg.TEST.EVAL_PERIOD = 0 # only check validation loss at the end of the lr search
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # for visualization, use 0.5; for evaluation, use 0.05
    
    #TODO: Also look into learning rate schedulers (i.e what type of decay/changes in base lr)
    
    # this will vary for the subset experiments. Also, 
    # detectron2 removes unannotated images by default
    # but only working on images with main label works 
    # around that problem. (the math expression results
    # in 1 epoch of training)
    cfg.SOLVER.MAX_ITER = 40 * int(round(len(DatasetCatalog.get(ds[0])) / cfg.SOLVER.IMS_PER_BATCH))
    
    #TODO: LR SCHEDULING (which scheduler, whether decay should be applied etc)

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label
    cfg.MODEL.RETINANET.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label

    cfg.TEST.EVAL_PERIOD = 0
    # print("Entering lr search... ")
    # lr = lr_search(cfg, resolution=2, n_epochs=1)
    # print("lr search finished, optimal lr is", lr)
    cfg.SOLVER.BASE_LR = float(1e-4) # could instead be assigned to cfg in lr_search but whatevs
    #cfg.MODEL.WEIGHTS = os.path.join("./experimental_results/testing/output", "best_model_999.pth")  # path to the model we just trained
    #print(cfg.MODEL.WEIGHTS)
    
    #cfg.MODEL.WEIGHTS = os.path.join("./output", "best_model.pth")  # path to the model we just trained
    #cfg.MODEL.WEIGHTS = os.path.join("./experimental_results/base_with_images_without_annotations/output", "best_model_7499.pth")  # path to the model we just trained
    cfg.MODEL.WEIGHTS = os.path.join("./", "model_final_b1acc2.pkl")  # path to the model we just trained
    #cfg.MODEL.WEIGHTS = os.path.join("./experimental_results/base_with_filter_annotations/output", "best_model_4814.pth")  # path to the model we just trained
    #cfg.MODEL.WEIGHTS = os.path.join("./experimental_results/filter_empty_annotations_2_labels/output", "best_model.pth")  # path to the model we just trained
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.75   # set a custom testing threshold
    # cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5   # set a custom testing threshold
    
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)  # load a file, usually from cfg.MODEL.WEIGHTS
    model.eval()

    # build data loader, essentially equivalent to test loader but 
    # with arbitrary batch size because inference time is not a metric I want.
    data_loader = build_eval_loader( # test loader would use batch size 1 for benchmarking, very slow
      DatasetCatalog.get(cfg.DATASETS.TEST[1]), #TODO: idk WHY but the early stopping code goes past the size of the validation set... either the test set (which is larger) is used, or i accidentally constructed too many epochs.... OOOOOOOOOOOORRRRRRRRRRRRRRRRR the training data loader is literally infinite, i.e it loops forever! LMAO
      batch_size=cfg.SOLVER.IMS_PER_BATCH,
      num_workers=cfg.DATALOADER.NUM_WORKERS,
      mapper=DatasetMapper(cfg,False), #do_train=True means we are in training mode.
      #aspect_ratio_grouping=False
    )

    with torch.no_grad():
      def get_all_inputs_outputs():
        for data in data_loader:
          yield data, model(data)

      evaluator = COCOEvaluator(ds[2], output_dir=cfg.OUTPUT_DIR, distributed=False, tasks=("bbox",))
      evaluator.reset()
      for inputs, outputs in get_all_inputs_outputs():
        evaluator.process(inputs, outputs)
      print("begin coco evaluation...")
      eval_results = evaluator.evaluate()
      print("finished coco evaluation!")

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

    parser.add_argument("--eval", action="store_true", default=False, help="perform evaluation")
    parser.add_argument("--coco", action="store_true", default=False, help="perform coco evaluation")
    parser.add_argument("--lr", action="store_true", default=False, help="learning rate search")
    parser.add_argument("--input-dir", default="output", help="the directory to read input task-related data such as trained models. By default, uses the output of the current executiion")
    parser.add_argument("--output-dir", default="output", help="the directory to output task-related data")

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