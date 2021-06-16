# general stuff 
import os 
import numpy as np
from tqdm import tqdm
import torch

# PROBLEM: Running multiple processes, one per GPU. As such, generation of the dataset may become completely 
# wonky especially if a preset seed is not used. 
# SOLUTION: Use shared seed? BAD
# SOLUTION: Use pre-generated splits, loaded from a json? GOOD
# SOLUTION: Let main process generate splits, dump to file, let other processes block/sync and then read that file? BEST

# SIMILAR PROBLEM: Is Albumentation really multi-GPU? ENSURE THIS.
# SOLUTION: No problem, augmentation is done after sampling which means it is done on individual processes, no comms needed.
    
# detectron2 stuff
from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.data import DatasetMapper
from detectron2.checkpoint import DetectionCheckpointer
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils import comm

import logging
from detectron2.utils.logger import setup_logger
from util.COCOEvaluator import COCOEvaluator

# actual training
from plain_train_net import do_train
from util.evaluate import evaluate, build_eval_loader
    
def lr_search(cfg, logger, lr_min_pow=-5, lr_max_pow=-2, resolution=20, n_iters=5000):
  powers = np.linspace(lr_min_pow, lr_max_pow, resolution)
  lrs = 10 ** powers
  best_val = float('-inf')
  best_lr = 0
  for lr in lrs:
    # do setup 
    cfg.SOLVER.BASE_LR = float(lr)
    cfg.SOLVER.MAX_ITER = n_iters
    model = build_model(cfg)
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    # train 5 epochs
    val_ap = do_train(cfg, model, resume=False, use_early_stopping=False, save_checkpoints=False) # trains on training data, evaluates on val data
    #losses.append() # TODO: Add logging of loss every n:th epoch during training.
    # calc val loss at the end
    if val_ap > best_val:
      best_val = val_ap
      best_lr = lr
    logger.info(f'Tested LR {lr}, with main label validation AP {val_ap}')
  logger.info(f'Grid search finished. Best learning rate is {best_lr}, with main label validation AP {best_val}')
  return best_lr
  
import random
import json
from detectron2.utils import comm

def setup_config(args, dataset, ds, training_size):
  cfg = get_cfg()
  cfg.OUTPUT_DIR = args.output_dir

  if dataset.dataset_name == "PascalVOC2007":
    cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"))
  elif dataset.dataset_name in ["CSAW-S", "MSCOCO"]:
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml")) # TODO: Potentially use longer schedule, 3x
  else:
    raise NotImplementedError(f"Dataset {dataset.dataset_name} not incorporated into experiments")
  
  cfg.INPUT.ALBUMENTATIONS = os.path.join("../configs/obj/augmentations", dataset.dataset_name + ".json")
  cfg.INPUT.FORMAT = "BGR"
  cfg.INPUT.DATASET_NAME = dataset.dataset_name
  
  # disable all of detectron's augmentations.
  cfg.INPUT.MIN_SIZE_TRAIN = (0,)
  cfg.INPUT.MIN_SIZE_TEST = (0,)
  cfg.INPUT.RANDOM_FLIP = "none"
  cfg.TEST.AUG.ENABLED = False

  # enables mixed precision, not super useful on my local GPU but might be free 
  # performance boost on remote!
  cfg.SOLVER.AMP.ENABLED = True

  cfg.DATASETS.TRAIN_SIZE = training_size if training_size != "full" else 263 # hard coded edge case for CSAW-S
  cfg.DATASETS.TRAIN = (ds[0],) # training name
  cfg.DATASETS.TEST = (ds[1], ds[2]) # validation, test names
  cfg.DATALOADER.NUM_WORKERS = 8

  # original_batchsize = cfg.SOLVER.IMS_PER_BATCH
  cfg.SOLVER.IMS_PER_BATCH = 16 # batch size is 2 images per gpu due to limitations  
  cfg.SOLVER.MAX_ITER = 100 * int( len(DatasetCatalog.get(ds[0])) / cfg.SOLVER.IMS_PER_BATCH ) # 100 epochs with early stopping
  cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * cfg.DATASETS.TRAIN_SIZE / len(DatasetCatalog.get(ds[0]))) if cfg.INPUT.DATASET_NAME == "CSAW-S" else cfg.SOLVER.MAX_ITER # since CSAW has pre-processed images that are part of augmentations, they don't count towards total (nah let's not do this)
  # TODO: Train for equally long with any amount of data or scale MAX_ITER by dataset fraction?
  # lr_cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(split_names[0])) / cfg.SOLVER.IMS_PER_BATCH)) # = 1 epoch
  cfg.TEST.EVAL_PERIOD = 0 # only check validation loss at the end of the lr search
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # NOTE: Cannot do this and evaluate in base experiment as well, must rebuild model with 0.05.
  
  #TODO: LR SCHEDULING (which scheduler, whether decay should be applied etc)

  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label
  cfg.MODEL.RETINANET.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label

  cfg.TEST.EVAL_PERIOD = 0
  return cfg

def base_experiment(args, dataset, training_size=200):
  logger = setup_logger(output=args.output_dir, distributed_rank=comm.get_rank(), name="experiments.base")
  main_label = args.main_label
  
  main_label_metrics = []
  for i in range(5): # repeat many times
    # no complementaries, should be identical across all processes since seed is set just before this
    if comm.is_main_process():
      ds, _ = dataset.subset(args.dataset + "_no_complementary_labels", nb_comp_labels=0, size=training_size, iteration=i+1)

    comm.synchronize()
    ds, _ = dataset.from_json() # multiple processes can access file no problem since it is read-only

    cfg = setup_config(args, dataset, ds, training_size)
    cfg.TEST.EVAL_PERIOD = 0
    max_iter = cfg.SOLVER.MAX_ITER

    #logger.info("Entering lr search... ")
    #lr = lr_search(cfg, logger, resolution=10, n_epochs=5)
    #logger.info(f"lr search finished, optimal lr is {lr}")

    cfg.SOLVER.BASE_LR = float(0.00046415888336127773) # could instead be assigned to cfg in lr_search but whatever
    cfg.SOLVER.MAX_ITER = max_iter
    cfg.TEST.EVAL_PERIOD = max_iter / 100 # 100 = nb_epochs, defined in setup_config
    logger.info(f'Configuration used: {cfg}')
    
    model = build_model(cfg)
    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
        
    do_train(cfg, model)

    model.eval()
    main_label_metrics.append(evaluate(cfg, model, logger))
    model.train()
  
  if comm.is_main_process():
    with open(os.path.join(args.output_dir, "metrics-base.json"), 'w') as fp:
      json.dump(main_label_metrics, fp)
    
def loo_experiment(args, dataset, training_size=200):
  logger = setup_logger(output=args.output_dir, distributed_rank=comm.get_rank(), name="experiments.loo")
  main_label = args.main_label
  
  for label in dataset.labels:
    # no complementaries, should be identical across all processes since seed is set just before this
    if comm.is_main_process():
      ds, _ = dataset.subset(args.dataset + "_leave_out_" + label, leave_out=label, size=training_size)

    comm.synchronize()
    ds, _ = dataset.from_json() # multiple processes can access file no problem since it is read-only

    cfg = setup_config(args, dataset, ds)
    cfg.TEST.EVAL_PERIOD = 0
    max_iter = cfg.SOLVER.MAX_ITER
    # print("Entering lr search... ")
    # lr = lr_search(cfg, logger, resolution=20, n_epochs=5)
    # print("lr search finished, optimal lr is", lr)
    cfg.SOLVER.BASE_LR = float(lr) # could instead be assigned to cfg in lr_search but whatever
    cfg.SOLVER.MAX_ITER = max_iter
    logger.info(f'Configuration used: {cfg}')

    main_label_metrics = []
    for i in range(5): # repeat many times
      cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get(ds[0])) # TODO: Fix this when re-adding early stopping

      model = build_model(cfg)
      distributed = comm.get_world_size() > 1
      if distributed:
          model = DistributedDataParallel(
              model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
          )

      do_train(cfg, model)

      model.eval()
      main_label_metrics.append(evaluate(cfg, model, logger))
      model.train()

    if comm.is_main_process():
      with open(os.path.join(args.output_dir, "metrics-loo.json"), 'w') as fp:
        json.dump(main_label_metrics, fp)
   
from detectron2.data import build_detection_train_loader
from util.augmentor import DummyAlbuMapper
from util.helpers import save_sample
def sample_experiment(args, dataset, nb_samples=3):
  logger = setup_logger(output=args.output_dir, distributed_rank=comm.get_rank(), name="experiments.sample")
  main_label = args.main_label

  # no complementaries, should be identical across all processes since seed is set just before this
  if comm.is_main_process():
    ds, _ = dataset.subset(args.dataset + "_sample", nb_comp_labels=0)
  
  comm.synchronize()
  ds, _ = dataset.from_json() # multiple processes can access file no problem since it is read-only
      
  cfg = setup_config(args, dataset, ds, training_size)
  cfg.TEST.EVAL_PERIOD = 0
  logger.info(f'Configuration used: {cfg}')
  cfg.TEST.EVAL_PERIOD = len(DatasetCatalog.get(ds[0])) # TODO: Fix this when re-adding early stopping

  model = build_model(cfg)
  distributed = comm.get_world_size() > 1
  if distributed:
      model = DistributedDataParallel(
          model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
      )
      
  data_loader = build_detection_train_loader(
      dataset=DatasetCatalog.get(cfg.DATASETS.TRAIN[0]), 
      mapper=DatasetMapper(cfg,False), #DummyAlbuMapper(cfg, is_train=True),
      total_batch_size=cfg.SOLVER.IMS_PER_BATCH,
      num_workers=cfg.DATALOADER.NUM_WORKERS
    )
  
  model.eval()
  for data, iteration in zip(data_loader, range(0, nb_samples)): 
    save_sample(cfg, model, data[0], f"../../samples/sample{iteration}.jpg")

# TODO: lr should be found separately, who cares which subset in particular is used;
# just do a grid search for all of the configurations (#labels, #data)
def lr_experiment(args, dataset, n_comp=0, training_size=200, n_iters=1500):
  logger = setup_logger(output=args.output_dir, distributed_rank=comm.get_rank(), name="experiments.lr")
  main_label = args.main_label
  
  # no complementaries, should be identical across all processes since seed is set just before this
  if comm.is_main_process():
    ds, _ = dataset.subset(args.dataset + "_lr", nb_comp_labels=n_comp, size=training_size, iteration=1) # iteration = 0 is only for CSAW-S

  comm.synchronize()
  ds, _ = dataset.from_json() # multiple processes can access file no problem since it is read-only

  cfg = setup_config(args, dataset, ds, training_size)
  cfg.TEST.EVAL_PERIOD = 0
  max_iter = cfg.SOLVER.MAX_ITER

  logger.info("Entering lr search... ")
  lr = lr_search(cfg, logger, resolution=10, n_iters=n_iters)
  logger.info(f"lr search finished, optimal lr is {lr}")
  
  if comm.is_main_process():
    lrs = {}
    with open("metrics-lr.json", 'r') as f:
      lrs = json.load(f)
      if dataset.dataset_name in lrs.keys():
        lrs[dataset.dataset_name][str(training_size)] = lr
      else:
        lrs[dataset.dataset_name] = {str(training_size): lr}
    with open("metrics-lr.json", 'w') as f:
      json.dump(lrs, f)