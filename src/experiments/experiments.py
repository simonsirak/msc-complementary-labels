# general stuff 
import os 
import numpy as np
from tqdm import tqdm
import torch

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
    
def lr_search(cfg, logger, lr_min_pow=-5, lr_max_pow=-2, resolution=20, n_epochs=5):
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
    val_loss = do_train(cfg, model, resume=False, use_early_stopping=False, save_checkpoints=False) # TODO: Use validation dataset, maybe by modding the config or adding option to do_train
    #losses.append() # TODO: Add logging of loss every n:th epoch during training.
    # calc val loss at the end
    if val_loss < best_val:
      best_val = val_loss
      best_lr = lr
    logger.info(f'Tested LR {lr}, with validation loss {val_loss}')
  logger.info(f'Grid search finished. Best learning rate is {best_lr}, with validation loss {val_loss}')
  return best_lr
  
import random
import json
from detectron2.utils import comm

def base_experiment(args, dataset):
  logger = setup_logger(output=args.output_dir, distributed_rank=comm.get_rank(), name="experiments.base")
  main_label = args.main_label

  # no complementaries, should be identical across all processes since seed is set just before this
  if comm.is_main_process():
    ds, _ = dataset.subset(args.dataset + "_no_complementary_labels", nb_comp_labels=0)
  
  comm.synchronize()
  ds, _ = dataset.from_json() # multiple processes can access file no problem since it is read-only
  
  # PROBLEM: Running multiple processes, one per GPU. As such, generation of the dataset may become completely 
  # wonky especially if a preset seed is not used. 
  # SOLUTION: Use shared seed? BAD
  # SOLUTION: Use pre-generated splits, loaded from a json? GOOD
  # SOLUTION: Let main process generate splits, dump to file, let other processes block/sync and then read that file? BEST
  
  # SIMILAR PROBLEM: Is Albumentation really multi-GPU? ENSURE THIS. 
      
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

  # enables mixed precision, not super useful on my local GPU but might be free 
  # performance boost on remote!
  cfg.SOLVER.AMP.ENABLED = True

  cfg.DATASETS.TRAIN = (ds[0],) # training name
  cfg.DATASETS.TEST = (ds[1], ds[2]) # validation, test names
  cfg.DATALOADER.NUM_WORKERS = 16

  original_batchsize = cfg.SOLVER.IMS_PER_BATCH
  cfg.SOLVER.IMS_PER_BATCH = 4 # batch size is 2 images per gpu due to limitations  
  cfg.SOLVER.MAX_ITER = int(cfg.SOLVER.MAX_ITER * cfg.SOLVER.IMS_PER_BATCH / original_batchsize) # now it is correctly set to the same number of epochs but with different batch size
  # TODO: Train for equally long with any amount of data or scale MAX_ITER by dataset fraction?
  # lr_cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(split_names[0])) / cfg.SOLVER.IMS_PER_BATCH)) # = 1 epoch
  cfg.TEST.EVAL_PERIOD = 0 # only check validation loss at the end of the lr search
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # NOTE: Cannot do this and evaluate in base experiment as well, must rebuild model with 0.05.
  
  #TODO: LR SCHEDULING (which scheduler, whether decay should be applied etc)

  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label
  cfg.MODEL.RETINANET.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label

  cfg.TEST.EVAL_PERIOD = 0
  # print("Entering lr search... ")
  lr = lr_search(cfg, logger, resolution=20, n_epochs=5)
  # print("lr search finished, optimal lr is", lr)
  cfg.SOLVER.BASE_LR = float(lr) # could instead be assigned to cfg in lr_search but whatever
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
  
  with open(os.path.join(args.output_dir, "metrics-base.json"), 'w') as fp:
    json.dump(main_label_metrics, fp)