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

import logging
from detectron2.utils.logger import setup_logger
from util.COCOEvaluator import COCOEvaluator

# actual training
from plain_train_net import do_train, build_eval_loader

def evaluate(cfg, model, logger):
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
    logger.info(f'Starting COCO evaluation preprocessing ... ')
    for inputs, outputs in tqdm(get_all_inputs_outputs()):
      evaluator.process(inputs, outputs)
    print("begin coco evaluation...")
    eval_results = evaluator.evaluate()
    print("finished coco evaluation!")
  return eval_results
    
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

  # no complementaries
  ds, _ = dataset.subset(args.dataset + "_no_complementary_labels", nb_comp_labels=0, manual_comp_labels=['bicycle','car','motorbike'])

  cfg = get_cfg()
  cfg.OUTPUT_DIR = args.output_dir

  if dataset.dataset_name == "PascalVOC2007":
    cfg.merge_from_file(model_zoo.get_config_file("PascalVOC-Detection/faster_rcnn_R_50_FPN.yaml"))
  elif dataset.dataset_name == "CSAW-S":
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_1x.yaml")) # TODO: Potentially use longer schedule, 3x
  else:
    raise NotImplementedError(f"Dataset {dataset.dataset_name} not incorporated into experiments")
  
  cfg.INPUT.ALBUMENTATIONS = os.path.join("../configs/obj/augmentations", dataset.dataset_name + ".json")
  cfg.INPUT.FORMAT = "BGR"

  # enables mixed precision, not super useful on my local GPU but might be free 
  # performance boost on remote!
  cfg.SOLVER.AMP.ENABLED = True

  cfg.DATASETS.TRAIN = (ds[0],) # training name
  cfg.DATASETS.TEST = (ds[1], ds[2]) # validation, test names
  cfg.DATALOADER.NUM_WORKERS = 8
  
  cfg.SOLVER.IMS_PER_BATCH = 4 # batch size is 2 images per gpu due to limitations  
  # lr_cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(split_names[0])) / cfg.SOLVER.IMS_PER_BATCH)) # = 1 epoch
  cfg.TEST.EVAL_PERIOD = 0 # only check validation loss at the end of the lr search
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # NOTE: Cannot do this and evaluate in base experiment as well, must rebuild model with 0.05.
  
  #TODO: LR SCHEDULING (which scheduler, whether decay should be applied etc)

  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label
  cfg.MODEL.RETINANET.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label

  cfg.TEST.EVAL_PERIOD = 0
  # print("Entering lr search... ")
  #lr = lr_search(cfg, logger, resolution=20, n_epochs=5)
  # print("lr search finished, optimal lr is", lr)
  cfg.SOLVER.BASE_LR = 1e-4 # float(lr) # could instead be assigned to cfg in lr_search but whatevs
  logger.info(f'Configuration used: {cfg}')
  
  main_label_metrics = []
  for i in range(1): # repeat many times
    #cfg.TEST.EVAL_PERIOD = 5000
    
    cfg.MODEL.WEIGHTS = os.path.join("../../checkpoints/empty_imgs_3_complabels_longtraining_nounderfitting/output", 'best_model.pth')
    model = build_model(cfg)
    DetectionCheckpointer(model).load(cfg.MODEL.WEIGHTS)

    #do_train(cfg, model)

    model.eval()
    #model.train()
    main_label_metrics.append(evaluate(cfg, model, logger))
  with open(os.path.join(args.output_dir, "metrics-base.json"), 'w') as fp:
    json.dump(main_label_metrics, fp)