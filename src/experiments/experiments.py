from detectron2.modeling import build_model
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.data import MetadataCatalog

from plain_train_net import do_train, evaluate

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
    val_loss = do_train(cfg, model, resume=False, use_early_stopping=False) # TODO: Use validation dataset, maybe by modding the config or adding option to do_train
    #losses.append() # TODO: Add logging of loss every n:th epoch during training.
    # calc val loss at the end
    if val_loss < best_val:
      best_val = val_loss
      best_lr = lr
    print("Tested LR", lr, "with validation loss", val_loss)
    
def base_experiment(args, dataset):
  main_label = args.main_label

  # no complementaries
  ds, _ = dataset.subset(args.dataset + "_no_complementary_labels", nb_comp_labels=3)

  cfg = get_cfg()

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
  
  cfg.SOLVER.IMS_PER_BATCH = 2 # batch size is 2 images due to limitations  
  # lr_cfg.TEST.EVAL_PERIOD = int(round(len(DatasetCatalog.get(split_names[0])) / cfg.SOLVER.IMS_PER_BATCH)) # = 1 epoch
  cfg.TEST.EVAL_PERIOD = 0 # only check validation loss at the end of the lr search
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05 # NOTE: Cannot do this and evaluate in base experiment as well, must rebuild model with 0.05.
  
  #TODO: LR SCHEDULING (which scheduler, whether decay should be applied etc)

  cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label
  cfg.MODEL.RETINANET.NUM_CLASSES = len(MetadataCatalog.get(ds[0]).get("thing_classes"))  # number of complementary labels + main label

  cfg.TEST.EVAL_PERIOD = 0
  # print("Entering lr search... ")
  lr = lr_search(cfg, resolution=20, n_epochs=5)
  # print("lr search finished, optimal lr is", lr)
  cfg.SOLVER.BASE_LR = float(1e-4) # could instead be assigned to cfg in lr_search but whatevs
  for i in range(1): # repeat many times
    cfg.TEST.EVAL_PERIOD = 5000
    
    model = build_model(cfg)

    do_train(cfg, model)

    model.eval()
    evaluate(cfg, model)
    model.train()
  pass