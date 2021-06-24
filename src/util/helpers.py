import torch
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
import detectron2.data.datasets.pascal_voc as pascal_voc 
import copy
from .augmentor import DummyAlbuMapper
from detectron2.utils import comm

def save_sample(cfg, model, data_dict, dst_path, show=False):
  old_threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
  with torch.no_grad():
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    outputs = model([data_dict])

    img = data_dict["image"]
    orig = utils.read_image(data_dict['file_name'], format="BGR")
    # permute C, H, W format to H, W, C format and flip C from BGR to RGB
    v = Visualizer(img.permute(1,2,0).numpy()[:,:,::-1], # = img[:,:,::-1] in numpy
        metadata=MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
        scale=1)
    out = v.draw_instance_predictions(outputs[0]["instances"].to("cpu"))
    #out = v.draw_dataset_dict(data_dict)

    if show:
      cv2.imshow(dst_path, out.get_image()[:,:,::-1]) # flip final image to BGR again because cv2 wants that lol
      cv2.waitKey()
      cv2.destroyWindow(dst_path)
    else:
      cv2.imwrite(dst_path, out.get_image()[:,:,::-1]) # flip final image to BGR again because cv2 wants that lol
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = old_threshold

import os 
from detectron2.utils.events import CommonMetricPrinter, JSONWriter, TensorboardXWriter
from typing import Optional

# NOTE: Only call this from main process! This avoids weird makedirs stuff.
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
    os.makedirs(output_dir, exist_ok=True)
    return [
        # It may not always print what you want to see, since it prints "common" metrics only.
        CommonMetricPrinter(max_iter),
        JSONWriter(os.path.join(output_dir, "metrics.json")),
        TensorboardXWriter(output_dir),
    ]


import matplotlib.pyplot as plt
# Points list = list of tuples (label, list of losses), 1 list per execution
def plot_runs(points_list):
  plt.figure(figsize=(8, 3))
  for label, points in points_list:
    #plt.subplot(121)
    plt.plot([iter for iter in range(len(points))], points, label=label)
    plt.legend()
    #plt.xscale('log')
    plt.ylabel("Loss")
    plt.xlabel("iterations")
    plt.title("Loss vs iterations for different runs")
    #plt.savefig("face" + str(int(latent_size)) + ".png", bbox_inches="tight")
  plt.show()

# because apparently release 0.4 is different on github and what i have :)
from .detectron_build import *

def build_optimizer(cfg, model):
  """
  Build an optimizer from config.
  """
  params = get_default_optimizer_params(
      model,
      base_lr=cfg.SOLVER.BASE_LR,
      weight_decay_norm=cfg.SOLVER.WEIGHT_DECAY_NORM,
      bias_lr_factor=cfg.SOLVER.BIAS_LR_FACTOR,
      weight_decay_bias=cfg.SOLVER.WEIGHT_DECAY_BIAS,
      weight_decay=cfg.SOLVER.WEIGHT_DECAY
  )

  # NOTE: If support for gradient clipping is desired, you need to wrap this 
  # in "maybe_add_gradient_clipping" function from same place as get_default_optimizer_params
  return maybe_add_gradient_clipping(cfg, torch.optim.Adam)(
      params,
      lr=cfg.SOLVER.BASE_LR,
      weight_decay=cfg.SOLVER.WEIGHT_DECAY,
  )

import json 
from detectron2.structures import BoxMode
from copy import deepcopy
def load_csaw(json_path, split="train"):
  # validation_split = ["000","001","012","018","025","030","034","038","053","062","068","075","083","127","131","139","156","181","245","249"]
  dataset = []
  with open(json_path, 'r') as fr:
    original_dataset = deepcopy(json.load(fr))
    dataset = []
    for sample in original_dataset:
      dataset.append(sample) 
      sample['file_name'] = '/'.join(sample['file_name'].split('/')[1:])
      for annotation in sample['annotations']:
        annotation['bbox_mode'] = BoxMode.XYXY_ABS
  
  if split == "train":
    splits_path = os.path.join('/'.join(json_path.split('/')[:-1]), "training_random_splits.json")
    with open(splits_path, 'r') as fr:
      random_splits = deepcopy(json.load(fr))
      for random_split in random_splits.keys():
        iterations = random_splits[random_split]
        for iteration in iterations.keys():
          imgs = iterations[iteration]
          iterations[iteration] = {"dataset": []}

          # extract relevant patches
          for sample in dataset:
            if sample["file_name"].split('/')[-1].split("-")[0] in imgs: # if patch belongs to image in split
              iterations[iteration]["dataset"].append(sample)
      random_splits["full"] = {"dataset": dataset}
      dataset = random_splits # this includes all splits
      
      # if comm.is_main_process():
      #  with open(os.path.join(".", "dataset-splits.json"), 'w') as fp:
      #    json.dump(dataset, fp)
      #exit(0)
              
  return dataset

from util.datasets import CustomDataset
import scripts.generate_obj_csaws as csaws
from detectron2.data.datasets.coco import register_coco_instances
from numpy.random import default_rng
from detectron2.utils import comm
from detectron2.data import MetadataCatalog, DatasetCatalog

def extract_dataset(dataset_name, main_label, args): # TODO: Add base path arg
  if dataset_name == "PascalVOC2007":
    labels = list(pascal_voc.CLASS_NAMES)
    base_dataset = {
      "train": pascal_voc.load_voc_instances(os.path.join(args.dataset_path, "VOC2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007"), "train", labels),
      "val": pascal_voc.load_voc_instances(os.path.join(args.dataset_path, "VOC2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007"), "val", labels),
      "test": pascal_voc.load_voc_instances(os.path.join(args.dataset_path, "VOC2007/voctest_06-nov-2007/VOCdevkit/VOC2007"), "test", labels),
    }
    return CustomDataset(labels, main_label, base_dataset, dataset_name, args.output_dir)
  elif dataset_name == "CSAW-S":
    labels = csaws.OUTPUT_CLASSES
    base_dataset = {
      "train": load_csaw("../configs/obj/datasets/csaw-s-obj-train.json", "train"),
      "val": load_csaw("../configs/obj/datasets/csaw-s-obj-val.json", "val"),
      "test": load_csaw("../configs/obj/datasets/csaw-s-obj-test.json", "test")
    }
    
    rng = default_rng(seed=comm.shared_random_seed()) # based on input seed
    shuffled_training = base_dataset["train"]
    #print(type(shuffled_training))
    for subset_size, subset in shuffled_training.items():
      if subset_size != "full":
        for iteration, repetition in subset.items():
          #print(type(repetition))
          rng.shuffle(repetition['dataset']) # in-place shuffling
      else:
        rng.shuffle(subset['dataset']) # in-place shuffling
    
    return CustomDataset(labels, main_label, base_dataset, dataset_name, args.output_dir)
  elif dataset_name == "MSCOCO":
    register_coco_instances("cocotrain", {}, os.path.join(args.dataset_path, "MSCOCO/coco/annotations/instances_train2017.json"), os.path.join(args.dataset_path, "MSCOCO/coco/train2017"))
    # register_coco_instances("cocotrain", {}, os.path.join(args.dataset_path, "MSCOCO/coco/annotations/instances_val2017.json"), os.path.join(args.dataset_path, "MSCOCO/coco/val2017"))
    register_coco_instances("cocoval", {}, os.path.join(args.dataset_path, "MSCOCO/coco/annotations/instances_val2017.json"), os.path.join(args.dataset_path, "MSCOCO/coco/val2017"))
    rng = default_rng(seed=comm.shared_random_seed())
    shuffled_training = DatasetCatalog.get("cocotrain")
    rng.shuffle(shuffled_training) # in-place shuffling
    base_dataset = {
      "train": shuffled_training[:int(0.95*len(shuffled_training))],
      "val": shuffled_training[int(0.95*len(shuffled_training)):], # don't need INFINITY for val, otherwise it'll take forever.
      "test": DatasetCatalog.get("cocoval"),
    }
    labels = MetadataCatalog.get("cocotrain").thing_classes
    return CustomDataset(labels, main_label, base_dataset, dataset_name, args.output_dir)
  else:
    raise NotImplementedError(f"Dataset {args.dataset} is not supported")