import torch
import cv2
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
import detectron2.data.datasets.pascal_voc as pascal_voc 
import copy
from .augmentor import DummyAlbuMapper

def save_sample(cfg, model, data_dict, dst_path, show=False):
  old_threshold = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
  with torch.no_grad():
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
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
  validation_split = ["000","001","012","018","025","030","034","038","053","062","068","075","083","127","131","139","156","181","245","249"]

  if split == "train":
    with open(json_path, 'r') as fr:
      original_dataset = deepcopy(json.load(fr))
      dataset = []
      for sample in original_dataset:
        if sample['image_id'].split('_')[0] not in validation_split:
          dataset.append(sample) 
          for annotation in sample['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
  elif split == "val":
    with open(json_path, 'r') as fr:
      original_dataset = deepcopy(json.load(fr))
      dataset = []
      for sample in original_dataset:
        if sample['image_id'].split('_')[0] in validation_split:
          dataset.append(sample) 
          for annotation in sample['annotations']:
            annotation['bbox_mode'] = BoxMode.XYXY_ABS
  elif split == "test":
    with open(json_path, 'r') as fr:
      original_dataset = deepcopy(json.load(fr))
      dataset = []
      for sample in original_dataset:
        dataset.append(sample) 
        for annotation in sample['annotations']:
          annotation['bbox_mode'] = BoxMode.XYXY_ABS
  else:
    raise NotImplementedError(f"Dataset split {split} is not supported for CSAW-S")

  return dataset

from util.datasets import CustomDataset
import scripts.generate_obj_csaws as csaws
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
      "train": load_csaw("../configs/obj/datasets/csaw-s-obj-trainval.json", "train"),
      "val": load_csaw("../configs/obj/datasets/csaw-s-obj-trainval.json", "val"),
      "test": load_csaw("../configs/obj/datasets/csaw-s-obj-test.json", "test")
    }
    return CustomDataset(labels, main_label, base_dataset, dataset_name, args.output_dir)
  else:
    raise NotImplementedError(f"Dataset {args.dataset} is not supported")