import torch, torchvision
#!pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
import detectron2

# import some common libraries
import numpy as np
import os, json, cv2, random
#from google.colab.patches import cv2_imshow

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.utils import comm

import random
import json 
from copy import copy, deepcopy

import logging
from detectron2.utils.logger import setup_logger
from detectron2.utils import comm

class CustomDataset:
  def __init__(self, labels, main_label, base_dicts, dataset_name, log_dir):
    self.id = 0
    self.labels = labels # all labels (python list) in the order of the category id's of the detectron dataset dictionary
    self.main_label = main_label # main label
    self.base_dicts = base_dicts # func that returns a dict of standard format dict {"train": {}, "val": {}, "test": {}}
    setup_logger(output=log_dir, distributed_rank=comm.get_rank(), name="datasets")
    self.logger = logging.getLogger("datasets")
    self.dataset_name = dataset_name
    self.log_dir = log_dir
  
  # currently only training dataset
  # TODO: Think about how to incorporate varying dataset size, 
  # since dataset might have different labels in same image so
  # it is unclear how to preserve distribution over different sizes.
  def subset(self, base_name, nb_comp_labels=0, seed=None, leave_out=None, size=None, manual_comp_labels=None, identity=False, iteration=None):
    # FIRST: Extract only the images that contain annotations for the main label.
    # This removes any variation in the amount of included images despite the chosen
    # dataset. Main label is in 100% of images this way.
    base_dicts = deepcopy(self.base_dicts)
    
    if not identity:
      for split in base_dicts.keys():
        if self.dataset_name == "CSAW-S" and split == "train":
          if str(size) in base_dicts[split].keys():
            if str(iteration) in base_dicts[split][str(size)].keys():
              new_split = base_dicts[split][str(size)][str(iteration)]["dataset"]
            else:
              self.logger.warning(f"iteration {iteration} is not among the supported iterations {list(base_dicts[split][str(size)].keys())}! Using full split.")
              new_split = base_dicts[split]["full"]["dataset"]
          else:
            self.logger.warning(f"size {size} is not among the supported sizes {list(base_dicts[split].keys())}! Using full split.")
            new_split = base_dicts[split]["full"]["dataset"]
        else:
          new_split = base_dicts[split]
          # new_split = []
          # old_split = base_dicts[split]
          # for record in old_split:
          #   if self.labels.index(self.main_label) in [annotation["category_id"] for annotation in record["annotations"]]:
          #     new_split.append(record)

          # SECOND: Include percentage amount of images
          # Use only for train dataset; have full validation set for early stopping so they all are based on same validation data
          # TODO: Is absolute values better choice?
          # TODO: Fix commented accumulation if it will be used in future.
          if split == "train" and size is not None: # if custom size is specified
            self.logger.info(f"full split size for {split} = {len(new_split)}")
            if size > len(new_split):
              self.logger.warning(f"size {size} is greater than training split size {len(new_split)}! Keeping the split size.")
              size = len(new_split)
            new_split = random.sample(new_split, size)
            self.logger.info(f"full split size for {split} = {len(new_split)}")
        base_dicts[split] = new_split

      # COUNT OCCURRENCES & FILTER OUT LABELS THAT ARE COMMON ENOUGH IN RELATION TO MAIN LABEL
      label_counter = [0 for x in self.labels] # use to only include labels that occur at least 10% as much as the main label in train data that contains main label
      # accumulate the occurrences of each label class in the filtered training data
      for record in base_dicts["train"]:
        for annotation in record["annotations"]:
          label_counter[annotation['category_id']] += 1
      #print("label counter: ", label_counter)
      valid_labels = []
      nb_train_occ = label_counter[self.labels.index(self.main_label)]
      self.logger.info("occurrences of main label in train data: {}".format(nb_train_occ))
      for i, c in enumerate(label_counter):
        #print("c:", c, "i:", i)
        if c / nb_train_occ >= 0: # used to be 0.02, but now accepts all labels in case MSCOCO has a label that is veery rare
          valid_labels.append(self.labels[i])
          #print("valid labels: ", valid_labels)

      self.logger.info("all valid labels (incl. main): {}".format(valid_labels))
      if leave_out is not None:
        # do leave one out changes instead and ignore complabels
        assert leave_out in valid_labels, f"{leave_out} is not a valid label in this dataset!"
        assert leave_out != self.main_label, f"{leave_out} is the main label -- it cannot be left out!"

        c2l = deepcopy(valid_labels)
        c2l.remove(leave_out)
        c2l.sort()

        self.logger.info(f"leaving out complementary label: {leave_out}")
      else:
        if manual_comp_labels is not None:
          c2l = copy(manual_comp_labels)
          c2l.append(self.main_label)
          c2l.sort()
        else:
          # DATASET
          comp_labels = copy(valid_labels)
          comp_labels.remove(self.main_label)
          # random.seed(seed) # TODO: Don't seed here, seed in the beginning of an experiment only. Otherwise it is not pseudorandom.
          self.logger.info(f"all complementary labels: {comp_labels}")
          c2l = random.sample(comp_labels, nb_comp_labels)
          c2l.append(self.main_label) # should always be included
          c2l.sort()
      
      self.logger.info("chosen valid labels (incl. main label): {}".format(c2l))
      l2c = {}
      for c, l in enumerate(c2l):
        l2c[l] = c
      # create mapping "old category id -> label name" (should be literally the list of labels)
      # randomize a chosen subset of labels.
      # create mapping "label name -> new category id" (a dict where new category id is based on random subset)
      # convert as follows: If label in chosen subset, then give it new category id. Else, remove the annotation.
    else:
      l2c = {}
      c2l = self.labels
      for c, l in enumerate(c2l):
        l2c[l] = c
    names = []
    #base_dicts = deepcopy(self.base_dicts) # so we don't screw with original
    for split in base_dicts.keys(): # if no test split exists this still works
      for record in base_dicts[split]:
        new_annotations = []
        for annotation in record["annotations"]:
          if self.labels[annotation['category_id']] in l2c.keys():
            good = copy(annotation)
            good['category_id'] = l2c[self.labels[good['category_id']]]
            good.pop('segmentation', None) # mainly to remove this from COCO because they were encoded as bytes, which cannot be JSON serialized, I don't use these anyway
            new_annotations.append(good)
        record["annotations"] = new_annotations # those with no annotation are filtered out by dataloader bc cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True by default

      # TODO: remove balloon from here.
      name = base_name + "_" + split + "_" + str(self.id)
      names.append(name)
      
      if name in DatasetCatalog.list():
        DatasetCatalog.remove(name) # remove if exists

      DatasetCatalog.register(name, lambda d=split: base_dicts[d])    
      self.logger.info(f"Generated dataset named '{name}'")

      # METADATA
      if name in MetadataCatalog.list():
        MetadataCatalog.remove(name) # remove if exists
      MetadataCatalog.get(name).set(thing_classes=c2l)
      MetadataCatalog.get(name).set(main_label=self.main_label) # used in custom COCOEvaluator
      if self.dataset_name == "PascalVOC2007":
        if split in ["train", "val"]:
          MetadataCatalog.get(name).set(dirname="VOC2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007")
        elif split == "test":
          MetadataCatalog.get(name).set(dirname="VOC2007/voctest_06-nov-2007/VOCdevkit/VOC2007")
        MetadataCatalog.get(name).set(split=split)
        MetadataCatalog.get(name).set(year=2007)
      elif self.dataset_name == "CSAW-S":
        MetadataCatalog.get(name).set(split=split)
        MetadataCatalog.get(name).set(year=2020)
      elif self.dataset_name == "MSCOCO":
        MetadataCatalog.get(name).set(split=split)
        MetadataCatalog.get(name).set(year=2017)

      self.logger.info(f"SPLIT SIZE: {len(base_dicts[split])}")

    self.id = self.id + 1

    if len(names) == 3:
      names.sort()
      ordered_names = []
      ordered_names.append(names[1])
      ordered_names.append(names[2])
      ordered_names.append(names[0])
    else:
      ordered_names = names
    
    # TODO: output this in the output dir instead, so multiple experiments 
    # can run in parallel without overwriting this.
    with open(os.path.join(self.log_dir, 'generated-dataset.json'), 'w') as fp:
      # self.logger.info(base_dicts)
      json.dump({"dict": base_dicts, "names": ordered_names, "thing_classes": c2l}, fp)
    
    return (ordered_names, c2l)

  def top_k_complementary_labels(self, main_label, k):
    labels = deepcopy(self.labels)
    self.logger.info(len(labels))
    # labels.remove(main_label)
    full_train_data = self.base_dicts["train"]
    occurrences = [0 for label in labels]
    self.logger.info(len(occurrences))
    for record in full_train_data:
      for annotation in record["annotations"]:
        # self.logger.info(annotation['category_id'])
        occurrences[annotation['category_id']] += 1
    sorted_indices = sorted(range(len(occurrences)), key = lambda k : occurrences[k], reverse = True)
    sorted_indices.remove(labels.index(main_label)) # remove the value representing the main label from this list; the order of indices remains.
    assert k <= len(sorted_indices), f"k = {k} is larger than the maximum number of labels = {len(labels)} for the dataset {self.dataset_name}!"
    return [labels[i] for i in sorted_indices[:k]]

  # prints the percentage of images that each label is in
  def print_percentage_of_occurrence_of_label_in_images(self):
    labels = deepcopy(self.labels)
    full_train_data = self.base_dicts["train"]
    occurrences = [0 for label in labels]
    for record in full_train_data:
      for label in labels:
        for annotation in record["annotations"]:
          if annotation['category_id'] == labels.index(label):
            occurrences[annotation['category_id']] += 1
            break
    for label in labels:
      self.logger.info(f"label {label} occurs in {100 * float(occurrences[labels.index(label)]) / len(full_train_data)}% of the images.")

  def from_json(self):
    with open(os.path.join(self.log_dir, 'generated-dataset.json'), 'r') as fr:
      mapping = {"train": 0, "val": 1, "test": 2}
      dataset = json.load(fr)
      base_dicts = dataset["dict"]
      ordered_names = dataset["names"]
      c2l = dataset["thing_classes"]
      for split in base_dicts.keys(): # if no test split exists this still works
        name = ordered_names[mapping[split]]

        if name in DatasetCatalog.list():
          DatasetCatalog.remove(name) # remove if exists

        DatasetCatalog.register(name, lambda d=split: base_dicts[d])    
        self.logger.info(f"Generated dataset named '{name}'")

        # METADATA
        if name in MetadataCatalog.list():
          MetadataCatalog.remove(name) # remove if exists
        MetadataCatalog.get(name).set(thing_classes=c2l)
        MetadataCatalog.get(name).set(main_label=self.main_label) # used in custom COCOEvaluator
        if self.dataset_name == "PascalVOC2007":
          if split in ["train", "val"]:
            MetadataCatalog.get(name).set(dirname="VOC2007/voctrainval_06-nov-2007/VOCdevkit/VOC2007")
          elif split == "test":
            MetadataCatalog.get(name).set(dirname="VOC2007/voctest_06-nov-2007/VOCdevkit/VOC2007")
          MetadataCatalog.get(name).set(split=split)
          MetadataCatalog.get(name).set(year=2007)
        elif self.dataset_name == "CSAW-S":
          MetadataCatalog.get(name).set(split=split)
          MetadataCatalog.get(name).set(year=2020)
        elif self.dataset_name == "MSCOCO":
          MetadataCatalog.get(name).set(split=split)
          MetadataCatalog.get(name).set(year=2017)

        self.logger.info(f"SPLIT SIZE: {len(base_dicts[split])}")
      return (ordered_names, c2l)