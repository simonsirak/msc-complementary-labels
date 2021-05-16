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

import random
import json 
from copy import copy, deepcopy

import logging
from detectron2.utils.logger import setup_logger

logger = setup_logger(output="./output", name="datasets")

class CustomDataset:
  def __init__(self, labels, main_label, base_dict_func, dataset_name):
    self.id = 0
    self.labels = labels # all labels (python list) in the order of the category id's of the detectron dataset dictionary
    self.main_label = main_label # main label
    self.base_dict_func = base_dict_func # func that returns a dict of standard format dict {"train": {}, "val": {}, "test": {}}
    self.logger = logging.getLogger("datasets")
    self.dataset_name = dataset_name
  
  # currently only training dataset
  # TODO: Think about how to incorporate varying dataset size, 
  # since dataset might have different labels in same image so
  # it is unclear how to preserve distribution over different sizes.
  def subset(self, base_name, nb_comp_labels=0, seed=None, leave_out=None, percentage=1, manual_comp_labels=None, identity=False):
    # FIRST: Extract only the images that contain annotations for the main label.
    # This removes any variation in the amount of included images despite the chosen
    # dataset. Main label is in 100% of images this way.
    base_dicts = deepcopy(self.base_dict_func)

    if not identity:
      for split in base_dicts.keys():
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
        if split == "train":
          print("full split size for", split, "=", len(new_split))
          new_split = random.sample(new_split, round(percentage*len(new_split)))
          print("reduced split size for", split, "=", len(new_split))
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
        if c / nb_train_occ >= 0.02:
          valid_labels.append(self.labels[i])
          #print("valid labels: ", valid_labels)

      self.logger.info("all valid labels (incl. main): {}".format(valid_labels))
      if leave_out is not None:
        # do leave one out changes instead and ignore complabels
        assert leave_out in valid_labels, f"{leave_out} is not a valid label in this dataset!"
        assert leave_out != self.main_label, f"{leave_out} is the main label -- it cannot be left out!"

        c2l = deepcopy(valid_labels)
        c2l.remove(leave_out)

        print("leaving out complementary label:", leave_out)
      else:
        if manual_comp_labels is not None:
          c2l = copy(manual_comp_labels)
          c2l.append(self.main_label)
        else:
          # DATASET
          comp_labels = copy(valid_labels)
          comp_labels.remove(self.main_label)
          random.seed(seed) # TODO: Don't seed here, seed in the beginning of an experiment only. Otherwise it is not pseudorandom.
          print("all complementary labels: ", comp_labels)
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
    #base_dicts = deepcopy(self.base_dict_func) # so we don't screw with original
    for split in base_dicts.keys(): # if no test split exists this still works
      for record in base_dicts[split]:
        new_annotations = []
        for annotation in record["annotations"]:
          if self.labels[annotation['category_id']] in l2c.keys():
            good = copy(annotation)
            good['category_id'] = l2c[self.labels[good['category_id']]]
            new_annotations.append(good)
        record["annotations"] = new_annotations # those with no annotation are filtered out by dataloader bc cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS = True by default

      # TODO: remove balloon from here.
      name = base_name + "_" + split + "_" + str(self.id)
      names.append(name)
      
      if name in DatasetCatalog.list():
        DatasetCatalog.remove(name) # remove if exists

      DatasetCatalog.register(name, lambda d=split: base_dicts[d])    
      print("Generated dataset named '", name, "'")

      # METADATA
      if name in MetadataCatalog.list():
        MetadataCatalog.remove(name) # remove if exists
      MetadataCatalog.get(name).set(thing_classes=c2l)
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

      print("SPLIT SIZE:", len(base_dicts[split]))

    self.id = self.id + 1

    if len(names) == 3:
      names.sort()
      ordered_names = []
      ordered_names.append(names[1])
      ordered_names.append(names[2])
      ordered_names.append(names[0])
    else:
      ordered_names = names
    print(ordered_names)
    
    return (ordered_names, c2l)