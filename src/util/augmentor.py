import copy
import json

import numpy as np
import torch
import albumentations as A

import detectron2.data.transforms as T
from detectron2.data import detection_utils as utils
from detectron2.structures import BoxMode

# BEWARE: DO NOT WORK ON THIS FILE WHILE RUNNING EXPERIMENTS, IT IS APPARENTLY READ 
# EVEN DURING EXECUTION.
class DummyAlbuMapper:
    """
    To use albumentations:
    1. Create serialized json file with augmentation config. See "sample-detection-albu-config.json"
    2. Define cfg.INPUT.ALBUMENTATIONS variable in detectron config file.
    Also I'm using detectron tools to resize and crop images 
    because they allow to use:
    ResizeShortestEdge
    RandomCrop relative to the original image size
    
    To change CROP/RESIZE parameters modify config variables:
    cfg.INPUT.CROP.[...]
    cfg.INPUT.[MAX|MIN]_SIZE_[TEST|TRAIN]
    """
    
    def __init__(self, cfg, is_train=True):
      self.aug = self._get_aug(cfg.INPUT.ALBUMENTATIONS)
      self.img_format = cfg.INPUT.FORMAT
      self.dataset_name = cfg.INPUT.DATASET_NAME
      self.is_train = is_train
      #TODO: BoxMode based on added custom cfg key since it varies based on dataset.
            
    def _get_aug(self, arg):
        with open(arg) as f:
            return A.from_dict(json.load(f))

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)

        # Reads image in format H, W, C. C is in color format self.img_format 
        # (usually BGR because default models in detectron2 use that)
        #print(self.img_format)
        img = utils.read_image(dataset_dict['file_name'], format=self.img_format)
        # print("SHAPE OF IMG", img.shape)

        boxes = [ann['bbox'] for ann in dataset_dict['annotations']]
        labels = [ann['category_id'] for ann in dataset_dict['annotations']]

        if self.is_train:
          # albumentations wants RGB format so we reverse ASSUMING BGR IS THE CURRENT FORMAT
          augm_annotation = self.aug(image=img[:,:,::-1], bboxes=boxes, category_id=labels)
          img = augm_annotation['image'][:,:,::-1]
        else: 
          augm_annotation = {'category_id': labels, 'bboxes': boxes, 'image': img[:,:,::-1]}
          # img = augm_annotation['image'][:,:,::-1]
        h, w, _ = img.shape

        # print(self.aug.processors["bboxes"].params.format)
        # exit(0)

        #print("IMAGES AFTER ALBUMENTATIONS", img.transpose(2, 0, 1))
        augm_boxes = np.array(augm_annotation['bboxes'], dtype=np.float32)
        #print(augm_boxes)
        # exit(0)
        # sometimes bbox annotations go beyond image 
        #augm_boxes[:, :] = augm_boxes[:, :].clip(min=[0, 0, 0, 0], max=[w, h, w, h])
        augm_labels = np.array(augm_annotation['category_id'])
        dataset_dict['annotations'] = [
            {
                'iscrowd': 0,
                'bbox': augm_boxes[i].tolist(),
                'category_id': augm_labels[i],
                'bbox_mode': BoxMode.XYXY_ABS if self.dataset_name in ["CSAW-S", "PascalVOC2007"] else BoxMode.XYWH_ABS
            }
            for i in range(len(augm_boxes))
        ]

        annos = [
            obj
            for obj in dataset_dict.pop("annotations")
            #if obj.get("iscrowd", 0) == 0
        ]

        dataset_dict['annotations'] = annos
        instances = utils.annotations_to_instances(
            annos, img.shape[:2]
        )
        
        # no need to filter empty instances
        dataset_dict["instances"] = instances # utils.filter_empty_instances(instances)

        dataset_dict['height'] = img.shape[0]
        dataset_dict['width'] = img.shape[1]

        # converts to tensor of format C, H, W,
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))

        #print("IMAGES AFTER DETECTRON2 STUFF", img.transpose(2, 0, 1))

        return dataset_dict