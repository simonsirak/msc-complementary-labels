import os
import glob
import cv2 as cv
import numpy as np
from tqdm import tqdm
import copy 
from PIL import Image

# List of dataset classes
CLASSES = [
    "cancer",
    "calcifications",
    "axillary_lymph_nodes",
    "thick_vessels",
    "foreign_object",
    "skin",
    "nipple",
    "text",
    "non-mammary_tissue",    
    "pectoral_muscle",
    "mammary_gland",
]

OUTPUT_CLASSES = [
    "cancer",
    "calcifications",
    "axillary_lymph_nodes",
    "thick_vessels",
    "foreign_object",
    "nipple",
    "text",
    "non-mammary_tissue",    
]

# Apply label smoothing to all but small objects
APPLY_SMOOTHING = False

# List os small classes in the dataset
CLASSES_SMALL = [
    "calcifications",
    "nipple",
    "axillary_lymph_nodes"
]

# Dictionary from label_num to label
NUM_TO_LABEL = {number: label for number, label in enumerate(CLASSES)}
NUM_TO_LABEL[len(CLASSES)] = "background"
LABEL_TO_NUM = {v: k for k, v in NUM_TO_LABEL.items()}

OUTPUT_NUM_TO_LABEL = {number: label for number, label in enumerate(OUTPUT_CLASSES)}
OUTPUT_NUM_TO_LABEL[len(OUTPUT_CLASSES)] = "background"
OUTPUT_LABEL_TO_NUM = {v: k for k, v in OUTPUT_NUM_TO_LABEL.items()}
# I will use the original full size images and construct patches via dataloader instead.
# This is because I don't know if what it means to crop for object detection. I can then
# enable cropping in albumentations by adding such things, and by using bboxparams.

def main(args):
  base = os.path.join(args.path, "CSAW-S/CsawS")
  trainval(base)
  test(base)

# for test, only care abt cancer vs no cancer. The other metrics do not matter.
def test(base):
  dataset = []
  mask_paths_1 = glob.glob(os.path.join(base, "test_data", "segmentation_maps", "annotator_1", "*.png"))
  mask_paths_2 = glob.glob(os.path.join(base, "test_data", "segmentation_maps", "annotator_2", "*.png"))
  mask_paths_3 = glob.glob(os.path.join(base, "test_data", "segmentation_maps", "annotator_3", "*.png"))

  for i in tqdm(range(len(mask_paths_1))):
    sample = {} 
    
    # H, W, C
    mask_1 = cv.imread(mask_paths_1[i], cv.IMREAD_ANYDEPTH).astype('uint8')
    mask_2 = cv.imread(mask_paths_2[i], cv.IMREAD_ANYDEPTH).astype('uint8')
    mask_3 = cv.imread(mask_paths_3[i], cv.IMREAD_ANYDEPTH).astype('uint8')

    mask = (mask_1 == LABEL_TO_NUM['cancer']).astype('uint8') + (mask_2 == LABEL_TO_NUM['cancer']).astype('uint8') + (mask_3 == LABEL_TO_NUM['cancer']).astype('uint8')
    mask[mask <= 1] = 0
    mask[mask > 1] = 255
    mask = mask.astype('uint8')

    # corresponding image path
    img_name = '_'.join(mask_paths_1[i].split('/')[-1].split('.')[0].split('_')[:-1])
    img_path = os.path.join(base, "test_data", "original_images", img_name)
    
    sample['file_name'] = img_path + '.png'
    sample['height'] = mask_1.shape[0]
    sample['width'] = mask_1.shape[1]
    sample['image_id'] = img_name

    annotations = []

    blurred_mask = cv.blur(mask, (25,25), 0)
    contours, hierarchy = cv.findContours(blurred_mask.astype(np.uint8), mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    rects = []
    for idx, contour in enumerate(contours):
      if hierarchy[0][idx][3] == -1:
        rects.append(cv.boundingRect(contour))

    rects = cv.groupRectangles(rects, groupThreshold=0, eps=0.05)
    for rect in rects[0]:
      x, y, w, h = rect
      annotation = {}
      annotation['bbox'] = [int(x),int(y),int(x+w),int(y+h)] # box mode is XYXY_ABS
      annotation['category_id'] = OUTPUT_LABEL_TO_NUM['cancer']
      annotations.append(annotation)

    sample['annotations'] = annotations 
    dataset.append(sample)

  import json
  with open('../../configs/obj/datasets/csaw-s-obj-test.json', 'w') as fp:
    json.dump(dataset, fp)

def trainval(base):
  dataset = []
  mask_paths = os.path.join(base, "segmentation_maps", "*.png")
  mask_paths = glob.glob(mask_paths)

  for mask_path in tqdm(mask_paths):
    sample = {} 

    # H, W, C
    img = cv.imread(mask_path, cv.IMREAD_ANYDEPTH).astype('uint8')

    # corresponding image path
    img_name = '_'.join(mask_path.split('/')[-1].split('.')[0].split('_')[:-1])
    img_path = os.path.join(base, "original_images", img_name)
    
    sample['file_name'] = img_path + '.png'
    sample['height'] = img.shape[0]
    sample['width'] = img.shape[1]
    sample['image_id'] = img_name 

    annotations = []
    for label in LABEL_TO_NUM.keys():
      if label not in OUTPUT_CLASSES:
        continue

      mask_val = LABEL_TO_NUM[label]
      mask = (img == mask_val) * 255
      if label == "text":
        blurred_mask = cv.blur(mask, (51,51), 0)
      elif label in CLASSES_SMALL:
        blurred_mask = cv.blur(mask, (11,11), 0)
      elif label == "thick_vessel":
        blurred_mask = cv.blur(mask, (1,1), 0)
      else:
        blurred_mask = cv.blur(mask, (25,25), 0)
      contours, hierarchy = cv.findContours(blurred_mask.astype(np.uint8), mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

      rects = []
      for idx, contour in enumerate(contours):
        if hierarchy[0][idx][3] == -1:
          rects.append(cv.boundingRect(contour))

      rects = cv.groupRectangles(rects, groupThreshold=0, eps=0.05)
      for rect in rects[0]:
        x, y, w, h = rect
        annotation = {}
        annotation['bbox'] = [int(x),int(y),int(x+w),int(y+h)] # box mode is XYXY_ABS
        annotation['category_id'] = OUTPUT_LABEL_TO_NUM[label]
        annotations.append(annotation)

    sample['annotations'] = annotations 
    dataset.append(sample)

  import json
  with open('../../configs/obj/datasets/csaw-s-obj-trainval.json', 'w') as fp:
    json.dump(dataset, fp)

import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", default="../../datasets", help="path to directory containing 'CSAW-S' directory. Default: '../../datasets'")
  args = parser.parse_args()
  main(args)