import albumentations as A

# These are the augmentations im thinking abt using. 
# Basically shift/scale/rotate, some sort of brightness/contrast manip,
# flip, cutting out a rectangle, and blurring the image. CLAHE and blur 
# might contradict each other a bit which in their purpose but at the 
# same time, one provides better brightness variation and one provides 
# blur to focus on the overall picture. So in a sense they complement 
# each other as well.
transform = A.Compose([
  # A.HorizontalFlip(p=0.5),
  # A.ShiftScaleRotate(p=0.5, rotate_limit=[-15, 15]), 
  # A.OneOf([
  #   A.CLAHE(p=0.65), # makes edges more clear by effective use of color (lightness?) spectrum
  #   A.ColorJitter(p=0.25),
  #   A.NoOp(p=0.1)
  # ], p=1),
  # A.Cutout(num_holes=1, max_h_size=60, max_w_size=60, fill_value=122, p=0.5),
  # A.Blur(p=0.5)
], p=1, bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category_id"]))
# the label_fields=["category_id"] creates a key that will be assigned during augmentation 
# and paired with the output image after augmentation, e.g labels. But can be any data that 
# does not need to change throughout augmentation! If bbox dissappears, so will the corresponding 
# entry of each added label field. They must match in size im pretty sure.
A.save(transform, './augs.json')