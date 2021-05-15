* Add Augmentation.
* Choose a reasonable lr scheduler. 
* Revisit the default configs used from [here](https://github.com/facebookresearch/detectron2/blob/master/configs/Base-RCNN-FPN.yaml) and [here](https://detectron2.readthedocs.io/en/latest/modules/config.html#config-references). Do they make sense in the low data regime?  
  * In relation to this, [here is a link discussing models in low-data regime for obj. detection](https://tenfifty.io/object-detection-in-the-low-data-regime/). [Another one specifically about generative modeling for object detection in low-data regime, accepted by IEEE](https://arxiv.org/abs/1910.07169).
  * [One that discusses model complexity vs low data, probably not very relevant](https://www.cs.cmu.edu/~deva/papers/moredata.pdf)
  * [These found that model complexity is critical in performance in low data regime, so probably want to construct smaller model? Very unknown paper but also very new](https://arxiv.org/pdf/2003.12843.pdf)
* Fix bug in lr search training, "FloatingPointError: Predicted boxes or scores contain Inf/NaN. Training has diverged.". It writes that then terminates for a particular lr. Suggests that the lr has diverged, but it kinda sucks if the whole program just terminates after that.
  * They straight up say in detectron2 issues on github that it is because of shitty LR. So just pick a smaller range since the larger range is ass.
* I'm thinking about not doing test-time augmentation. Or validation-time augmentation either for that matter. It does give an improvement during inference-time but most likely takes longer during evaluation due to the extra augmentations. Plus, it would probably give varying amounts of improvement depending on how many complementary labels exist, and how much data exists (diminishing returns) as suggested in a paper i found, which might make the experimental results difficult to accurately discuss in isolation. Then again, if TTA/augmentation during validation is considered standard, I should probably include it to see if comp labels improves performance on top of TTA. 

Things to reconsider (try looking up papers that discusses these things in low data regime):
* Should model be smaller/simpler?
* Should model have smaller number of RoIs to use for training? Probably not necessary to think abt this.
* Should less anchors be used?
* Should augmentations be ordered in some way, e.g random crop before resize? 
* Also which augmentations are worth in natural and which are worth in medical?
  * I think resize is not necessarily good in natural datasets like Pascal VOC and COCO if you don't use TTA or any inference-time preprocessing that also resizes. Also these datasets have different aspect ratios which can morph natural images a bit too much.
  * [This paper](https://arxiv.org/abs/1906.11172) is like AutoAugment but for object detection, and suggests translations, shears (of bbox and/or whole image), rotations, color changes but not resizing or cropping. So I could implement their policy that they found on COCO since it seems to work on Pascal VOC too according to their paper.
---------------------------

Current TODO: 
* Plot augmentation just to see that it works, do it within the mapper, no need to use the full model just some default pretrained model works.
  * ended up having bugs in the augmentations so good thing i did this!
* Refactor code
* Hyperparameter search on base experiment.

---------------------------
Current TODO:
* Write code so that it returns loss on the "training data" from throughout the training. 
* Write code that plots the loss evolution of multiple runs. Using matplotlib.
* Fix code for lr search so that it is only applied once in a special script.
  * Also fix so that it uses the training data as well, not the validation data. But still use val data for early stopping. In other words exactly same setup as normal training.

Done today (2021-04-27):
* Use Adam
* Write simple plot code for a list of list of points as a function of number of iterations assuming identical batch size.
* Use default lr scheduler not cyclic, so multistep scheduler.
* TODO: Return losses throughout training (not every epoch but like every n:th). Use this and the n to plot different loss graphs and see which lr is actually best rather than just the one with best at the end.
* Ran experiment with only person, full data even no annotation images. Got pretty good but took long, also still had images e.g cows or car lights mistaken as human. Ran for 11239 iterations, stopped on 3rd early stopping check.
* NOOOOOTEEEEEE: Detectron2 removes images that physically has no "annotations"-key in them, NOTTTTTTT images with ds\["annotations"\] = []. So I have had to manually change ds creator to include/exclude images to test empty vs no empty images instead. 
* Ran experiment with only person, data excluding images with no annotations. Also seems like pretty good results. Both experiments seem to have quick stagnation though, so probably need to reduce learning rate earlier than the preset because only 1 label is used probably.
* notation [see implementation here](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/roi_heads/fast_rcnn.py)
  * gt = ground truth
  * cls_accuracy = fraction of annotated boxes that were correctly classified as it's respective gt class or correctly classified as background (bg).
  * false_negative = index out (extract) the predictions that SHOULD be of any foreground class. Then, count how many of these were classified as background boxes. Then, divide this by the number of foreground (fg) images there SHOULD be, that is the false negative rate (how many were incorrectly classified as background).
  * fg_cls_accuracy: extract the predictions that SHOULD be any foreground class, check how many of those are of the correct class. For a single label, this is essentially the complement of false_negative.

Done today (2021-04-28):
* LR search for 0 comp labels in pascal voc resolution 10, found that lr 0.0001 works best i.e 1e-4. Didn't need early stopping after 4 epochs meaning it could see improvement beyond that which is also a nice sign.
* Also did manual testing and investigation of the parameters. It seems that loss_cls and loss_reg are the ones that are noisiest and highest, the others seem to work fine. Really weird. Perhaps this means that one half of the net is acting up? Yes, it seems like the box head (that predicts the class and location of a box) has the worst loss. But it also seems like the RPN severely overfit to the train data since the box sizes are very bad on test data.

TODO:
* during evaluation, have 0.05. But during visualization, have 0.5 or higher (threshold/NMS? Whichever one is 0.05 by default).
  * Fixed, although needs to be done at model building time, not just randomly afterwards.
* Look into potential bugs in visualization
  * Fixed, I never actually loaded the model during evaluation lmao.
  * Also, configuration of thresholds etc is done ONCE during model construction, cannot be arbitrarily changed afterwards. If you want to evaluate and visualize using same model, you should load two different models.
  * Also, I noticed one could use [visualize_training for visualization](https://github.com/facebookresearch/detectron2/blob/master/detectron2/modeling/meta_arch/rcnn.py#L87) 
* mAP COCO on trained models
  * Fixed, had to mod the COCOEvaluator. There is a potential bug but as long as we have proper testing data that includes at least 1 instance of each used label we should be good.
* Big initial drops in loss is common in obj. det.
* Go for including empty images, and don't filter out empty images through detectron2. This is because it is natural to have images e.g that don't have tumors, so contextualizing "otherwise empty images" is a valid use-case.
* Look for obj. det. dataset or consider reusing CSAW from segmentation task.
  * [New dataset](https://nucls.grand-challenge.org/NuCLS/) but annotated largely by non-pathologists, so probably pretty noisy (about 50% was approved by pathologists, the rest was not, but NPs got examples to go off of. However, still feel like this is... bad). Has many classes tho, like 11 that can be included in analysis, they are grouped as well.
  * [CSAW-S](https://zenodo.org/record/4030660#.YJGG4SaxU5l), but convert segmentations to bboxes should be straight forward enough. The labels to keep are cancer, calcification, lymph node, foreign object, nipple, non-mammary tissue, text and perhaps also pectoral muscle/mammary gland? Let's say 7 guaranteed and maybe 9. 6 of more worth than the others.
* Discuss with kevin whether voc+coco, or just voc or just coco. But pascal has fewer instances per image compared to coco.
* Currently working on getting datasets to not filter out empty images as well as logging data.
  * done, needed to create logger that also backs up to log file
* TODO: Still need to fix logger in COCOEvaluatorMODDED, not logging the full AP table, probably again due to logging levels or not setting it up using setup_logger.
  * Done
* TODO: Before meeting, train 1 model without empty images with same 3 comp labels and same config otherwise, and look at how AP is affected. Cause I feel like this run was very trash.
  * Done, still trash. It was due to underfitting

Final things before "automating" rest of process; clean up code. set up csaw-s for object detection. modify code to run lr search for one epoch (not with early stopping since then we have just trained a whole thing lol). perform lr search once on no comp labels to get base lr for next few numbers of comp labels, then start using the settings used for full comp label set (I actually also train on pretrained imagenet, dope). 

Be VERY careful that when loading a model for evaluation, the 1. correct comp labels and 2. the correct order is used!!!!!!!!!!!!

Final things before "automating" rest of process; clean up code. set up csaw-s for object detection. modify code to run lr search for one epoch (not with early stopping since then we have just trained a whole thing lol). perform lr search once on no comp labels to get base lr for next few numbers of comp labels, then start using the settings used for full comp label set (I actually also train on pretrained imagenet, dope). 
