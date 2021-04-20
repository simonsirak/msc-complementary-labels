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
* Refactor code
* Hyperparameter search on base experiment.