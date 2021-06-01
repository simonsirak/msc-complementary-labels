import torch

from detectron2.data import MetadataCatalog, DatasetCatalog, DatasetMapper
from detectron2.data.common import DatasetFromList, MapDataset
from detectron2.data.samplers import InferenceSampler

from util.COCOEvaluator import COCOEvaluator

# builds a loader that iterates over the dataset once,
# like build_detection_test_loader, but with arbitrary
# batch size.

from tqdm import tqdm

def identity(batch):
  return batch

def build_eval_loader(dataset, *, mapper, sampler=None, num_workers=0, batch_size=1):
  if isinstance(dataset, list):
    dataset = DatasetFromList(dataset, copy=False)
  if mapper is not None:
    dataset = MapDataset(dataset, mapper)
  if sampler is None:
    sampler = InferenceSampler(len(dataset))
  # Always use 1 image per worker during inference since this is the
  # standard when reporting inference time in papers.
  batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, batch_size, drop_last=False)
  data_loader = torch.utils.data.DataLoader(
    dataset,
    num_workers=num_workers,
    batch_sampler=batch_sampler,
    collate_fn=identity, # identity function
  )

  return data_loader

def evaluate(cfg, model, logger, dataset_index=1):
  # build data loader, essentially equivalent to test loader but 
  # with arbitrary batch size because inference time is not a metric I want.
  data_loader = build_eval_loader( # test loader would use batch size 1 for benchmarking, very slow
    DatasetCatalog.get(cfg.DATASETS.TEST[dataset_index]), #TODO: idk WHY but the early stopping code goes past the size of the validation set... either the test set (which is larger) is used, or i accidentally constructed too many epochs.... OOOOOOOOOOOORRRRRRRRRRRRRRRRR the training data loader is literally infinite, i.e it loops forever! LMAO
    batch_size=cfg.SOLVER.IMS_PER_BATCH,
    num_workers=cfg.DATALOADER.NUM_WORKERS,
    mapper=DatasetMapper(cfg,False), #do_train=True means we are in training mode.
    #aspect_ratio_grouping=False
  )

  with torch.no_grad():
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[dataset_index], output_dir=cfg.OUTPUT_DIR, distributed=True, tasks=("bbox",))
    evaluator.reset()
    logger.info(f'Starting COCO evaluation preprocessing ... ')
    for data in tqdm(data_loader):
      evaluator.process(data, model(data))
    logger.info("begin coco evaluation...")
    eval_results = evaluator.evaluate()
    logger.info("finished coco evaluation!")
  return eval_results