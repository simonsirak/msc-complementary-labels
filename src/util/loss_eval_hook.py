from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
from detectron2.data import MetadataCatalog, DatasetCatalog

import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np 
import math
import logging
from detectron2.utils.logger import setup_logger

from .evaluate import evaluate

# MovingMeans taken from ChrisMats
class MovingMeans:
    def __init__(self, window=5):
        self.window = window
        self.values = []
        
    def add(self, val):
        self.values.append(val)
        
    def get_value(self):
        return np.convolve(np.array(self.values), np.ones((self.window,))/self.window, mode='valid')[-1]

# TODO: ADD PATIENCE AND BEST MODEL, AND STOP IF PATIENCE EXCEEDED
class EarlyStoppingHook:
  def __init__(self, cfg, eval_period, model, model_name, dataset_name, checkpointer, patience=0, save_checkpoints=True):
    self.cfg = cfg
    self.model = model
    self.model_name = model_name
    self.period = eval_period
    self.dataset_name = dataset_name
    self.patience = patience # patience is specified in evaluation period units (e.g 3 evaluation periods)
    self.cur_patience = 0
    self.checkpointer = checkpointer
    self.max_ap = float('-inf')
    self.latest_ap = float('-inf') 
    self.moving_ap = MovingMeans(window=10)
    self.save_checkpoints = save_checkpoints
    self.logger = setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="earlystopping")
    
  
  def evaluate(self, cur_iter, storage):    
    stop_early = False
    result = evaluate(self.cfg, self.model, self.logger, dataset_index=0)
    if comm.is_main_process(): # non-main processes will get None/{}, hence the KeyError but correct logging on main process
      ap = result['bbox'][MetadataCatalog.get(self.dataset_name).main_label]['AP']

      self.latest_ap = ap
      if not np.isnan(self.latest_ap):
        # tensorboard plot
        storage.put_scalar('main_label_AP', self.latest_ap, smoothing_hint=False) # avoid NaN-ing up the main label AP plot
        
        # moving means logic
        self.moving_ap.add(self.latest_ap)
        
        # begin earlystopping logic once buffer is filled with sensible values
        if len(self.moving_ap.values) > self.moving_ap.window:
          cur_mov = self.moving_ap.get_value()
          
          # if moving avg is at least a bit higher than max recorded moving avg
          if cur_mov > self.max_ap + 0.1
            # reset patience
            self.cur_patience = 0

            # update new maximum
            self.max_ap = cur_mov
            if self.save_checkpoints:
              self.checkpointer.save(self.model_name)
          else:
            self.cur_patience += 1
            if self.cur_patience > self.patience:
              stop_early = True
      else:
        # if we have gotten a passable model before, do earlystopping logic
        # otherwise, do nothing since it is still early in training
        if not np.isinf(self.max_ap):
          self.cur_patience += 1
          if self.cur_patience > self.patience:
            stop_early = True

    stop_early = True in comm.all_gather(stop_early) # all processes now get the call to stop early! synchronized operation

    return (self.max_ap, stop_early)
      
  def after_step(self, cur_iter, max_iter, storage):
    next_iter = cur_iter + 1
    is_final = next_iter == max_iter
    if is_final or (self.period > 0 and next_iter % self.period == 0):
        (_, stop_early) = self.evaluate(cur_iter, storage)
        # self.logger.info("validation hook finished!")
        return stop_early
    # print("validation loss hook finished!")
    return False
