from detectron2.engine.hooks import HookBase
from detectron2.evaluation import inference_context
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import DatasetMapper, build_detection_test_loader
import detectron2.utils.comm as comm
import torch
import time
import datetime
import numpy as np 
import logging

# TODO: ADD PATIENCE AND BEST MODEL, AND STOP IF PATIENCE EXCEEDED
class LossEvalHook:
  def __init__(self, eval_period, model, model_name, data_loader, checkpointer, patience=0):
    self._model = model
    self._model_name = model_name
    self._period = eval_period
    self._data_loader = data_loader
    self._patience = patience # patience is specified in evaluation period units (e.g 3 evaluation periods)
    self._cur_patience = 0
    self._checkpointer = checkpointer
    self._min_loss = float('inf')
    self._latest_loss = float('inf') 
  
  def _do_loss_eval(self, cur_iter, storage):
    # Copying inference_on_dataset from evaluator.py
    total = len(self._data_loader)
    num_warmup = min(5, total - 1)
        
    start_time = time.perf_counter()
    total_compute_time = 0
    losses = []
    stop_early = False
    for idx, inputs in enumerate(self._data_loader):       
      if idx == num_warmup:
        start_time = time.perf_counter()
        total_compute_time = 0
      start_compute_time = time.perf_counter()
      if torch.cuda.is_available():
        torch.cuda.synchronize()
      total_compute_time += time.perf_counter() - start_compute_time
      iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
      seconds_per_img = total_compute_time / iters_after_start
      if idx >= num_warmup * 2 or seconds_per_img > 5:
        total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
        eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
        # print("Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
        #     idx + 1, total, seconds_per_img, str(eta)
        #   ))
        log_every_n_seconds(
          logging.INFO,
          "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
            idx + 1, total, seconds_per_img, str(eta)
          ),
          n=5,
          name="detectron2"
        )
      loss_batch = self._get_loss(inputs)
      losses.append(loss_batch)
    mean_loss = np.mean(losses)

    self._latest_loss = mean_loss
    if self._latest_loss < self._min_loss:
      self._patience = 0
      self._min_loss = self._latest_loss
      self._checkpointer.save(self._model_name + "_" + str(cur_iter))
    else:
      self._cur_patience += 1
      if self._cur_patience > self._patience:
        stop_early = True
    storage.put_scalar('validation_loss', self._latest_loss)
    comm.synchronize()

    return (losses, stop_early)
          
  def _get_loss(self, data):
      # How loss is calculated on train_loop 
      # Note that no backwards step is done so no updates are made
      metrics_dict = self._model(data)
      metrics_dict = {
          k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
          for k, v in metrics_dict.items()
      }
      total_losses_reduced = sum(loss for loss in metrics_dict.values())
      return total_losses_reduced
      
  def after_step(self, cur_iter, max_iter, storage):
    next_iter = cur_iter + 1
    is_final = next_iter == max_iter
    if is_final or (self._period > 0 and next_iter % self._period == 0):
        (_, stop_early) = self._do_loss_eval(cur_iter, storage)
        return stop_early
      #print("validation loss hook finished!")
    return False
