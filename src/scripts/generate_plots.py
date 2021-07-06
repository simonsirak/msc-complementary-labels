import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import json

def plotter(ax, data1, data2, param_dict):
    """
    A helper function to make a graph

    Parameters
    ----------
    ax : Axes
        The axes to draw to

    data1 : array
       The x data

    data2 : array
       The y data

    param_dict : dict
       Dictionary of kwargs to pass to ax.plot

    Returns
    -------
    out : list
        list of artists added
    """
    out = ax.plot(data1, data2, **param_dict)
    return out

def differential(data):
  pass

def graph(data):
  for point in data:
    
  pass

def bar(data):
  pass 

def main(args):
  assert args.dataset in ["CSAW-S", "MSCOCO"], f"dataset '{args.dataset}' not supported!"
  assert args.experiment in ["base", "loo", "vary-labels", "vary-data"], f"experiment '{args.experiment}' not supported!"
  assert args.plot in ["bar", "graph", "differential"], f"plot type '{args.plot}' not supported!"

  main_label = "cancer" if args.dataset == "CSAW-S" else "person"
  
  result_path = os.path.join(args.path, args.dataset)
  print(result_path)

  paths = glob(os.path.join(result_path, f"metrics-{args.experiment}*.json"))
  data = []
  for path in paths:
    with open(path, 'r') as fr:
      data.append(json.load(fr))

  if args.plot == "bar":
    pass # TODO

  elif args.plot == "graph":
    if args.experiment == "vary-labels":
      with open(os.path.join(result_path, f"metrics-base-zero.json"), 'r') as fr:
        data.append(json.load(fr))
      with open(os.path.join(result_path, f"metrics-base-all.json"), 'r') as fr:
        data.append(json.load(fr))

    if args.experiment == "vary-data":
      with open(os.path.join(result_path, f"metrics-base-all.json"), 'r') as fr:
        data.append(json.load(fr))

  elif args.plot == "differential":
    if args.experiment in ["loo", "vary-data"]:
      with open(os.path.join(result_path, f"metrics-base-all.json"), 'r') as fr:
        data.append(json.load(fr)) # in other words, data with all labels is last index
    else: # if experiment is vary-labels or (weirdly) base
      with open(os.path.join(result_path, f"metrics-base-zero.json"), 'r') as fr:
        data.append(json.load(fr)) # in other words, data with all labels is last index

import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", default="../../results", help="path to directory containing the results for each dataset. Default: '../../results'")
  parser.add_argument("--dataset", default="MSCOCO", help="dataset under consideration. Must be either 'CSAW-S' or 'MSCOCO'. Default: 'MSCOCO'")
  parser.add_argument("--experiment", default="base", help="one of [base, loo, vary-labels, vary-data]. Default: 'base'")
  parser.add_argument("--plot", default="bar", help="one of [bar, graph, differential]. Default: 'graph'")
  args = parser.parse_args()
  main(args)