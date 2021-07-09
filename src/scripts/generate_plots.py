import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import json

def plotter(mode, ax, data1, data2, param_dict = {}):
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

    if mode == "line":
      out = ax.plot(data1, data2, **param_dict)
    elif mode == "box":
      out = ax.boxplot(data2, labels=data1, **param_dict)
    return out

# explanation of the various notation:
# https://se.mathworks.com/matlabcentral/answers/461791-what-do-lines-that-double-back-on-themselves-mean-box-plots

# explanation of why bootstrapping:
# https://ocw.mit.edu/courses/mathematics/18-05-introduction-to-probability-and-statistics-spring-2014/readings/MIT18_05S14_Reading24.pdf
def differential(labels, data, main_label, args):
  ys = []
  for point in data:
    y = []
    for p in point:
      y.append(p["bbox"][main_label]["AP"])
    ys.append(y)

  baseline = np.asarray(ys[-1])
  ys = ys[:-1]
  ys_array = [np.asarray(y) for y in ys]
  baseline_mean = np.mean(baseline)
  # ci = 1.96 * np.std(ys_array, axis=1)/y_mean
  fig, ax = plt.subplots()
  # print(y_mean, labels)
  added_artists = plotter('box', ax, labels, ys_array, {
    "notch": True, 
    "patch_artist": True, 
    "bootstrap": 1000,
    #"showfliers": False,
    "boxprops": {"color": (150./255,200./255,150./255,1), "facecolor": (150./255,200./255,150./255,1)}})
  ax.axhline(y=baseline_mean, color='r', linestyle='--')
  ax.set_ylabel('AP')
  ax.set_title(f"{args.experiment}, {args.dataset}")
  plt.show()

import seaborn as sns
import pandas as pd
def graph(x, data, main_label, args):
  # preprocessing
  processed_data = []
  for d in data:
    ys = []
    for point in d:
      y = []
      for p in point:
        y.append(p["bbox"][main_label]["AP"])
      ys.append(y)
    processed_data.append(ys)

  if len(x) == 1:
    ys = processed_data[0]
    x = x[0]
    ys_array = np.asarray(ys)
    # y_mean = np.mean(ys_array, axis=1)
    # ci = 1.96 * np.std(ys_array, axis=1)/y_mean
    fig, ax = plt.subplots()
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    # print(np.zeros((len(x), 5), dtype=np.int).T + np.asarray(x, dtype=np.int))
    # print(ys_array)
    xs_array = (np.zeros((len(x), 5), dtype=np.int).T + np.asarray(x, dtype=np.int)).T
    x_desc = 'number of labels' if args.experiment == "vary-labels" else 'size of subset'
    d = {x_desc: np.repeat(np.asarray(x), 5), 'AP': ys_array.flatten()}
    print(ys_array.flatten())
    #exit(0)
    frame = pd.DataFrame(d)
    print(frame)
    sns.lineplot(ax = ax,
              data = frame,
              x = x_desc, y = 'AP',
              ci = 68,
              n_boot=1000,
              marker='o')
    # added_artists = plotter('line', ax, x, y_mean, {'marker': 'o'})
    ax.set_title(f"{args.experiment}, {args.dataset}")
    # ax.fill_between(x, (y_mean-ci), (y_mean+ci), alpha=.1)
  elif len(x) == 2:
    ys_all = np.asarray(processed_data[0])
    x_all = x[0]
    # y_mean = np.mean(ys_array, axis=1)
    # ci = 1.96 * np.std(ys_array, axis=1)/y_mean
    fig, ax = plt.subplots()
    ax.set_xticks(x_all)
    ax.set_xticklabels(x_all)
    # print(np.zeros((len(x), 5), dtype=np.int).T + np.asarray(x, dtype=np.int))
    # print(ys_array)
    # xs_array = (np.zeros((len(x_all), 5), dtype=np.int).T + np.asarray(x_all, dtype=np.int)).T
    x_desc = 'number of labels' if args.experiment == "vary-labels" else 'size of subset'
    
    ys_zero = np.asarray(processed_data[1])
    x_zero = x[1]
    subset_1 = np.repeat(np.asarray(['all labels' for i in range(len(x_all))], dtype=object), 5)
    subset_2 = np.repeat(np.asarray(['zero labels' for i in range(len(x_zero))], dtype=object), 5)
    d = {
      x_desc: np.hstack((np.repeat(np.asarray(x_all), 5), np.repeat(np.asarray(x_zero), 5))), 
      'AP': np.hstack((ys_all.flatten(), ys_zero.flatten())),
      'mode': np.hstack((subset_1, subset_2))
    }
    print(ys_all.flatten())
    #exit(0)
    frame = pd.DataFrame(d)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
      print(frame)
    sns.lineplot(ax = ax,
              data = frame,
              x = x_desc, y = 'AP',
              hue = 'mode',
              ci = 68,
              n_boot=1000,
              marker='o')
    # added_artists = plotter('line', ax, x, y_mean, {'marker': 'o'})
    ax.set_title(f"{args.experiment}, {args.dataset}")
    # ax.fill_between(x, (y_mean-ci), (y_mean+ci), alpha=.1)

  plt.savefig(f"{args.experiment}-{args.dataset}.png")
  # plt.show()

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
    if "nocomp" not in path:
      with open(path, 'r') as fr:
        data.append(json.load(fr))

  if args.plot == "bar":
    pass # TODO

  elif args.plot == "graph":
    if args.experiment == "vary-labels":
      with open(os.path.join(result_path, f"metrics-base-zero.json"), 'r') as fr:
        data.insert(0, json.load(fr))
      with open(os.path.join(result_path, f"metrics-base-all.json"), 'r') as fr:
        data.append(json.load(fr))

      data_x = [int(path.split('/')[-1].split('.')[0].split('-')[-1]) for path in paths]
      # organize data based on dataset
      if args.dataset == "CSAW-S":
        data_x.insert(0, 0)
        data_x.append(7)
      elif args.dataset == "MSCOCO":
        data_x.insert(0, 0)
        data_x.append(79)

      graph([data_x], [data], main_label, args)

    if args.experiment == "vary-data":
      with open(os.path.join(result_path, f"metrics-base-all.json"), 'r') as fr:
        data.append(json.load(fr))

      # organize data based on dataset
      data_x = [int(path.split('/')[-1].split('.')[0].split('-')[-1]) for path in paths if "nocomp" not in path]
      data_x.append(263 if args.dataset == "CSAW-S" else 256)

      paths = glob(os.path.join(result_path, f"metrics-{args.experiment}-*-nocomp.json"))
      data_zero = []
      for path in paths:
        with open(path, 'r') as fr:
          data_zero.append(json.load(fr))

      with open(os.path.join(result_path, f"metrics-base-zero.json"), 'r') as fr:
        data_zero.append(json.load(fr))
      
      data_x_zero = [int(path.split('/')[-1].split('.')[0].split('-')[-2]) for path in paths]
      data_x_zero.append(263 if args.dataset == "CSAW-S" else 256)

      graph([data_x, data_x_zero], [data, data_zero], main_label, args)

      plt.show()

  elif args.plot == "differential":
    labels = [path.split('/')[-1].split('.')[0].split('-')[-1] for path in paths]
    if args.experiment in ["loo", "vary-data"]:
      with open(os.path.join(result_path, f"metrics-base-all.json"), 'r') as fr:
        data.append(json.load(fr)) # in other words, data with all labels is last index
    else: # if experiment is vary-labels or (weirdly) base
      with open(os.path.join(result_path, f"metrics-base-zero.json"), 'r') as fr:
        data.append(json.load(fr)) # in other words, data with all labels is last index

    differential(labels, data, main_label, args)

import argparse
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--path", default="../../results", help="path to directory containing the results for each dataset. Default: '../../results'")
  parser.add_argument("--dataset", default="MSCOCO", help="dataset under consideration. Must be either 'CSAW-S' or 'MSCOCO'. Default: 'MSCOCO'")
  parser.add_argument("--experiment", default="base", help="one of [base, loo, vary-labels, vary-data]. Default: 'base'")
  parser.add_argument("--plot", default="bar", help="one of [bar, graph, differential]. Default: 'graph'")
  args = parser.parse_args()
  main(args)