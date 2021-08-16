import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import json
import scipy.stats
import itertools
def mean_confidence_interval(data, confidence=0.95):
  cis = []
  for point in data:
    a = 1.0 * np.asarray(point)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    cis.append(h)
    # return m, m-h, m+h
  return np.asarray(cis)
    
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
    elif mode == "bar":
      out = ax.bar(x=data1, height=data2, **param_dict)
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
  fig, ax = plt.subplots(figsize=(10,6))
  ax.grid(linestyle='--', dashes=(5,10,5,10), linewidth=1, zorder=0)
  for tick in ax.get_xticklabels():
    tick.set_rotation(45)
    tick.set_ha('right')
  # print(y_mean, labels)
  added_artists = plotter('box', ax, labels, ys_array, {
    "notch": True, 
    "patch_artist": True, 
    #"bootstrap": 1000,
    "zorder": 2,
    #"showfliers": False,
    "boxprops": {"color": (150./255,200./255,150./255,1), "facecolor": (150./255,200./255,150./255,1)}})
  ax.axhline(y=baseline_mean, color='r', linestyle='--')
  ax.set_ylabel('AP')
  ax.set_title(f"{args.experiment}, {args.dataset}")

  plt.autoscale()
  plt.tight_layout()
  plt.savefig(os.path.join(args.path, f"{args.experiment}-{args.dataset}.png"), dpi=300)
  # plt.show()

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
        # print(len(p))
        if not np.isnan(p["bbox"][main_label]["AP"]):
          y.append(p["bbox"][main_label]["AP"])
        
      while len(y) < 5:
        y.append(np.mean(np.asarray(y))) # pad with mean value if nan existed
        #print(len(y))
      print(len(y))

      ys.append(y)
    processed_data.append(ys)

  for i in range(len(x)):
    lists = sorted(zip(*[x[i], processed_data[i]]))
    x[i], processed_data[i] = list(zip(*lists))
  x_desc = 'number of labels' if args.experiment == "vary-labels" else 'size of subset'
  if len(x) == 1:
    ys = processed_data[0]
    x = x[0]

    ys_array = np.asarray(ys)
    y_mean = np.mean(ys_array, axis=1)
    # ci = 1.96 * np.std(ys_array, axis=1)/y_mean
    ci0 = mean_confidence_interval(ys, confidence=0.68)
    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(linestyle='--', dashes=(5,10,5,10), linewidth=1, zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(x)
    # print(np.zeros((len(x), 5), dtype=np.int).T + np.asarray(x, dtype=np.int))
    # print(ys_array)
    xs_array = np.asarray(x, dtype=int)
    # d = {x_desc: np.repeat(np.asarray(x), 5), 'AP': ys_array.flatten()}
    # print(ys_array.flatten())
    # #exit(0)
    # frame = pd.DataFrame(d)
    # print(frame)
    # sns.lineplot(ax = ax,
    #           data = frame,
    #           x = x_desc, y = 'AP',
    #           ci = 68,
    #           n_boot=1000,
    #           marker='o',
    #           # kwargs
    #           zorder= 2)
    added_artists = plotter('line', ax, x, y_mean, {'marker': 'o'})
    ax.set_title(f"{args.experiment}, {args.dataset}")
    ax.fill_between(x, (y_mean-ci0), (y_mean+ci0), alpha=.1)
    ax.set_xlabel(x_desc)
    ax.set_ylabel("AP")
  elif len(x) == 2:
    ys_all = np.asarray(processed_data[0])
    x_all = x[0]
    y_mean_all = np.mean(ys_all, axis=1)
    # ci = 1.96 * np.std(ys_array, axis=1)/y_mean
    fig, ax = plt.subplots(figsize=(10,6))
    ax.grid(linestyle='--', dashes=(5,10,5,10), linewidth=1, zorder=0)
    ax.set_xticks(x_all)
    ax.set_xticklabels(x_all)
    # print(np.zeros((len(x), 5), dtype=np.int).T + np.asarray(x, dtype=np.int))
    # print(ys_array)
    # xs_array = (np.zeros((len(x_all), 5), dtype=np.int).T + np.asarray(x_all, dtype=np.int)).T
    x_desc = 'number of labels' if args.experiment == "vary-labels" else 'size of subset'
    
    ys_zero = np.asarray(processed_data[1])
    y_mean_zero = np.mean(ys_zero, axis=1)
    x_zero = x[1]
    # subset_1 = np.repeat(np.asarray(['all labels' for i in range(len(x_all))], dtype=object), 5)
    # subset_2 = np.repeat(np.asarray(['zero labels' for i in range(len(x_zero))], dtype=object), 5)
    # d = {
    #   x_desc: np.hstack((np.repeat(np.asarray(x_all), 5), np.repeat(np.asarray(x_zero), 5))), 
    #   'AP': np.hstack((ys_all.flatten(), ys_zero.flatten())),
    #   'mode': np.hstack((subset_1, subset_2))
    # }
    # print(ys_all.flatten())
    # #exit(0)
    # frame = pd.DataFrame(d)
    # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    #   print(frame)
    # sns.lineplot(ax = ax,
    #           data = frame,
    #           x = x_desc, y = 'AP',
    #           hue = 'mode',
    #           ci = 68,
    #           n_boot=1000,
    #           marker='o',
    #           # kwargs
    #           zorder= 2)
    added_artists = plotter('line', ax, x_all, y_mean_all, {'marker': 'o'})
    ci_all = mean_confidence_interval(ys_all, confidence=0.68)
    ax.fill_between(x_all, (y_mean_all-ci_all), (y_mean_all+ci_all), alpha=.1)
    
    added_artists = plotter('line', ax, x_zero, y_mean_zero, {'marker': 'o'})
    ci_zero = mean_confidence_interval(ys_zero, confidence=0.68)
    ax.fill_between(x_zero, (y_mean_zero-ci_zero), (y_mean_zero+ci_zero), alpha=.1)

    ax.legend(['all complementary labels', 'no complementary labels'])
    ax.set_title(f"{args.experiment}, {args.dataset}")
    ax.set_xlabel(x_desc)
    ax.set_ylabel("AP")

  # for tick in ax.get_xticklabels():
  #   tick.set_rotation(90)
  plt.autoscale()
  plt.tight_layout()
  plt.savefig(os.path.join(args.path, f"{args.experiment}-{args.dataset}.png"), dpi=300)
  # plt.show()

def bar(labels, data, main_label, args):
  ys = []
  for point in data:
    y = []
    for p in point:
      y.append(p["bbox"][main_label]["AP"])
    ys.append(y)


  means = np.mean(np.asarray(ys), axis=1)
  print("MEANS", means[1] - means[0])
  ys_array = [np.asarray(y) for y in ys]
  ci = mean_confidence_interval(ys, confidence=0.95)
  print(ci.shape)
  fig, ax = plt.subplots(figsize=(10,6))
  ax.grid(linestyle='--', dashes=(5,10,5,10), linewidth=1, zorder=0)  
  # for tick in ax.get_xticklabels():
  #   tick.set_rotation(45)
  #   tick.set_ha('right')

  print(ci)
  print(labels)
  print(ys_array)
  added_artists = plotter('bar', ax, labels, np.mean(ys_array, axis=1), {
    "yerr": [y for y in ci], 
    "width": 0.5, 
    "capsize": 10,
    "color": ["tab:blue", "tab:orange"],
    "zorder": 2})

  ax.set_yticks(list(ax.get_yticks()) + means.tolist())

  ax.set_ylabel('AP')
  ax.set_title(f"{args.experiment}, {args.dataset}")

  # ys = np.asarray(ys)
  # subset_1 = np.repeat(np.asarray(['no complementary labels'], dtype=object), 5)
  # subset_2 = np.repeat(np.asarray(['all complementary labels'], dtype=object), 5)
  # d = {
  #   'AP': np.hstack((ys[0].flatten(), ys[1].flatten())),
  #   'mode': np.hstack((subset_1, subset_2))
  # }
  # # print(ys_all.flatten())
  # #exit(0)
  # frame = pd.DataFrame(d)
  # with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
  #   print(frame)
  # sns.barplot(ax = ax,
  #           data = frame,
  #           x = 'mode', y = 'AP',
  #           # hue = 'mode',
  #           ci = 95,
  #           n_boot=1000,
  #           capsize=0.1, 
  #           # kwargs
  #           zorder=2)
  #           # marker='o')
  # added_artists = plotter('line', ax, x, y_mean, {'marker': 'o'})
  ax.set_title(f"{args.experiment}, {args.dataset}")

  plt.autoscale()
  plt.tight_layout()
  plt.savefig(os.path.join(args.path, f"{args.experiment}-{args.dataset}.png"), dpi=300)

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
    labels = ["no complementary labels", "all complementary labels"]
    data = []
    with open(os.path.join(result_path, f"metrics-base-zero.json"), 'r') as fr:
      data.insert(0, json.load(fr))
    with open(os.path.join(result_path, f"metrics-base-all.json"), 'r') as fr:
      data.append(json.load(fr))
    bar(labels, data, main_label, args)
    # pass # TODO

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

  elif args.plot == "differential":
    labels = [' '.join('-'.join(path.split('/')[-1].split('.')[0].split('-')[2:]).split('_')) for path in paths]
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