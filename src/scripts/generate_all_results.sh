#!/bin/bash

# loo
python generate_plots.py --dataset="CSAW-S" --plot="differential" --experiment="loo";
python generate_plots.py --dataset="MSCOCO" --plot="differential" --experiment="loo";

# vary data
python generate_plots.py --dataset="CSAW-S" --plot="graph" --experiment="vary-data"
python generate_plots.py --dataset="MSCOCO" --plot="graph" --experiment="vary-data"

# vary labels
python generate_plots.py --dataset="CSAW-S" --plot="graph" --experiment="vary-labels"
python generate_plots.py --dataset="MSCOCO" --plot="graph" --experiment="vary-labels"