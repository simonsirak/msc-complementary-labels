#!/bin/bash

# loo
python3 generate_plots.py --dataset="CSAW-S" --plot="differential" --path="../../results-v3" --experiment="loo";
python3 generate_plots.py --dataset="MSCOCO" --plot="differential" --path="../../results-v3" --experiment="loo";

# vary data
python3 generate_plots.py --dataset="CSAW-S" --plot="graph" --path="../../results-v3" --experiment="vary-data";
python3 generate_plots.py --dataset="MSCOCO" --plot="graph" --path="../../results-v3" --experiment="vary-data";

# vary labels
python3 generate_plots.py --dataset="CSAW-S" --plot="graph" --path="../../results-v3" --experiment="vary-labels";
python3 generate_plots.py --dataset="MSCOCO" --plot="graph" --path="../../results-v3" --experiment="vary-labels";

# base
python3 generate_plots.py --dataset="CSAW-S" --plot="bar" --path="../../results-v3" --experiment="base";
python3 generate_plots.py --dataset="MSCOCO" --plot="bar" --path="../../results-v3" --experiment="base";