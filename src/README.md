Sample commands:

Below command runs on PascalVOC2007 and outputs in the specified output directory, with 2 gpus, 
running the base experiment.
```bash
python main.py --output-dir="../../checkpoints/base" --base --dataset-path="../../datasets" --num-gpus=2

python main.py --output-dir="../../checkpoints/base" --base --dataset-path="../../datasets" --num-gpus=2 --dataset="MSCOCO" --main-label="person"


```