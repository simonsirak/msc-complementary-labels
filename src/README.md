Sample commands:

Below command runs on PascalVOC2007 and outputs in the specified output directory, with 2 gpus, 
running the base experiment.
```bash
python main.py --output-dir="../../checkpoints/base" --base --dataset-path="../../datasets" --num-gpus=2

python main.py --output-dir="../../checkpoints/base" --base --dataset-path="../../datasets" --num-gpus=2 --dataset="MSCOCO" --main-label="person"

python main.py --output-dir="../../checkpoints/base" --base --dataset-path="../../datasets" --num-gpus=2 --dataset="CSAW-S" --main-label="cancer"

rm -rf /storage/ssirak/checkpoints/lr/CSAW-S && python main.py --output-dir="../../checkpoints/lr/CSAW-S" --dataset-path="../../datasets" --num-gpus=2 --dataset="CSAW-S" --main-label="cancer" --seed=898 --lr --dataset-size=263

rm -rf /storage/ssirak/checkpoints/lr/MSCOCO && python main.py --output-dir="../../checkpoints/lr/MSCOCO" --dataset-path="../../datasets" --num-gpus=2 --dataset="MSCOCO" --main-label="person" --seed=898 --lr --dataset-size=256

python main.py --output-dir="../../checkpoints/lr/CSAW-S/100" --dataset-path="../../datasets" --num-gpus=2 --dataset="CSAW-S" --main-label="cancer" --seed=898 --lr --dataset-size=100
```