# general imports
import argparse 
import sys
import random
import os

# detectron2 stuff
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.utils.logger import setup_logger
from detectron2.config import get_cfg
from detectron2.utils import comm
from detectron2.utils.env import seed_all_rng

import logging

# for experiments
from util.helpers import extract_dataset
from experiments.experiments import base_experiment, loo_experiment, sample_experiment, lr_experiment, vary_data_experiment, vary_labels_experiment, longrun, coco
def main(args):
  # if args.create_all_datasets:
  # * create subsets for experiments
  # * do lr search for each of them (no intermediate val_loss)

  rank = comm.get_rank()
  dataset_name = args.dataset 
  main_label = args.main_label
  base_dataset = extract_dataset(dataset_name, main_label, args)
  # base_dataset.print_percentage_of_occurrence_of_label_in_images()
  # exit(0)
  # print(base_dataset.base_dict_func["train"][0]["file_name"])
  
  # default_setup literally only sets the cfg rng seed, the output directory, and whether cudnn.benchmark should be used.
  # I only load it because of the setup.
  base_cfg = get_cfg()
  base_cfg.SEED = args.seed
  base_cfg.OUTPUT_DIR = args.output_dir

  # https://towardsdatascience.com/properly-setting-the-random-seed-in-machine-learning-experiments-7da298d1320b

  # set seed for all frameworks used (python hashing, python random, numpy, pytorch/detectron2)
  # also sets up logger and some cudnn benchmark thingy that idk. 
  # TODO: Make sure the seeding is redone before each experiment
  # (i.e once before base experiments, once before leave-one-out, 
  # once before varying labels etc). Only once tho, not for each 
  # repetition of each experiment!
  default_setup(base_cfg, args)
  main_logger = setup_logger(args.output_dir, distributed_rank=comm.get_rank(), name="main")
  
  main_logger.info("Dataset loaded successfully, basic configuration completed.")
  
  if args.sample:
    main_logger.info("Entering sample experiment...")
    sample_experiment(args, base_dataset)
    main_logger.info("Sample experiment finished!")

  if args.base:
    seed_all_rng(None if base_cfg.SEED < 0 else base_cfg.SEED + rank)
    main_logger.info("Entering base experiment...")
    training_size = "full" if dataset_name == "CSAW-S" else 256
    base_experiment(args, base_dataset, training_size=training_size, use_complementary_labels=False)
    base_experiment(args, base_dataset, training_size=training_size, use_complementary_labels=True)
    main_logger.info("Base experiment finished!")
    
  if args.leave_one_out:
    seed_all_rng(None if base_cfg.SEED < 0 else base_cfg.SEED + rank)
    main_logger.info("Entering leave-one-out experiment...")
    training_size = "full" if dataset_name == "CSAW-S" else 256
    loo_experiment(args, base_dataset, training_size=training_size)
    main_logger.info("Leave-one-out experiments finished!")
    
  if args.vary_data:
    seed_all_rng(None if base_cfg.SEED < 0 else base_cfg.SEED + rank)
    main_logger.info("Entering vary data experiment...")
    sizes = [5, 10, 25, 50, 100, 200] if dataset_name == "CSAW-S" else [64, 128, 512, 1024, 2048]
    # natural domains get slightly higher regimes overall, 
    # to accomodate for possibility of getting no main label 
    # in an image
    vary_data_experiment(args, base_dataset, sizes=sizes) # chosen
    main_logger.info("Vary data experiments finished!")
    
  if args.vary_labels:
    seed_all_rng(None if base_cfg.SEED < 0 else base_cfg.SEED + rank)
    main_logger.info("Entering vary labels experiment...")
    sizes = [1, 3, 5] if dataset_name == "CSAW-S" else [1, 2, 4, 8, 16, 32, 64]
    training_size = "full" if dataset_name == "CSAW-S" else 256
    vary_labels_experiment(args, base_dataset, sizes=sizes, training_size=training_size)
    main_logger.info("Vary labels experiments finished!")
    
  if args.lr:
    seed_all_rng(None if base_cfg.SEED < 0 else base_cfg.SEED + rank)
    main_logger.info("Entering lr finder...")
    if args.dataset == "CSAW-S" and args.dataset_size == 263:
      lr_experiment(args, base_dataset, n_comp=args.num_comp_labels, training_size="full")
    else:
      lr_experiment(args, base_dataset, n_comp=args.num_comp_labels, training_size=args.dataset_size)
    main_logger.info("lr finder finished!")

  if args.longrun:
    seed_all_rng(None if base_cfg.SEED < 0 else base_cfg.SEED + rank)
    main_logger.info("Entering longrun...")
    if args.dataset == "CSAW-S" and args.dataset_size == 263:
      longrun(args, base_dataset, training_size="full")
    else:
      longrun(args, base_dataset, training_size=args.dataset_size)
    main_logger.info("longrun finished!")

  if args.coco:
    seed_all_rng(None if base_cfg.SEED < 0 else base_cfg.SEED + rank)
    main_logger.info("Entering coco eval...")
    coco(args, base_dataset, weights_path=args.weights_path, nb_comp_labels=args.num_comp_labels)
    main_logger.info("coco eval finished!")

import argparse 
import sys
def argument_parser():
    """
    Create a parser with some common arguments used by detectron2 users.

    Args:
        epilog (str): epilog passed to ArgumentParser describing the usage.

    Returns:
        argparse.ArgumentParser:
    """
    parser = argparse.ArgumentParser(f"""
Examples:

Run on single machine:
    $ {sys.argv[0]} --num-gpus 8 --config-file cfg.yaml

Change some config options:
    $ {sys.argv[0]} --config-file cfg.yaml MODEL.WEIGHTS /path/to/weight.pth SOLVER.BASE_LR 0.001

Run on multiple machines:
    (machine0)$ {sys.argv[0]} --machine-rank 0 --num-machines 2 --dist-url <URL> [--other-flags]
    (machine1)$ {sys.argv[0]} --machine-rank 1 --num-machines 2 --dist-url <URL> [--other-flags]
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Whether to attempt to resume from the checkpoint directory. "
        "See documentation of `DefaultTrainer.resume_or_load()` for what it means.",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    parser.add_argument("--dataset", default="PascalVOC2007", help="dataset used for training and evaluation: 'PascalVOC2007', 'CSAW-S' or 'MSCOCO'")
    parser.add_argument("--dataset-path", default="../datasets", help="the directory to the dataset")
    parser.add_argument("--main-label", default="person", help="main label used for training and evaluation")
 
    parser.add_argument("--seed", type=int, default=-1, help="seed used for randomization")
    
    parser.add_argument("--sample", action="store_true", default=False, help="perform sample inference on some images")

    # experiments to run in one sitting
    parser.add_argument("--base", action="store_true", default=False, help="perform base experiment")
    parser.add_argument("--leave-one-out", action="store_true", default=False, help="perform leave one out experiment")
    parser.add_argument("--vary-data", action="store_true", default=False, help="perform varying data experiments")
    parser.add_argument("--vary-labels", action="store_true", default=False, help="perform varying complementary labels experiments")

    parser.add_argument("--num-comp-labels", type=int, default=0, help="number of complementary labels for vary label experiments")
    parser.add_argument("--dataset-size", type=int, default=263, help="training subset size during all experiments except for the vary data experiments")

    parser.add_argument("--eval", action="store_true", default=False, help="perform evaluation")
    parser.add_argument("--coco", action="store_true", default=False, help="perform coco evaluation")
    parser.add_argument("--weights-path", default=".", help="the path to model weights; only needed for --coco")
    parser.add_argument("--lr", action="store_true", default=False, help="learning rate search")
    parser.add_argument("--longrun", action="store_true", default=False, help="do a longrun with specified dataset size, no complementary labels")
    parser.add_argument("--input-dir", default="output", help="the directory to read input task-related data such as trained models. By default, uses the output of the current executiion")
    parser.add_argument("--output-dir", default="output", help="the directory to output task-related data")
    

    # TODO: Maybe an argument for datasets directory.

    # PyTorch still may leave orphan processes in multi-gpu training.
    # Therefore we use a deterministic way to obtain port,
    # so that users are aware of orphan processes by seeing the port occupied.
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument("--id", type=int, default=0, help="id of current training session, used to create separate tcp session port for each session in case that matters (not sure if it does)")
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )
    return parser

import torch
if __name__ == "__main__":
    torch.set_num_threads(8)
    args = argument_parser().parse_args()
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        # machine_rank=args.machine_rank, # default = 0
        dist_url="tcp://127.0.0.1:{}".format(port + args.id),
        args=(args,),
    )