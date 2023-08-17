import torch
import gc
import numpy as np
import random
import os
from argparse import ArgumentParser
import re

def free_mem():
    torch.cuda.empty_cache()
    gc.collect()


def seed_everything(seed=42):
    """Seed everything for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--sweep_id", type=str, default=None)
    parser.add_argument("--model_type", type=str, default="CNN", help="CNN or LSTM")
    parser.add_argument("--esm_version", type=str, default="8m")
    parser.add_argument("--label_smoothing", type=bool, default=0.1)
    parser.add_argument(
        "--optimize_metric", type=str, default="F1_CV"
    )  # F1_CV, Loss_CV
    parser.add_argument(
        "--monitor_ckpt", type=str, default="val_f1_score"
    )  # val_f1_score, val_loss
    parser.add_argument(
        "--feature_mode",
        type=int,
        default=4,
        help="1: ESM, 2: SF, 3: one-hot AA, 4: SF + ESM",
    )
    parser.add_argument("--do_sweep", type=bool, default=True)
    parser.add_argument("--wandb_log", type=bool, default=False)
    parser.add_argument("--cross_val", type=bool, default=False)
    parser.add_argument("--undersample", type=bool, default=False)
    parser.add_argument(
        "--excl_mode",
        type=int,
        default=0,
        help="0: no exclusion, 1: exclude K12, ATCC_IF, 2: exclude K12, PaLML1_DVT419",
    )
    parser.add_argument("--run_mode", type=str, default="train", help="train or eval")

    args = parser.parse_args()
    return args


def find_nums(input_str):
    """This function finds the numbers in a string and returns them as a list
    ----------
    Returns
    -------
    nums : list
        A list of string which are numbers found in the input"""

    nums = re.findall(r"\d+", input_str)
    return nums