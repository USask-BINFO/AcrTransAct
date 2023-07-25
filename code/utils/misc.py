import torch
import gc
import numpy as np
import random
import os

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