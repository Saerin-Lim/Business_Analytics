import random
import torch
import numpy as np

def set_seed(seed:int=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
