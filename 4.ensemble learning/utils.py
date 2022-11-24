import random
import torch
import math
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed:int=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)