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
    
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
def initialize_weights(model: nn.Module, activation='relu'):
    """Initialize trainable weights."""

    for _, m in model.named_modules():

        if activation == 'relu':
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose1d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, a=math.sqrt(5), mode='fan_in', nonlinearity=activation)
                if m.bias is not None:
                    fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    nn.init.uniform_(m.bias, -bound, bound)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                try:
                    nn.init.constant_(m.bias, 1)
                except AttributeError:
                    pass

        else:
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose1d, nn.Conv1d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                try:
                    nn.init.constant_(m.bias, 1)
                except AttributeError:
                    pass