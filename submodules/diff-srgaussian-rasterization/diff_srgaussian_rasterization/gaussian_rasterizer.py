# Check https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0/diff_gaussian_rasterization/__init__.py
# as a guide to implement this module

import torch
import torch.nn as nn
from . import _C

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward():
        pass

    @staticmethod
    def backward():
        pass

class GaussianRasterizer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        return _RasterizeGaussians.apply()
