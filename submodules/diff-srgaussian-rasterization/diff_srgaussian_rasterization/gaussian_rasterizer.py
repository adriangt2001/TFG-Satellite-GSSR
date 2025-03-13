# Check https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0/diff_gaussian_rasterization/__init__.py
# as a guide to implement this module

import torch
import torch.nn as nn
from . import _C

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        opacity,
        means,
        stds,
        rhos,
        colors,
        image_height,
        image_width,
        scale_factor,
        raster_ratio
    ):
        args = (
            opacity,
            means,
            stds,
            rhos,
            colors,
            image_height,
            image_width,
            scale_factor,
            raster_ratio
        )

        # The output of this methos should include relevant
        # tensors for backward (Don't know which ones)
        out_image = _C.rasterize_gaussians(*args)

        ctx.num_rendered = num_rendered
        ctx.save_for_backward()

    @staticmethod
    def backward(
        ctx,
        
    ):
        args = (

        )

        # Invoke _C backward method

        grads = (
            
            None,
        )

class GaussianRasterizer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self):
        return _RasterizeGaussians.apply()
