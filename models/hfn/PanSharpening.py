import torch
import torch.nn as nn

from .ResidualChannelAttentionBlock import ResidualChannelAttentionBlock
from .ResidualDenseBlock import ResidualDenseBlock

class PanSharpening(nn.Module):
    def __init__(self, channels, lr_channels, features, reduction, n_layers, growth_rate, n_rdb, kernel_size):
        super(PanSharpening, self).__init__()
        
        self.conv1 = nn.Conv2d(channels, features, kernel_size, padding=kernel_size//2)
        self.relu1 = nn.ReLU()
        
        self.rcab = ResidualChannelAttentionBlock(
            features=features,
            reduction=reduction,
            kernel_size=kernel_size
        )
        
        self.rdbs = nn.Sequential(
            *[ResidualDenseBlock(features, n_layers, growth_rate) for _ in range(n_rdb)]
        )
        
        self.conv2 = nn.Conv2d(features, lr_channels, kernel_size, padding=kernel_size//2)

    def forward(self, inputs):
        sr, hr = inputs
        
        x1 = torch.cat([sr, hr], dim=1)

        x1 = self.relu1(self.conv1(x1))
        x1 = self.rcab(x1)
        x1 = self.rdbs(x1)
        x1 = self.conv2(x1)
        
        return sr + x1