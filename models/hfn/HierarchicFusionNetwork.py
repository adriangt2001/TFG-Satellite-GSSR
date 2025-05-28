import torch.nn as nn
import torchvision

from .SubPixel import SubPixel
from .PanSharpening import PanSharpening
from .ResidualDenseBlock import ResidualDenseBlock

class Phase1(nn.Module):
    def __init__(self, in_channels, features):
        super(Phase1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, features, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.rdb = ResidualDenseBlock(features, 6, 32)

        self.conv2 = nn.Conv2d(features, in_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x, target_size = x
        x1 = torchvision.transforms.Resize(
            size=target_size,
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            antialias=True
        )(x)

        x2 = self.relu1(self.conv1(x1))
        x2 = self.rdb(x2)
        x2 = self.conv2(x2)

        return x1 + x2

class HFN_Default(nn.Module):
    def __init__(self, lr_bands, hr_bands):
        super(HFN_Default, self).__init__()

        self.p1 = Phase1(
            in_channels=lr_bands,
            features=128
        )
        self.p2 = PanSharpening(
            channels=lr_bands + hr_bands,
            lr_channels=lr_bands,
            features=128,
            reduction=16,
            n_layers=6,
            growth_rate=32,
            n_rdb=6,
            kernel_size=3
        )
    
    def forward(self, x):
        hr, lr = x
        sr = self.p1([lr, hr.shape[-2:]])
        out = self.p2([sr, hr])
        return out

class HFN_SubPixel(nn.Module):
    def __init__(self, lr_res, hr_res, lr_bands, hr_bands):
        super(HFN_SubPixel, self).__init__()
        
        scaling_factor = lr_res // hr_res
        
        self.p1 = SubPixel(
            in_channels=lr_bands,
            channels=64,
            upscale_factor=scaling_factor
        )
        
        self.p2 = PanSharpening(
            channels=lr_bands + hr_bands,
            lr_channels=lr_bands,
            features=128,
            reduction=16,
            n_layers=6,
            growth_rate=32,
            n_rdb=6,
            kernel_size=3
        )

    def forward(self, x):
        hr, lr = x
        sr = self.p1(lr)
        out = self.p2([sr, hr])
        return out