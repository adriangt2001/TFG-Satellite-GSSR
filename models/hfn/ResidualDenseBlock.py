import torch
import torch.nn as nn

class DenseBlock(nn.Module):
    """
    Most code from: https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch/blob/master/model/model.py
    """
    def __init__(self, channels, growth_rate, kernel_size):
        super(DenseBlock, self).__init__()
        self.conv = nn.Conv2d(channels, growth_rate,
                              kernel_size, padding=kernel_size//2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x1 = self.relu(self.conv(x))
        x1 = torch.cat((x, x1), 1)
        return x1

class ResidualDenseBlock(nn.Module):
    """
    Most code from: https://github.com/lizhengwei1992/ResidualDenseNetwork-Pytorch/blob/master/model/model.py
    """
    def __init__(self, features, n_layers, growth_rate):
        super(ResidualDenseBlock, self).__init__()

        features_ = features
        modules = []
        for _ in range(n_layers):
            modules.append(DenseBlock(features_, growth_rate, kernel_size=3))
            features_ += growth_rate
        self.dense_layers = nn.Sequential(*modules)
        self.conv = nn.Conv2d(in_channels=features_, out_channels=features, 
                              kernel_size=1, padding=0)
    
    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv(out)
        out = out + x
        return out
