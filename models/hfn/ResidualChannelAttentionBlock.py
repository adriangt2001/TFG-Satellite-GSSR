import torch.nn as nn

class ResidualChannelAttentionBlock(nn.Module):
    """
    Architecture from: https://www.sciencedirect.com/science/article/abs/pii/S0924271622003331
    """
    def __init__(self, features, reduction, kernel_size):
        super(ResidualChannelAttentionBlock, self).__init__()

        self.conv1 = nn.Conv2d(features, features, kernel_size, padding=kernel_size//2)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(features, features, kernel_size, padding=kernel_size//2)
        
        self.pooling = nn.AdaptiveAvgPool2d(1) # The parameter indicates the size of the output tensor
        self.conv3 = nn.Conv2d(features, features // reduction, 1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(features // reduction, features, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Feature Extraction
        x1 = self.relu1(self.conv1(x))
        x1 = self.conv2(x1)

        # Channel Attention Weights
        x2 = self.pooling(x1)
        x2 = self.relu3(self.conv3(x2))
        x2 = self.sigmoid(self.conv4(x2))

        # Apply Attention Weights
        out = x + (x1 * x2)
        return out