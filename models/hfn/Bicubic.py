import torch.nn as nn
import torchvision

class Bicubic(nn.Module):
    """Dummy model to compare with HFN. Simply performs bilinear upsampling on the LR bands."""
    def __init__(self):
        super(Bicubic, self).__init__()
    
    def forward(self, x):
        hr, lr = x
        
        sr = torchvision.transforms.Resize(
            size=hr.shape[-2:],
            interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
            antialias=True
        )(lr)
        
        return sr