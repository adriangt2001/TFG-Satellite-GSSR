import torch.nn as nn

# So the idea is as follows:
# - Apply ESDR head+body to the LowResolution image (Conv2d + N Resblocks)
# - Apply a Conv2d layer to the output of the EDSR body
# - Feature map ready for the rest of the network

class Encoder(nn.Module):
    def __init__(self, backbone, out_features):
        super().__init__()
        self.backbone = backbone
        self.feature_adapter = nn.Conv2d(backbone.out_dim, out_features, kernel_size=3)

    def forward(self, x):
        out = self.backbone(x)
        out = self.feature_adapter(out)
        # Check if I need to reshape or permute the output
        return out

from models.backbones.edsr import EDSR

if __name__ == '__main__':
    backbone = EDSR()
