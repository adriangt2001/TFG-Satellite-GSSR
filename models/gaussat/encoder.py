import torch
import torch.nn as nn

# So the idea is as follows:
# - Apply ESDR head+body to the LowResolution image (Conv2d + N Resblocks)
# - Apply a Conv2d layer to the output of the EDSR body
# - Feature map ready for the rest of the network

class Encoder(nn.Module):
    def __init__(self, backbone, out_features):
        super().__init__()
        kernel_size = 3

        self.backbone = backbone
        self.feature_adapter = nn.Conv2d(backbone.out_features, out_features, kernel_size, padding=(kernel_size//2))

    def forward(self, x):
        out = self.backbone(x)
        out = self.feature_adapter(out)
        return out

if __name__ == '__main__':
    from models.backbones import EDSR
    backbone = EDSR(12, 16, 64)
    encoder = Encoder(backbone, 128)
    t = torch.randn(8, 12, 256, 256)
    out = encoder(t)
    print('Success!')
    print(f'Output shape: {out.shape}')

