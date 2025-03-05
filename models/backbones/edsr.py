# Adapted from the official implementation of ESDR and the adapted version of GaussianSR

import torch
import torch.nn as nn

class EDSR(nn.Module):
    def __init__(self, in_channels, n_resblocks, n_feats):
        super().__init__()
        self.out_features = n_feats
        kernel_size = 3

        # define head module
        self.head = nn.Conv2d(in_channels, n_feats, kernel_size, padding=(kernel_size//2))

        # define body module
        m_body = [
            ResBlock(
                n_feats, kernel_size
            ) for _ in range(n_resblocks)
        ]
        m_body.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2)))

        self.body = nn.Sequential(*m_body)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        return x

class ResBlock(nn.Module):
    def __init__(self, n_feats, kernel_size):
        super().__init__()
        act = nn.ReLU(True)
        m = []
        for i in range(2):
            m.append(nn.Conv2d(n_feats, n_feats, kernel_size, padding=(kernel_size//2), bias=True))
            if i == 0: m.append(act)
        
        self.body = nn.Sequential(*m)

    def forward(self, x):
        res = self.body(x)
        res += x

        return res

if __name__ == '__main__':
    model = EDSR(3, 16, 64)
    t = torch.randn(5, 3, 256, 256)
    out = model(t)
    print('Success!')
    print(f'Output shape: {out.shape}')
