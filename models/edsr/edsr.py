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

    def load_state_dict(self, state_dict, strict = True, assign = False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

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
    import os
    print(os.path.join(os.path.dirname(__file__), 'weights', 'EDSR_x2.pt'))
    model = EDSR(3, 16, 64)
    state_dict = torch.load(os.path.join(os.path.dirname(__file__), 'weights', 'edsr_baseline_x2.pt'))
    model.load_state_dict(state_dict, strict=False)
    model.requires_grad_(requires_grad=False)
    # out = model(t)
    print('Success!')
    # print(f'Output shape: {out.shape}')
