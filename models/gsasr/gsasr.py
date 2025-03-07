import torch
import torch.nn as nn
import math

from modules import Encoder, ConditionInjectionBlock, GaussianInteractionBlock, GaussianPrimaryHead

# How does the MLP for scaling work?????

class GSASR(nn.Module):
    def __init__(
            self,
            backbone, # Encoder
            out_features, # Encoder & Condition Injection Block & Gaussian Interaction Block
            window_size, num_heads, # Condition Injection Block & Gaussian Interaction Block
            n_gaussian_interaction_blocks, # Gaussian Interaction Block
            m=16,
            mlp_ratio=4.,
    ):
        super().__init__()
        self.m = m
        self.window_size = window_size
        self.out_features = out_features

        # Embedding
        self.embedding = nn.Parameter(torch.randn(1, m * window_size[0] * window_size[1], out_features)) # Shape must match B x N x C. C is the number of features in the feature map. N is the wH*wW. B is the batch size combined with the number of images.

        self.encoder = Encoder(backbone, out_features)
        self.condition_injection_block = ConditionInjectionBlock(out_features, window_size, num_heads, m)
        mlp_hidden_dim = int(out_features * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, out_features),
        )
        gaussian_interaction_block = [
            GaussianInteractionBlock(out_features, window_size, num_heads)
            for _ in range(n_gaussian_interaction_blocks)
        ]
        self.gaussian_interaction_block = nn.ModuleList(gaussian_interaction_block)
        # self.gaussian_primary_head = GaussianPrimaryHead(out_features, mlp_ratio)

    def forward(self, x, scaling_factor):
        B, C, H, W = x.shape
        out = self.encoder(x).permute(0, 2, 3, 1).contiguous() # (B x C x H x W) -> (B x H x W x C)

        # Get Reference position of each embed
        i = torch.linspace(0, H, steps=H//self.window_size[0])
        j = torch.linspace(0, W, steps=W//self.window_size[1])
        ref_pos = torch.stack(torch.meshgrid(i, j, indexing='ij'), dim=-1).view(-1, 2).unsqueeze(0) # (1 x num_windows x 2)
        
        H, W = int(math.log2(self.m)) * H, int(math.log2(self.m)) * W
        out = self.condition_injection_block(self.embedding, out).view(B, H*W, self.out_features)
        
        mlp_out = self.mlp(scaling_factor).unsqueeze(1)
        for block in self.gaussian_interaction_block:
            out = block(out, mlp_out, H, W)
        
        # out = self.gaussian_primary_head(out, ref_pos)
        return out

if __name__ == '__main__':
    from models.backbones import EDSR
    backbone = EDSR(12, 16, 64)
    model = GSASR(backbone, 64, [4, 4], 4, 10)
    t = torch.randn(8, 12, 64, 64)
    scaling_factor = torch.randn(8, 1)
    out = model(t, scaling_factor)
    print('Success!')
    print(f'Out shape: {out.shape}')
