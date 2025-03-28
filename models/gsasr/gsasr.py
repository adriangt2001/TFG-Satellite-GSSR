import torch
import torch.nn as nn
import math

from modules import Encoder, ConditionInjectionBlock, GaussianInteractionBlock, GaussianPrimaryHead
from diff_srgaussian_rasterization import GaussianRasterizer

class GSASR(nn.Module):
    def __init__(
            self,
            backbone, # Encoder
            out_features, # Encoder & Condition Injection Block & Gaussian Interaction Block
            window_size, num_heads, # Condition Injection Block & Gaussian Interaction Block
            n_gaussian_interaction_blocks, # Gaussian Interaction Block
            num_colors, # Gaussian Primary Head
            raster_ratio=0.1, # Gaussian Rasterizer
            m=16,
            mlp_ratio=4.,
    ):
        super().__init__()
        self.m = m
        self.window_size = window_size
        self.out_features = out_features
        self.raster_ratio = raster_ratio

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
        self.gaussian_primary_head = GaussianPrimaryHead(out_features, num_colors)
        self.gaussian_rasterizer = GaussianRasterizer()

    def forward(self, x: torch.Tensor, scaling_factor):
        B, C, H, W = x.shape
        m_log = int(math.log2(self.m))
        H_gauss, W_gauss = m_log * H, m_log * W
        out = self.encoder(x).permute(0, 2, 3, 1).contiguous() # (B x C x H x W) -> (B x H x W x C)

        # Get Reference position of each embed
        i = torch.linspace(0, H, steps=H_gauss, device=x.device)
        j = torch.linspace(0, W, steps=W_gauss, device=x.device)
        ref_pos = torch.stack(torch.meshgrid(i, j, indexing='ij'), dim=-1).to(device=x.device).view(-1, 2).unsqueeze(0) # (1 x num_windows x 2)
        
        out = self.condition_injection_block(self.embedding, out).view(B, H_gauss*W_gauss, self.out_features)

        mlp_out = self.mlp(torch.tensor(scaling_factor, device=x.device).unsqueeze(0).expand(B, -1)).unsqueeze(1)
        for block in self.gaussian_interaction_block:
            out = block(out, mlp_out, H_gauss, W_gauss)
        
        opacity, color, std, position, corr = self.gaussian_primary_head(out, ref_pos)
        # Debug: check tensor shapes
        print(f"opacity shape: {opacity.shape}")
        print(f"color shape: {color.shape}")
        print(f"std shape: {std.shape}")
        print(f"position shape: {position.shape}")
        print(f"corr shape: {corr.shape}")
        
        out: torch.Tensor = self.gaussian_rasterizer(opacity, position, std, corr, color, H, W, scaling_factor, self.raster_ratio, debug=True)
        out = out.permute(0, 3, 1, 2)
        return out

if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    from models.backbones import EDSR
    device = 'cuda'
    batch_size = 2
    num_channels = 3
    backbone = EDSR(num_channels, 16, 180)
    model = GSASR(backbone, 180, [12, 12], 4, 6, num_channels)
    model.to(device=device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    # t = torch.randn(batch_size, num_channels, 48, 48, device=device)
    # scaling_factor = 2.
    # with torch.no_grad():
    #     model.eval()
    #     out = model(t, scaling_factor)
    #     t2 = torch.zeros_like(out)
    # print(f'Is everything a zero? {torch.all(t2 == out)}')
    # print(f"Min/Max values - output: {out.min().item():.4f}/{out.max().item():.4f}")
    # print('Success!')
    # print(f'Out shape: {out.shape}')
