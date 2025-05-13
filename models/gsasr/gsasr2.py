import torch
import torch.nn as nn
import math

from .modules import Encoder, ConditionInjectionBlock, GaussianInteractionBlock, GaussianPrimaryHead
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
        self.out_features = out_features
        self.window_size = window_size
        self.raster_ratio = raster_ratio
        self.m = m

        # Embedding
        self.embedding = nn.Parameter(torch.randn(1, self.m * self.window_size * self.window_size, self.out_features))

        # Encoder
        self.encoder = Encoder(backbone, self.out_features)

        # Condition Injection Block
        self.condition_injection_block = ConditionInjectionBlock(self.out_features, self.window_size, num_heads, self.m)

        # Gaussian Interaction Block (Stacks L times)
        mlp_hidden_dim = int(out_features * mlp_ratio)
        self.scale_mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, self.out_features)
        )
        self.gaussian_interaction_block = nn.ModuleList([
            GaussianInteractionBlock(self.out_features, self.window_size, num_heads)
            for _ in range(n_gaussian_interaction_blocks)
        ])

        # Gaussian Primary Head
        self.gaussian_primary_head = GaussianPrimaryHead(self.out_features, num_colors, self.window_size)

        # Gaussian Rasterizer
        self.gaussian_rasterizer = GaussianRasterizer()

    def forward(self, input_image: torch.Tensor, scaling_factor):
        B, C, H, W = input_image.shape
        
        # Extract features with the encoder
        lr_features = self.encoder(input_image).permute(0, 2, 3, 1).contiguous()
        
        # Prepare Gaussian Embeddings. Duplicate them and assign to each copy a reference position
        # Reference positions
        m_root = int(math.sqrt(self.m))
        rows = torch.linspace(0, H, steps=m_root * H, device=input_image.device)
        cols = torch.linspace(0, W, steps=m_root * W, device=input_image.device)
        ref_pos = torch.stack(torch.meshgrid(cols, rows, indexing='xy'), dim=-1).view(-1, 2).unsqueeze(0)

        # Duplicate Gaussians and reference position
        gauss_embeds = self.embedding.repeat(B, int(H//self.window_size * W//self.window_size), 1)
        ref_pos = ref_pos.repeat(B, 1, 1)

        # Condition Injection Block
        gauss_embeds = self.condition_injection_block(gauss_embeds, lr_features)

        # Gaussian Interaction Block
        scale_features = self.scale_mlp(
                torch.tensor(scaling_factor, device=rows.device, dtype=torch.float32).unsqueeze(0).repeat(B, 1)
            ).unsqueeze(1)

        for block in self.gaussian_interaction_block:
            gauss_embeds = block(gauss_embeds, scale_features, m_root * H, m_root * W)
        
        # Gaussian Primary Head
        opacity, rho, mean, std, color = self.gaussian_primary_head(gauss_embeds, ref_pos)

        # Gaussian Rasterizer
        output_image = self.gaussian_rasterizer(opacity, mean, std, rho, color, H, W, scaling_factor, self.raster_ratio, debug=False)
        output_image = output_image.permute(0, 3, 1, 2)
        return output_image
