import torch
import torch.nn as nn

# This module is a CrossAttention Module followed by two successive Swin Transformer Blocks
class GaussianInteractionBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.cross_attention = CrossAttention(dim, num_heads)
        self.swin_block1 = SwinBlock(dim, window_size, num_heads)
        self.swin_block2 = SwinBlock(dim, window_size, num_heads, shift=True)
    
    def forward(self, gauss_embeds, scale_features, height, width):
        out_embeds = self.cross_attention(gauss_embeds, scale_features)
        out_embeds = out_embeds + gauss_embeds

        out_embeds = self.swin_block1(out_embeds, height, width)
        out_embeds = self.swin_block2(out_embeds, height, width)

        return out_embeds

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, gauss_embeds, scale_features):
        # Preparation
        B1, N1, C1 = gauss_embeds.shape
        B2, N2, C2 = scale_features.shape

        q = self.q(gauss_embeds).view(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3).contiguous()
        kv = self.kv(scale_features).view(B2, N2, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        # Attention
        attn = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = attn @ v
        attn = attn.transpose(1, 2).contiguous().view(B1, N1, C1)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)

        return attn

class SwinBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads, shift=False, mlp_ratio=4., qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = self.window_size // 2 if shift else None

        # Part 1
        self.norm1 = nn.LayerNorm(dim)
        self.window_attention = WindowAttention(dim, window_size, num_heads, qkv_bias, attn_drop, proj_drop)

        # Part 2
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
        )

    def window_partition(self, x: torch.Tensor):
        B, H, W, C = x.shape
        windows = x.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        return windows

    def window_reverse(self, x: torch.Tensor, H, W):
        B = int(x.shape[0] / (H * W / self.window_size / self.window_size))
        x = x.view(B, H // self.window_size, W // self.window_size, self.window_size, self.window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, gauss_embeds, H, W):
        B, N, C = gauss_embeds.shape

        # PART 1
        x1 = gauss_embeds
        x1 = self.norm1(x1)

        x1 = x1.view(B, H, W, C)
        if self.shift_size:
            x1 = torch.roll(x1, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        x1 = self.window_partition(x1)

        x1 = x1.view(-1, self.window_size * self.window_size, C)
        x1 = self.window_attention(x1)
        x1 = x1.view(-1, self.window_size, self.window_size, C)

        x1 = self.window_reverse(x1, H, W)
        if self.shift_size:
            x1 = torch.roll(x1, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        x1 = x1.view(B, H*W, C)

        x1 = x1 + gauss_embeds

        # PART 2
        x2 = self.norm2(x1)
        x2 = self.mlp(x2)
        # x2 = self.drop_path(x2)
        out_embeds = x2 + x1

        return out_embeds

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(torch.randn((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        # Preparation
        B, N, C = x.shape 

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        ).to(device=x.device)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()

        # Attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = attn @ v
        attn = attn.transpose(1, 2).reshape(B, N, C)

        out = self.proj(attn)
        out = self.proj_drop(out)
        return out