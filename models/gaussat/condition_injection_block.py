import torch
import torch.nn as nn

class ConditionInjectionBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads, m):
        super().__init__()
        self.window_size = window_size

        self.window_cross_attention = WindowCrossAttention(dim, window_size, num_heads, m)
    
    def window_partition(self, image_features):
        B, H, W, C = image_features.shape
        windows = image_features.view(B, H // self.window_size, self.window_size, W // self.window_size, self.window_size, C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size * self.window_size, C)
        return windows

    def forward(self, gauss_embeds, image_features):
        B1, N1, C1 = gauss_embeds.shape
        B2, H2, W2, C2 = image_features.shape
        
        # Prepare embeds        
        gauss_embeds = gauss_embeds.view(B1 * H2 // self.window_size * W2 // self.window_size, -1, C1)

        # Prepare features
        windows = self.window_partition(image_features)

        out_embeds = self.window_cross_attention(gauss_embeds, windows)

        out_embeds = out_embeds.view(B1, -1, C1)

        return out_embeds

class WindowCrossAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, m):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.m = m
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))

        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'), dim=-1)
        coords_flatten = coords.view(-1, 2)
        relative_coords = coords_flatten.unsqueeze(1) - coords_flatten.unsqueeze(0)
        relative_coords[..., 0] += self.window_size - 1
        relative_coords[..., 1] += self.window_size - 1
        relative_coords[..., 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, gauss_embeds, window_features, mask=None):
        # Preparation
        B1, N1, C1 = gauss_embeds.shape
        B2, N2, C2 = window_features.shape

        q = self.q(gauss_embeds).view(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3).contiguous()
        kv = self.kv(window_features).view(B2, N2, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size * self.window_size, self.window_size * self.window_size, -1
        ).permute(2, 0, 1).repeat(1, self.m, 1).contiguous()

        # Attention
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = attn @ v
        attn = attn.transpose(1, 2).contiguous().view(B1, N1, C1)

        attn = self.proj(attn)
        attn = self.proj_drop(attn)

        return attn

if __name__ == '__main__':
    num_features = 64
    window_size = 12
    num_heads = 4
    m = 16
    module = ConditionInjectionBlock(num_features, window_size, num_heads, m)
    embeds = torch.randn(8 * (48 // 12) * (48 // 12), m * window_size * window_size, num_features)
    image = torch.randn(8, 48, 48, num_features)
    output = module(embeds, image)
    print('Success!')
    print(f'Output shape: {output.shape}')



