import torch
import torch.nn as nn

class ConditionInjectionBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads, m):
        super().__init__()
        self.window_size = window_size

        self.window_cross_attention = WindowCrossAttention(dim, window_size, num_heads, m)
    
    def window_partition(self, x):
        B, H, W, C = x.shape
        windows = x.view(B, H // self.window_size[0], self.window_size[0], W // self.window_size[1], self.window_size[1], C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size[0] * self.window_size[1], C)
        return windows
    
    def window_reverse(self, x, H, W, C):
        x = x.view(-1, H // self.window_size[0], W // self.window_size[1], self.window_size[0], self.window_size[1], C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, H*W, C)
        return x

    def forward(self, embeds, x):
        B1, N1, C1 = embeds.shape
        B2, H2, W2, C2 = x.shape

        windows = self.window_partition(x)
        B_ = windows.shape[0]
        embeds = embeds.expand(B_, -1, -1)

        out = self.window_cross_attention(embeds, windows)

        return out

class WindowCrossAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, m):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.m = m
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'), dim=-1)
        coords_flatten = coords.view(-1, 2)
        relative_coords = coords_flatten.unsqueeze(1) - coords_flatten.unsqueeze(0)
        relative_coords[..., 0] += self.window_size[0] - 1
        relative_coords[..., 1] += self.window_size[1] - 1
        relative_coords[..., 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer('relative_position_index', relative_position_index)

        self.kv = nn.Linear(dim, dim * 2)
        
        self.attn_drop = nn.Dropout(0.)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(0.)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, e, x):
        B1, N1, C1 = e.shape
        B2, N2, C2 = x.shape

        q = e.view(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3).contiguous()
        kv = self.kv(x).view(B2, N2, 2, self.num_heads, C2 // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        k, v = kv[0], kv[1]

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1
        ).permute(2, 0, 1).repeat(1, self.m, 1).contiguous()

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        attn = attn + relative_position_bias.unsqueeze(0)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        attn = attn @ v
        attn = attn.transpose(1, 2).contiguous().view(B1, N1, C1)

        attn = self.proj(attn)
        attn = self.proj_drop(attn)
        
        return attn

if __name__ == '__main__':
    num_features = 64
    window_size = (4, 4)
    num_heads = 4
    m = 16
    module = ConditionInjectionBlock(num_features, window_size, num_heads, m)
    embeds = torch.randn(1, m*window_size[0] * window_size[1], num_features)
    image = torch.randn(8, 256, 256, num_features)
    output = module(embeds, image)
    print('Success!')
    print(f'Output shape: {output.shape}')
