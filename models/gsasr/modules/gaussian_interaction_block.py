import torch
import torch.nn as nn

# This module is a CrossAttention Module followed by two successive Swin Transformer Blocks
class GaussianInteractionBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.cross_attention = CrossAttention(dim, num_heads)
        self.swin_block1 = SwinBlock(dim, window_size, num_heads)
        self.swin_block2 = SwinBlock(dim, window_size, num_heads, shift=True)

    def forward(self, x, mlp_out, H, W):
        out = self.cross_attention(x, mlp_out)
        out = out + x

        out = self.swin_block1(out, H, W)
        out = self.swin_block2(out, H, W)
        return out

class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, mlp_out):
        B1, N1, C1 = x.shape
        B2, N2, C2 = mlp_out.shape

        q = self.q(x).view(B1, N1, self.num_heads, C1 // self.num_heads).permute(0, 2, 1, 3).contiguous() # B x num_heads x N x C // num_heads
        k = self.k(mlp_out).view(B2, N2, self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3).contiguous() # B x num_heads x N x C // num_heads
        v = self.v(mlp_out).view(B2, N2, self.num_heads, C2 // self.num_heads).permute(0, 2, 1, 3).contiguous() # B x num_heads x N x C // num_heads

        # Attention(Q, K, V) = SoftMax((Q @ K^T)/d**0.5 + Bias) @ V
        attn = q * self.scale # Q / d**0.5
        attn = q @ k.transpose(-2, -1) # @ K^T
        attn = self.softmax(attn) # SoftMax(...)
        attn = self.attn_drop(attn)
        attn = attn @ v # @ V
        attn = attn.transpose(1, 2).contiguous().view(B1, N1, C1) # B x N x C
        attn = self.proj(attn) # B x N x C
        attn = self.proj_drop(attn)

        return attn

class SwinBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads, shift=False, mlp_ratio=4., qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size
        self.shift_size = (window_size[0] // 2, window_size[1] // 2) if shift else None

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
        windows = x.view(B, H // self.window_size[0], self.window_size[0], W // self.window_size[1], self.window_size[1], C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size[0] * self.window_size[1], C)
        return windows

    def window_reverse(self, x: torch.Tensor, H, W):
        B = int(x.shape[0] / (H * W / self.window_size[0] / self.window_size[1]))
        x = x.view(B, H // self.window_size[0], W // self.window_size[1], self.window_size[0], self.window_size[1], -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x, H, W):
        B, N, C = x.shape

        # PART 1
        x1 = x
        x1 = self.norm1(x1)

        x1 = x1.view(B, H, W, C)
        if self.shift_size:
            x1 = torch.roll(x1, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
        x1 = self.window_partition(x1)

        x1 = self.window_attention(x1).view(-1, self.window_size[0], self.window_size[1], C)

        x1 = self.window_reverse(x1, H, W)
        if self.shift_size:
            x1 = torch.roll(x, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
        x1 = x1.view(B, H*W, C)

        x1 = x1 + x

        # PART 2
        x2 = self.norm2(x1)
        x2 = self.mlp(x2)
        # x2 = self.drop_path(x2)
        out = x2 + x1

        return out

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()

        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # This initialization can be this random matrix or just 0s.
        self.relative_position_bias_table = nn.Parameter(torch.randn((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Relative position index, calculated here and stored as a non trainable parameter using register_buffer
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij')) # 2 x wH x wW.
        coords_flatten = torch.flatten(coords, 1) # 2 x wH*wW
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2 x wH*wW x wH*wW, because PyTorch handles automatic broadcasting
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # wH*wW x wH*wW x 2
        relative_coords[:, :, 0] += self.window_size[0] - 1 # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) # wH*wW x wH*wW
        self.register_buffer('relative_position_index', relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor):
        B, N, C = x.shape 

        # Get q, k, v from the input x
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 2 x B x num_heads x N x C // num_heads
        q, k, v = qkv[0], qkv[1], qkv[2] # B x num_heads x N x C // num_heads

        # Get the positional bias. Could this be moved to the __init__ method?
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1 # wH*wW x wH*wW x num_heads
        ).to(device=x.device)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous() # num_heads x wH*wW x wH*wW

        # Attention(Q, K, V) = SoftMax((Q @ K^T)/d**0.5 + Bias) @ V
        q = q * self.scale # Q / d**0.5. Shape: B x num_heads x N x C // num_heads
        attn = (q @ k.transpose(-2, -1)) # @ K^T. Shape: B x num_heads x N x N
        attn = attn + relative_position_bias.unsqueeze(0) # + Bias. Shape: B x num_heads x N x N

        attn = self.softmax(attn) # SoftMax(...). Shape: B x num_heads x N x N
        attn = self.attn_drop(attn)
        attn = attn @ v # @ V. Shape: B x num_heads x N x C // num_heads
        attn = attn.transpose(1, 2).reshape(B, N, C) # B x N x C

        out = self.proj(attn) # B x N x C
        out = self.proj_drop(out)
        return out

if __name__ == '__main__':
    model = GaussianInteractionBlock(96, (4, 4), 4)
    t = torch.randn(8, 64*64, 96)
    
    mlp_out = torch.randn(8, 64*64, 96)
    out = model(t, mlp_out, 64, 64)
    print('Success!')
    print(f'Out shape: {out.shape}')
