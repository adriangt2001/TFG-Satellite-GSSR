# This is the Condition Injection Block module from the GSASR model.
# Key sources for this module:
# - Swin Transformer: https://arxiv.org/abs/2103.14030
# - Swin Transformer PyTorch implementation: https://github.com/microsoft/Swin-Transformer

# Start implementing once the sources are fully understood.
# Import libraries
import torch
import torch.nn as nn

# Main module
class ConditionInjectionBlock(nn.Module):
    def __init__(self, dim, window_size, num_heads, num_features, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.window_size = window_size

        # Embedding
        self.embedding = nn.Parameter(torch.randn(1, window_size[0] * window_size[1], num_features)) # Shape must match B x N x C. C is the number of features in the feature map. N is the wH*wW. B is the batch size combined with the number of images.

        # Window Cross Attention
        self.window_cross_attention = WindowCrossAttention(
            dim, window_size, num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=proj_drop
        )

    def window_partition(self, x):
        """
        Args:
            x: An tensor of shape (B x C x H x W)
        Returns:
            windows: The previous tensor partitioned in H/wH x W/wW windows. Shape: (B x wH x wW x C)
        """
        B, C, H, W = x.shape # Input shape
        windows = x.permute(0, 2, 3, 1).contiguous()
        windows = x.view(B, H // self.window_size[0], self.window_size[0], W // self.window_size[1], self.window_size[1], C)
        windows = windows.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, self.window_size[0], self.window_size[1], C)
        return windows

    def forward(self, x):
        """
        Args:
            x: The feature map from the LR image. Shape: (B x C x H x W)
        Returns:
            out: Currently unknown
        """
        # Input is the feature map from the LR image. Shape: (B x C x H x W)
        B, C, H, W = x.shape # Input shape.

        # Divide the input in windows and reshape to (B*NumWindows x N x C), where N is wH*wW
        out = self.window_partition(x) # (B * num_windows x wH x wW x C)
        out = out.view(-1, self.window_size[0] * self.window_size[1], C) # (B * num_windows x N x C), where N is wH*wW

        # Do Window Cross Attention
        out = self.window_cross_attention(out, self.embedding) # (B * num_windows x N x C)

        # Here the original paper stacks the output, but they're already a single tensor.
        return out

# Secondary modules
class WindowCrossAttention(nn.Module):
    """
    Window Based Cross Attention from Swin Transformer.
    """

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
        coords = torch.stack(torch.meshgrid([coords_h, coords_w])) # 2 x wH x wW
        coords_flatten = torch.flatten(coords, 1) # 2 x wH*wW
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :] # 2 x wH*wW x wH*wW, because PyTorch handles automatic broadcasting
        relative_coords = relative_coords.permute(1, 2, 0).contiguous() # wH*wW x wH*wW x 2
        relative_coords[:, :, 0] += self.window_size[0] - 1 # Shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1) # wH*wW x wH*wW
        self.register_buffer('relative_position_index', relative_position_index)

        # Since q and k, v come from different sources, just one Linear layer to extract k and v.
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)

        # Dropout layer after attention
        self.attn_drop = nn.Dropout(attn_drop)

        # Final linear layer to project the output. Project where? I don't know yet
        self.proj = nn.Linear(dim, dim)

        # Dropout layer after projection
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, e):
        """
        Args:
            x: The windowed feature map from the LR image. Shape: (B*NumWindows x C x H x W)
            q: The query extracted from the learnable embeddings E. Shape: (Unknown)
        Returns:
            out: Currently unknown
        """
        # Input is windowed feature map. Shape: (B*NumWindows x N x C), where N is wH*wW and C is the feature dimension.
        B, N, C = x.shape # Input shape. N is wH*wW and B is a combination of batch size and number of windows.

        # Get q, k, v from the embeds e and the input x
        q = self.q(e.expand(B, -1, -1)).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B x num_heads x N x C // num_heads
        kv = self.kv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 2 x B x num_heads x N x C // num_heads
        k, v = kv[0], kv[1] # B x num_heads x N x C // num_heads

        # Get the positional bias. Could this be moved to the __init__ method?
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1 # wH*wW x wH*wW x num_heads
        )
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

# Test section
if __name__ == '__main__':
    module = ConditionInjectionBlock(96, (4, 4), 3, 96)
    image = torch.randn(8, 96, 64, 164)
    output = module(image)
    print('Success!')
    print(f'Output shape: {output.shape}')
