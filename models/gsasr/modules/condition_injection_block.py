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
    def __init__(self):
        super().__init__()
        # Query
        # shape_query = unknown
        # self.query = nn.Parameter(torch.randn(shape_query))

        # Window Cross Attention
        self.window_cross_attention = WindowCrossAttention()
    
    def forward(self, x):
        """
        Args:
            x: The feature map from the LR image. Shape: (B x C x H x W)
        Returns:
            out: Currently unknown
        """
        # Input is the feature map from the LR image. Shape: (B x C x H x W)

        # Divide the input in windows and reshape to (B*NumWindows x N x C), where N is wH*wW

        # Do Window Cross Attention

        # The output (Gaussian Embeddings?) go into the next module (Gaussian Interaction Block)

        pass

# Secondary modules
class WindowCrossAttention(nn.Module):
    # Questions:
    # - What are heads and why do they use them in the Swin transformer? Do I need them?
    # - In transformers, input is divided in tokens. Does ESDR already do this?

    def __init__(self, dim, window_size, num_heads, kv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()

        # Relative position bias is a table with values for the attention bias depending on the position
        self.relative_position_bias = nn.Parameter(torch.randn((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # Relative position index, calculated here and stored as a non trainable parameter using register_buffer
        relative_position_index = torch.randn(1)
        self.register_buffer('relative_position_index', relative_position_index)

        # Since q and k, v come from different sources, just one Linear layer to extract k and v.
        self.kv = nn.Linear(dim, dim * 2, bias=kv_bias)

        # Dropout layer after attention
        self.attn_drop = nn.Dropout(attn_drop)

        # Final linear layer to project the output. Project where? I don't know yet
        self.proj = nn.Linear(dim, dim)

        # Dropout layer after projection
        self.proj_drop = nn.Dropout(proj_drop)

        pass

    def forward(self, x, q):
        """
        Args:
            x: The windowed feature map from the LR image. Shape: (B*NumWindows x C x H x W)
            q: The query extracted from the learnable embeddings E. Shape: (Unknown)
        Returns:
            out: Currently unknown
        """
        # Input is windowed feature map. Shape: (B*NumWindows x N x C), where N is wH*wW and C is the feature dimension.

        # Query comes in q

        # Key an value extracted from x

        # Get position bias from the relative position bias table

        # Calculate attention

        # Apply dropout

        # Apply projection	

        # Apply dropout

        # Return output

        pass

# Test section
if __name__ == '__main__':
    module = ConditionInjectionBlock()
    print('Success!')
