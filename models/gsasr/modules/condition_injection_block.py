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

        # Divide the input in windows. Shape: (B x C x H x W) -> (B*NumWindows x C x H x W)

        # It goes into the WindowCrossAttention module
        pass

# Secondary modules
class WindowCrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x, q):
        """
        Args:
            x: The windowed feature map from the LR image. Shape: (B*NumWindows x C x H x W)
            q: The query extracted from the learnable embeddings E. Shape: (Unknown)
        Returns:
            out: Currently unknown
        """
        # Input: is the windowed feature map from the LR image. Shape: (B*NumWindows x C x WindowHeight x WindowWidth)

        # Query comes in q

        # Key is extracted from x

        # Value is extracted from x

        # In Swin Transformer, the qkv is extracted using a Linear layer: nn.Linear(dim, dim * 3, bias=qkv_bias[True by default])

        # Perform cross attention as follows: Attention(Q, K, V) = SoftMax(QK^T/sqrt(d) + B)V
        # where, B is the relative position bias in the feature map

        pass

# Test section
if __name__ == '__main__':
    module = ConditionInjectionBlock()
    print('Success!')
