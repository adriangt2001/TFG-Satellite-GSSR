import torch.nn as nn

class SubPixel(nn.Module):
    def __init__(self, in_channels, channels, upscale_factor):
        super(SubPixel, self).__init__()

        hidden_channels = channels // 2
        out_channels = int(in_channels * (upscale_factor ** 2))

        # Feature mapping
        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=5, padding=2)
        self.tanh1 = nn.Tanh()
        self.conv2 = nn.Conv2d(channels, hidden_channels, kernel_size=3, padding=1)
        self.tanh2 = nn.Tanh()

        # Sub-Pixel Convolution
        self.conv3 = nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)
    
    def forward(self, inputs):
        out = self.tanh1(self.conv1(inputs))
        out = self.tanh2(self.conv2(out))
        out = self.pixel_shuffle(self.conv3(out))
        return out
