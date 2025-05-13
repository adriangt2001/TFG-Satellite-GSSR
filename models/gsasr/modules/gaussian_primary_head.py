import torch
import torch.nn as nn

class GaussianPrimaryHead(nn.Module):
    def __init__(self, in_features, num_colors):
        super().__init__()
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

        mlp_layer1 = int(2*in_features/3)
        mlp_layer2 = int(in_features/3)

        feat_opacity = 1
        feat_color = num_colors
        feat_offs_std = 2
        feat_corr = 1

        self.mlp_opacity = nn.Sequential(
            nn.Linear(in_features, mlp_layer1),
            self.act_fn,
            nn.Linear(mlp_layer1, mlp_layer2),
            self.act_fn,
            nn.Linear(mlp_layer2, feat_opacity),
            self.sigmoid
        )
        self.mlp_color = nn.Sequential(
            nn.Linear(in_features, mlp_layer1),
            self.act_fn,
            nn.Linear(mlp_layer1, mlp_layer2),
            self.act_fn,
            nn.Linear(mlp_layer2, feat_color),
            self.sigmoid
        )
        self.mlp_std = nn.Sequential(
            nn.Linear(in_features, mlp_layer1),
            self.act_fn,
            nn.Linear(mlp_layer1, mlp_layer2),
            self.act_fn,
            nn.Linear(mlp_layer2, feat_offs_std),
            # self.sigmoid
        )
        self.mlp_offset = nn.Sequential(
            nn.Linear(in_features, mlp_layer1),
            self.act_fn,
            nn.Linear(mlp_layer1, mlp_layer2),
            self.act_fn,
            nn.Linear(mlp_layer2, feat_offs_std),
        )
        self.mlp_corr = nn.Sequential(
            nn.Linear(in_features, mlp_layer1),
            self.act_fn,
            nn.Linear(mlp_layer1, mlp_layer2),
            self.act_fn,
            nn.Linear(mlp_layer2, feat_corr),
            self.tanh,
        )

    def forward(self, x, ref_pos):
        opacity = self.mlp_opacity(x)
        color = self.mlp_color(x)
        std = self.mlp_std(x)
        offset = self.mlp_offset(x)
        corr = self.mlp_corr(x)
        
        # Apply safety mechanisms to prevent edge cases
        std = std + 1e-5  # Add small epsilon to prevent zeros
        corr = corr * 0.99  # Scale to prevent exact -1 or 1 values

        position = offset + ref_pos
        return opacity, color, std, position, corr

if __name__ == '__main__':
    model = GaussianPrimaryHead()
    t = torch.randn(8, 64*64, 96)
    out = model(t)
    print('Success!')
    print(f'Out shape: {out.shape}')