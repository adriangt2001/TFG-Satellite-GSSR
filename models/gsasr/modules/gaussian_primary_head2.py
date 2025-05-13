import torch
import torch.nn as nn

class GaussianPrimaryHead(nn.Module):
    def __init__(self, in_features, num_colors, window_size):
        super().__init__()
        self.window_size = window_size
        self.act_fn = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softplus = nn.Softplus()
        self.tanh = nn.Tanh()

        mlp_layer1 = int(in_features * 2/3)
        mlp_layer2 = int(in_features * 1/3)

        feat_opacity_rho = 1
        feat_offs_std = 2
        feat_color = num_colors

        self.mlp_opacity = nn.Sequential(
            nn.Linear(in_features, mlp_layer1),
            self.act_fn,
            nn.Linear(mlp_layer1, mlp_layer2),
            self.act_fn,
            nn.Linear(mlp_layer2, feat_opacity_rho),
            self.sigmoid
        )
        self.mlp_rho = nn.Sequential(
            nn.Linear(in_features, mlp_layer1),
            self.act_fn,
            nn.Linear(mlp_layer1, mlp_layer2),
            self.act_fn,
            nn.Linear(mlp_layer2, feat_opacity_rho),
            self.tanh,
        )
        self.mlp_offset = nn.Sequential(
            nn.Linear(in_features, mlp_layer1),
            self.act_fn,
            nn.Linear(mlp_layer1, mlp_layer2),
            self.act_fn,
            nn.Linear(mlp_layer2, feat_offs_std),
        )
        self.mlp_std = nn.Sequential(
            nn.Linear(in_features, mlp_layer1),
            self.act_fn,
            nn.Linear(mlp_layer1, mlp_layer2),
            self.act_fn,
            nn.Linear(mlp_layer2, feat_offs_std),
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
    
    def forward(self, gauss_embeds, ref_pos):
        opacity = self.mlp_opacity(gauss_embeds)
        rho = self.mlp_rho(gauss_embeds)
        offset = self.mlp_offset(gauss_embeds)
        std = self.mlp_std(gauss_embeds) # Try upscaling the std by the image height or width
        color = self.mlp_color(gauss_embeds) #* self.window_size

        # Safety mechanisms to prevent edge cases
        std = std + 1e-5
        rho = rho * 0.9999

        mean = offset + ref_pos
        return opacity, rho, mean, std, color

if __name__ == '__main__':
    model = GaussianPrimaryHead(64, 3)

    x = torch.linspace(0, 48, steps = 48 * 4)
    y = torch.linspace(0, 48, steps = 48 * 4)

    ref_pos = torch.stack(torch.meshgrid(x, y, indexing='xy'), dim=-1).view(-1, 2).unsqueeze(0)

    t = torch.randn(8, 16*48*48, 64)
    opacity, rho, mean, std, color = model(t, ref_pos)
    print('Success!')
    print(f'Out shape: {color.shape}')
