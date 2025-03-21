import os

import argparse
import torch
import torch.nn as nn

from models.gsasr import GSASR
from models.backbones import EDSR

def parse_args():
    parser = argparse.ArgumentParser()
    # Training arguments
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--save_interval', type=int, default=1000)

    # Backbone arguments
    parser.add_argument('--num_resblocks', type=int, default=16)
    parser.add_argument('--num_features', type=int, default=64)

    # Model arguments
    parser.add_argument('--out_features', type=int, default=64)
    parser.add_argument('--window_size', type=int, nargs=2, default=[4, 4])
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--n_gaussian_interaction_blocks', type=int, default=4)
    parser.add_argument('--num_colors', type=int, default=10)
    parser.add_argument('--raster_ratio', type=int, default=4)
    parser.add_argument('--m', type=int, default=16)
    parser.add_argument('--mlp_ratio', type=float, default=4)


    args = parser.parse_args()
    return args

def train(args):
    pass

def main():
    args = parse_args
    
    backbone = EDSR(args.num_colors, args.num_resblocks, args.num_features)
    model = GSASR(
        backbone,
        args.out_features,
        args.window_size,
        args.num_heads,
        args.n_gaussian_interaction_blocks,
        args.num_colors,
        args.raster_ratio,
        args.m,
        args.mlp_ratio
    )

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5) # Don't know what this does but I'll use it c:

    train(args)

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    main()