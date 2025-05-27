import os
import pandas as pd
import argparse
import yaml
import cv2

import torch.utils.tensorboard
from tqdm import tqdm

import torch
import torchvision

from DISTS_pytorch import DISTS
import lpips

from data import DIV2K
from models import GSASR, EDSR
from metrics import MetricsList, PSNR, CustomSSIM, CustomDists, CustomLPIPS

# Warning suppression
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum")
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")

import logging
logging.getLogger("lpips").setLevel(logging.ERROR)

def check_args(args):
    # Dataset arguments
    if not args.dataset: raise ValueError("dataset is required")

    # Backbone arguments
    if args.resblocks <= 0: raise ValueError("resblocks should be greater than 0")
    if args.backbone_features <= 0: raise ValueError("backbone_features should be greater than 0")

    # Model arguments
    if args.window_size <= 0: raise ValueError("window_size should be greater than 0")
    if args.num_heads <= 0: raise ValueError("num_heads should be greater than 0")
    if args.gaussian_interaction_blocks <= 0: raise ValueError("n_gaussian_interaction_blocks should be greater than 0")
    if args.channels <= 0: raise ValueError("num_channels should be greater than 0")
    if args.raster_ratio <= 0: raise ValueError("raster_ratio should be greater than 0")
    if args.gaussian_density <= 0: raise ValueError("m should be greater than 0")
    if args.mlp_ratio <= 0: raise ValueError("mlp_ratio should be greater than 0")

    # Test arguments
    if not args.scales: raise ValueError("scale is required")
    for scale in args.scales:
        if scale <= 1: raise ValueError("scale must be greater than 1")
    if args.batch_size <= 0: raise ValueError("batch_size should be greater than 0")

def parse_yaml(config):
    if not config: return None

    yaml_args = {}
    with open(config, 'r') as f:
        contents: dict = yaml.safe_load(f)
        for _, params in contents.items():
            if params:
                for key, value in params.items():
                    yaml_args[key] = value
    
    return yaml_args

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None, help="YAML configuration file with argument values.")

    # Dataset arguments
    parser.add_argument('--dataset', type=str, help="Path to the dataset.")

    # Backbone
    parser.add_argument('--resblocks', type=int, default=16, help="Number of Residual Blocks in the backbone.")
    parser.add_argument('--backbone_features', type=int, default=64, help="Embedding size of the backbone.")

    # Model arguments
    parser.add_argument('--weights', type=str)
    parser.add_argument('--channels', type=int, default=3, help="Number of channels of input images.")
    parser.add_argument('--model_features', type=int, default=180, help="Embedding size of the model.")
    parser.add_argument('--window_size', type=int, default=12, help="Window size.")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads.")
    parser.add_argument('--gaussian_interaction_blocks', type=int, default=6, help="Number of Gaussian Interaction Blocks.")
    parser.add_argument('--raster_ratio', type=float, default=0.1, help="Raster ratio of the Gaussian Rasterizer.")
    parser.add_argument('--gaussian_density', type=int, default=16, help="Density of gaussians.")
    parser.add_argument('--mlp_ratio', type=float, default=4., help="Decreasing ratio of MLPs.")

    # Test arguments
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size per iteration.")
    parser.add_argument('--scales', type=int, nargs='+', help="Scales to test the model")
    parser.add_argument('--log', type=str, default='logs', help="Root directory to store logs.")
    parser.add_argument('--results', type=str, default='results/metrics_patches_run4.csv', help="Relative path to file directory to store results.")

    tmp_args = parser.parse_args()

    # YAML file
    yaml_args = parse_yaml(tmp_args.config)

    if yaml_args:
        parser.set_defaults(**yaml_args)

    args = parser.parse_args()

    check_args(args)
    return args

def pad_image_divisible(image, divisible):
    _, _, H, W = image.shape
    padding_h = divisible - H % divisible
    padding_w = divisible - W % divisible
    return torch.nn.functional.pad(image, (0, padding_w, 0, padding_h))

def window_image(image, window_size):
    _, C, H, W = image.shape
    window = image.view(-1, C, H // window_size, window_size, W // window_size, window_size).permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return window

def reverse_window_image(window, H, W):
    _, C, window_size, _ = window.shape
    image = window.view(-1, H // window_size, W // window_size, C, window_size, window_size).permute(0, 3, 1, 4, 2, 5).contiguous().view(-1, C, H, W)
    return image

def test(model, dataloader,  scale, metrics, writer, device):
    model.eval()
    metrics.reset()
    
    for gt in tqdm(dataloader, desc=f"Testing x{scale} scaling", unit="batches", ascii=True):
        gt = gt.to(device='cuda')
        padded_gt = pad_image_divisible(gt, 48 * scale)
        _, _, H, W = padded_gt.shape
        patched_gt = window_image(padded_gt, 48 * scale)
        padded_lr = torch.nn.functional.interpolate(padded_gt, scale_factor=1/scale, mode='bicubic', antialias=True)
        patched_lr = window_image(padded_lr, 48)

        patched_output = torch.zeros_like(patched_gt)
        for idx, lr_patch in enumerate(patched_lr):
            patched_output[idx] = model(lr_patch.unsqueeze(0), scale)
            metrics(patched_output[idx].unsqueeze(0), patched_gt[idx].unsqueeze(0))
    
    writer.add_images('Images/gt', torchvision.utils.make_grid(patched_gt[:4], nrow=2).unsqueeze(0), scale)
    writer.add_images('Images/input', torchvision.utils.make_grid(patched_lr[:4], nrow=2).unsqueeze(0), scale)
    writer.add_images('Images/output', torchvision.utils.make_grid(patched_output[:4], nrow=2).unsqueeze(0), scale)

    cv2.imwrite(f'imgs_patches_run4/lr_x{scale}.png', cv2.cvtColor(patched_lr[0].cpu().permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))
    cv2.imwrite(f'imgs_patches_run4/output_x{scale}.png', cv2.cvtColor(patched_output[0].cpu().permute(1, 2, 0).numpy(), cv2.COLOR_RGB2BGR))

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Cant't train")
        return
    
    device='cuda'

    args = parse_args()

    # Load model
    backbone = EDSR(args.channels, args.resblocks, args.backbone_features)
    model = GSASR(backbone, args.model_features, args.window_size, args.num_heads, args.gaussian_interaction_blocks, args.channels,
                  raster_ratio=args.raster_ratio, m=args.gaussian_density, mlp_ratio=args.mlp_ratio).to(device=device)
    model.load_state_dict(torch.load(args.weights)['model'])

    # Load data
    test_dataset = DIV2K(args.dataset, phase='test')
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4)

    # Metrics
    my_dists = DISTS().to(device=device)
    my_dists.eval()
    my_lpips = lpips.LPIPS(net='alex').to(device=device)
    my_lpips.eval()
    metrics = MetricsList(PSNR(data_range=1.), CustomSSIM(data_range=1.), CustomDists(my_dists), CustomLPIPS(my_lpips))
    metrics_dict = {metric.name: [] for metric in metrics}
    metrics_dict['scale'] = []

    # Logs
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=os.path.join(args.log, 'test_results_patches_run4'))

    # Test
    for scale in args.scales:
        test(model, test_dataloader, scale, metrics, writer, device)
        metrics_dict['scale'].append(scale)
        for metric in metrics:
            metrics_dict[metric.name].append(metric.get_value())
            writer.add_scalar(metric.name, metric.get_value(), scale)
    
    # Save to .csv
    pd.DataFrame(metrics_dict).to_csv(args.results)

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    with torch.no_grad():
        main()
