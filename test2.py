import os
import cv2
import yaml
import random
import argparse
import pandas as pd
from tqdm import tqdm

import rasterio
import rasterio.io

import torch
import torch.utils.tensorboard

import torchvision

from imresize import imresize

from DISTS_pytorch import DISTS
from lpips import LPIPS

from data import DIV2K, Sentinel2Processed
from models import EDSR, GSASR
from metrics import PSNR_RGB, CustomSSIM_RGB, CustomLPIPS, CustomDISTS, MetricsList

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
    parser.add_argument('--imgs', type=str, default='imgs', help="Root directory to save images.")
    parser.add_argument('--results', type=str, default='results/metrics_patches_run4.csv', help="Relative path to file directory to store results.")

    tmp_args = parser.parse_args()

    # YAML file
    yaml_args = parse_yaml(tmp_args.config)

    if yaml_args:
        parser.set_defaults(**yaml_args)

    args = parser.parse_args()

    check_args(args)
    return args

def generate_dataset(args):
    if os.path.basename(args.dataset) == 'DIV2K':
        test_dataset = DIV2K(args.dataset, phase='test')
        return test_dataset
    elif os.path.basename(args.dataset) == 'Sentinel-2':
        mode = 'rgb' if args.channels == 3 else 'ms'
        test_dataset = Sentinel2Processed(args.dataset, mode=mode, phase='test')
        return test_dataset

def crop_image(imgs, window_size):
    if imgs.dim == 3:
        imgs.unsqueeze(0)
    
    B, C, H, W = imgs.shape
    
    extra_height = H % window_size
    extra_width = W % window_size

    num_height_windows = H // window_size
    num_width_windows = W // window_size

    windowed_imgs = imgs[..., :H-extra_height, :W-extra_width].view(-1, C, num_height_windows, window_size, num_width_windows, window_size).permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, C, window_size, window_size)
    return windowed_imgs

def downsampling(imgs, scale):
    if imgs.dim == 3:
        imgs.unsqueeze(0)

    imgs_list = [torch.from_numpy(imresize(sample.permute(1, 2, 0).cpu().numpy(), scalar_scale=1/scale)).permute(2, 0, 1).contiguous() for sample in imgs]
    lr_images = torch.stack(imgs_list, dim=0).to(dtype=torch.float32)
    return lr_images

def test(model, dataloader, scale, metrics, writer, imgs_folder, device, args):
    print(imgs_folder)
    model.eval()
    metrics.reset()

    for gt in tqdm(dataloader, desc=f"Testing x{scale} scaling", unit="batches", ascii=True):
        gt_patches = crop_image(gt, 48 * scale)

        lr_patches = downsampling(gt_patches, scale)
        
        output_patches = torch.zeros_like(gt_patches)
        for batch in range(0, lr_patches.shape[0], args.batch_size):
            output_patches[batch:batch+min(args.batch_size, output_patches.shape[0] - batch)] = model(lr_patches[batch:batch+min(args.batch_size, lr_patches.shape[0] - batch)].to(device=device), scale).to(device='cpu')
            metrics(output_patches[batch:batch+min(args.batch_size, output_patches.shape[0] - batch)].to(device=device), gt_patches[batch:batch+min(args.batch_size, gt_patches.shape[0] - batch)].to(device=device))

    channel_idx = random.randint(0, args.channels - 4) if args.channels > 3 else 0

    writer.add_images('Images/test/input', torchvision.utils.make_grid(lr_patches[-4:, channel_idx:channel_idx + 3], nrow=2).unsqueeze(0), scale)
    writer.add_images('Images/test/groundtruth', torchvision.utils.make_grid(gt_patches[-4:, channel_idx:channel_idx + 3], nrow=2).unsqueeze(0), scale)
    writer.add_images('Images/test/output', torchvision.utils.make_grid(output_patches[-4:, channel_idx:channel_idx + 3], nrow=2).unsqueeze(0), scale)
    for metric in metrics:
        writer.add_scalar(f'{metric.name}/test', metric.get_value(), scale)

    img_idx = random.randint(0, lr_patches.shape[0] - 1)
    if args.channels == 3:
        cv2.imwrite(os.path.join(imgs_folder, f"lr_x{scale}.png"), cv2.cvtColor(lr_patches[img_idx].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(imgs_folder, f"output_x{scale}.png"), cv2.cvtColor(output_patches[img_idx].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255, cv2.COLOR_RGB2BGR))
    else:
        with rasterio.open(os.path.join(imgs_folder, f"lr_x{scale}.jp2"), 'w', height=int(lr_patches.shape[2]), width=int(lr_patches.shape[3]), dtype=rasterio.uint8, count=args.channels) as dst:
            raster_img = (lr_patches[img_idx].clamp(0, 1).cpu().numpy() * 255).astype('uint8')
            for c in range(args.channels):
                dst.write_band(c + 1, raster_img[c])
        
        with rasterio.open(os.path.join(imgs_folder, f"output_x{scale}.jp2"), 'w', height=int(output_patches.shape[2]), width=int(output_patches.shape[3]), dtype=rasterio.uint8, count=args.channels) as dst:
            raster_img = (output_patches[img_idx].clamp(0, 1).cpu().numpy() * 255).astype('uint8')
            for c in range(args.channels):
                dst.write_band(c + 1, raster_img[c])
        
def main():
    if not torch.cuda.is_available():
        print("CUDA is not available. Cant't test")
        return
    
    device='cuda'

    args = parse_args()

    # Load model
    backbone = EDSR(args.channels, args.resblocks, args.backbone_features)
    model = GSASR(backbone, args.model_features, args.window_size, args.num_heads, args.gaussian_interaction_blocks, args.channels,
                  raster_ratio=args.raster_ratio, m=args.gaussian_density, mlp_ratio=args.mlp_ratio).to(device=device)
    model.load_state_dict(torch.load(args.weights)['model'])

    # Load data
    test_dataset = generate_dataset(args)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=4)

    # Metrics
    my_dists = DISTS().to(device=device)
    my_dists.eval()
    my_lpips = LPIPS(net='alex').to(device=device)
    my_lpips.eval()
    metrics = MetricsList(PSNR_RGB(data_range=1.), CustomSSIM_RGB(channels=args.channels, data_range=1.), CustomDISTS(my_dists), CustomLPIPS(my_lpips))
    metrics_dict = {metric.name: [] for metric in metrics}
    metrics_dict['scale'] = []
    for metric in metrics:
        metrics_dict[metric.name] = []

    # Logs
    test_name = f"Test{len(os.listdir(args.log))}"
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=os.path.join(args.log, test_name))

    imgs_folder = os.path.join(args.imgs, test_name)
    os.makedirs(imgs_folder, exist_ok=True)

    # Test
    for scale in args.scales:
        test(model, test_dataloader, scale, metrics, writer, imgs_folder, device, args)
        metrics_dict['scale'].append(scale)
        for metric in metrics:
            metrics_dict[metric.name].append(metric.get_value())
    
    # Save to .csv
    pd.DataFrame(metrics_dict).to_csv(os.path.join(args.results, f'{test_name}.csv'))

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with torch.no_grad():
        main()
