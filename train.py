import os
import yaml

import argparse
import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import time
from DISTS_pytorch import DISTS
import lpips
from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation, Compose

from models.gsasr import GSASR
from models.backbones import EDSR
from data import DIV2K, ScaleBatchSampler, CustomCompose, CustomRandomHorizontalFlip, CustomRandomVerticalFlip, CustomRandomRotation
from metrics import MetricsList, PSNR, CustomSSIM, CustomDists, CustomLPIPS

import warnings
# Filter out specific warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum")
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")

# Suppress messages from lpips
import logging
logging.getLogger("lpips").setLevel(logging.ERROR)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', type=str, default=None)

    # Training arguments
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_interval', type=int, default=2)
    parser.add_argument('--warmup_epochs', type=int, default=2000)
    parser.add_argument('--decay_steps', type=int, nargs='+', default=[250000, 400000, 450000, 475000])
    parser.add_argument('--decay_factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--gradient_clipping', type=float, default=0.0)
    parser.add_argument('--gradient_clipping_type', type=str, default=None)

    # Dataset arguments    
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--num_data', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)

    # Backbone arguments
    parser.add_argument('--num_resblocks', type=int, default=16)
    parser.add_argument('--backbone_features', type=int, default=64)
    parser.add_argument('--pretrained_backbone', type=str, default=None)

    # Model arguments
    parser.add_argument('--window_size', type=int, default=12)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--model_features', type=int, default=64)
    parser.add_argument('--n_gaussian_interaction_blocks', type=int, default=6)
    parser.add_argument('--num_channels', type=int, default=3)
    parser.add_argument('--raster_ratio', type=float, default=0.1)
    parser.add_argument('--m', type=int, default=16)
    parser.add_argument('--mlp_ratio', type=float, default=4.)

    temp_args, _ = parser.parse_known_args()

    yaml_values = parse_yaml(temp_args.config)

    if yaml_values:
        parser.set_defaults(**yaml_values)

    args = parser.parse_args()
    return args

def parse_yaml(config_file):
    if not config_file:
        return None

    yaml_values = {}
    print(f"Loading configuration from: {config_file}")
    with open(config_file, 'r') as f:
        try:
            loaded_yaml = yaml.safe_load(f)
            if loaded_yaml:
                for section, params in loaded_yaml.items():
                    if params:
                        for key, value in params.items():
                            yaml_values[key] = value
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
        except FileNotFoundError:
            print(f"Error: Config file not found at {config_file}")
    return yaml_values

def args_check(args):
    # Training arguments
    if args.data_dir is None:
        raise ValueError("data_dir is required")
    if args.epochs <= 0:
        raise ValueError("epochs should be greater than 0")
    if args.batch_size <= 0:
        raise ValueError("batch_size should be greater than 0")
    if args.lr <= 0:
        raise ValueError("lr should be greater than 0")
    if args.log_interval <= 0:
        raise ValueError("log_interval should be greater than 0")
    if args.save_interval <= 0:
        raise ValueError("save_interval should be greater than 0")
    if args.warmup_epochs < 0:
        raise ValueError("warmup_epochs should be greater than or equal to 0")
    if args.decay_steps is None or len(args.decay_steps) == 0:
        raise ValueError("decay_steps should be a list of integers")
    if args.decay_factor <= 0:
        raise ValueError("decay_factor should be greater than 0")
    if args.patience < 0:
        raise ValueError("patience should be greater than or equal to 0")
    if args.resume is not None and not os.path.isfile(args.resume):
        raise ValueError(f"resume file {args.resume} does not exist")
    if args.gradient_clipping_type is not None and args.gradient_clipping_type not in ['norm', 'value']:
        raise ValueError("gradient_clipping_type should be either None, 'norm' or 'value'")
    if args.gradient_clipping_type is not None and args.gradient_clipping <= 0:
        raise ValueError("gradient_clipping should be greater than 0 when gradient_clipping_type is not None")
    
    # Backbone arguments
    if args.num_resblocks <= 0:
        raise ValueError("num_resblocks should be greater than 0")
    if args.backbone_features <= 0:
        raise ValueError("backbone_features should be greater than 0")
    
    # Model arguments
    if args.window_size <= 0:
        raise ValueError("window_size should be greater than 0")
    if args.num_heads <= 0:
        raise ValueError("num_heads should be greater than 0")
    if args.n_gaussian_interaction_blocks <= 0:
        raise ValueError("n_gaussian_interaction_blocks should be greater than 0")
    if args.num_channels <= 0:
        raise ValueError("num_channels should be greater than 0")
    if args.raster_ratio <= 0:
        raise ValueError("raster_ratio should be greater than 0")
    if args.m <= 0:
        raise ValueError("m should be greater than 0")
    if args.mlp_ratio <= 0:
        raise ValueError("mlp_ratio should be greater than 0")
    
    print("All arguments checks passed!")

def train(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.L1Loss,
        optimizer: torch.optim.Adam,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        scaler: GradScaler,
        metrics: MetricsList,
        epoch: int,
        writer: SummaryWriter,
        gradient_clipping,
        args,
        device='cuda'
    ):
    # Batch Training Loop
    model.train()
    metrics.reset()
    epoch_loss = 0.0
    progress_description = f"Epoch [{epoch+1}/{args.epochs}]"
    progress_unit = f"batches"
    
    # Calculate global iteration
    global_iter = epoch * len(dataloader)
    
    with tqdm(dataloader, desc=progress_description, unit=progress_unit, ascii=True) as pbar:
        for i, (lr, gt, scale) in enumerate(pbar):
            current_iter = global_iter + i
            # Predict
            optimizer.zero_grad(set_to_none=True)
            lr, gt = lr.to(device=device), gt.to(device=device)

            with autocast(device):
                output = model(lr, scale[0].item())
                loss = criterion(output, gt)

            # Log results
            writer.add_scalar('Loss/LR_iter', optimizer.param_groups[0]['lr'], current_iter)
            writer.add_scalar('Loss/train_iter', loss.item(), current_iter)

            # Update stuff
            scaler.scale(loss).backward()
            
            if gradient_clipping:
                scaler.unscale_(optimizer)
                gradient_clipping(model.parameters(), args.gradient_clipping)
            
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            epoch_loss += loss.item()
            
            # Metrics
            with torch.no_grad():
                metrics(output, gt)
                pbar.set_postfix(loss=loss.item())
    
    # Log Images
    writer.add_images('Images/train/input', lr[0:2], epoch)
    writer.add_images('Images/train/output', output[0:2], epoch)
    writer.add_images('Images/train/gt', gt[0:2], epoch)

    # Log Scalars
    writer.add_scalar('Loss/train', epoch_loss/len(dataloader), epoch)
    for metric in metrics.metrics:
        writer.add_scalar(f'{metric.name}/train', metric.get_value(), epoch)

@torch.no_grad()
def valid(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.L1Loss,
        metrics: MetricsList,
        epoch: int,
        writer: SummaryWriter,
        args,
        device='cuda'
    ):
    model.eval()
    metrics.reset()
    valid_loss = 0.0
    for (lr, gt, scale) in tqdm(dataloader, desc=f"Validation", unit="batches", ascii=True):
        # Predict
        lr, gt = lr.to(device=device), gt.to(device=device)
        output = model(lr, scale[0].item())

        # Metrics
        valid_loss += criterion(output, gt)
        metrics(output, gt)
    
    # Log Images
    writer.add_images('Images/valid/input', lr[0:2], epoch)
    writer.add_images('Images/valid/output', output[0:2], epoch)
    writer.add_images('Images/valid/gt', gt[0:2], epoch)

    # Log Scalars
    writer.add_scalar('Loss/valid', valid_loss/len(dataloader), epoch)
    for metric in metrics.metrics:
        writer.add_scalar(f'{metric.name}/valid', metric.get_value(), epoch)

def main():
    if (not torch.cuda.is_available()):
        print("CUDA is not available. Can't train")
        return
    # Parse arguments
    args = parse_args()
    args_check(args)

    # Initialize training variables
    device = 'cuda'
    start_epoch = 0
    best_metric = 0.0
    best_epoch = 0

    # Load data
    transforms = None
    dataset_train = DIV2K(args.data_dir, phase='train', transforms=transforms, seed=args.seed)
    dataset_valid = DIV2K(args.data_dir, phase='valid', seed=args.seed)
    sampler_train = ScaleBatchSampler(len(dataset_train), args.batch_size)
    sampler_valid = ScaleBatchSampler(len(dataset_valid), args.batch_size, shuffle=False)
    dataloader_train = DataLoader(
        dataset_train,
        batch_sampler=sampler_train,
        num_workers=4
    )
    dataloader_valid = DataLoader(
        dataset_valid,
        batch_sampler=sampler_valid,
        num_workers=4
    )

    # Load model
    backbone = EDSR(args.num_channels, args.num_resblocks, args.backbone_features)
    if args.pretrained_backbone:
        backbone.load_state_dict(torch.load(args.pretrained_backbone), strict=False)
        # backbone.requires_grad_(requires_grad=False)
    model = GSASR(
        backbone,
        args.model_features,
        args.window_size,
        args.num_heads,
        args.n_gaussian_interaction_blocks,
        args.num_channels,
        args.raster_ratio,
        args.m,
        args.mlp_ratio
    ).to(device=device)

    # Hyperparameters
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = GradScaler()
    lr_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr/(args.warmup_epochs + 1), total_iters=args.warmup_epochs)
    lr_decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_steps, gamma=args.decay_factor)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [lr_warmup_scheduler, lr_decay_scheduler],
        [args.warmup_epochs]
    )
    gradient_clipping = None
    if args.gradient_clipping_type:
        if args.gradient_clipping_type == 'norm':
            gradient_clipping = torch.nn.utils.clip_grad_norm_
        elif args.gradient_clipping_type == 'value':
            gradient_clipping = torch.nn.utils.clip_grad_value_
        else:
            raise ValueError(f"Unknown gradient clipping type: {args.gradient_clipping_type}")

    # Metrics
    my_dists = DISTS().to(device=device)
    my_lpips = lpips.LPIPS(net='alex').to(device=device)
    metrics_train = MetricsList(PSNR(data_range=1.), CustomSSIM(data_range=1.), CustomDists(my_dists), CustomLPIPS(my_lpips))
    metrics_valid = MetricsList(PSNR(data_range=1.), CustomSSIM(data_range=1.), CustomDists(my_dists), CustomLPIPS(my_lpips))

    # Checkpointing
    checkpoint_dir = os.path.join(args.checkpoint_dir, f'run{len(os.listdir(args.checkpoint_dir)) + 1}')
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Resume from checkpoint if specified
    if hasattr(args, 'resume') and args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1  # Start from the next epoch
            best_metric = checkpoint.get('best_metric', 0.0)
            best_epoch = checkpoint.get('best_epoch', 0)
            print(f"Resumed training from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}', starting from scratch")

    # Logging
    log_dir = os.path.join(args.log_dir, f'run{len(os.listdir(args.log_dir)) + 1}')
    writer = SummaryWriter(log_dir=log_dir)
    
    args_text = "\n".join([f"{arg}: {value}" for arg, value in vars(args).items()])
    writer.add_text('Arguments', args_text, 0)
    print("Arguments logged to Tensorboard")

    print(f"Training {backbone.__class__.__name__} with {sum(p.numel() for p in backbone.parameters())} parameters")
    print(f"Training {model.__class__.__name__} with {sum(p.numel() for p in model.parameters())} parameters")
    for e in range(start_epoch, args.epochs):
        train(
            model,
            dataloader_train,
            criterion,
            optimizer,
            lr_scheduler,
            scaler,
            metrics_train,
            e,
            writer,
            gradient_clipping,
            args,
            device=device
        )
        valid(
            model,
            dataloader_valid,
            criterion,
            metrics_valid,
            e,
            writer,
            args,
            device=device
        )

        # Save checkpoints
        if e % args.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': e,
                'best_metric': best_metric,
                'best_epoch': best_epoch,
            }, checkpoint_path)

        # Save best model
        current_metric = metrics_valid.metrics[0].get_value()
        if current_metric > best_metric:
            best_metric = current_metric
            best_epoch = e
            best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': e,
                'best_metric': best_metric,
                'best_epoch': best_epoch,
            }, best_model_path)

        # Early stopping
        if e - best_epoch > args.patience:
            print(f"Early stopping at epoch {e} as no improvement in {metrics_valid.metrics[0].name} for {args.patience} epochs")
            break

        # Reset metrics
        metrics_train.reset()
        metrics_valid.reset()
    
    # Log end and close
    writer.add_text('Training Status', f'Training completed after {e} epochs. Best {metrics_valid.metrics[0].name}: {best_metric:.4f} at epoch {best_epoch}', 0)
    writer.close()

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main()
