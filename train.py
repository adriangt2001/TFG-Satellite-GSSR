import os

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from models.gsasr import GSASR
from models.backbones import EDSR
from data import DIV2K, ScaleBatchSampler, CustomRandomHorizontalFlip, CustomRandomVerticalFlip, CustomRandomRotation
from metrics import MetricsList, PSNR, SSIM, CustomDists, CustomLPIPS

# TODO:
# - Add a log at the beginning of the training with the model parameters count
# - Add ability to resume training
# - Look into gradient clipping, copilot says it helps stability
# - Look into AMP (Automatic Mixed Precision), copilot says it helps speed up training 

def parse_args():
    parser = argparse.ArgumentParser()
    # Training arguments
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--warmup_epochs', type=int, default=3)
    parser.add_argument('--decay_steps', type=int, nargs='+', default=[40, 60, 80])
    parser.add_argument('--decay_factor', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=10)  # Added patience parameter
    
    # Backbone arguments
    parser.add_argument('--num_resblocks', type=int, default=16)
    parser.add_argument('--num_features', type=int, default=64)

    # Model arguments
    parser.add_argument('--out_features', type=int, default=64)
    parser.add_argument('--window_size', type=int, nargs=2, default=[12, 12])
    parser.add_argument('--num_heads', type=int, default=12)
    parser.add_argument('--n_gaussian_interaction_blocks', type=int, default=4)
    parser.add_argument('--num_colors', type=int, default=10)
    parser.add_argument('--raster_ratio', type=int, default=4)
    parser.add_argument('--m', type=int, default=16)
    parser.add_argument('--mlp_ratio', type=float, default=4)

    args = parser.parse_args()
    return args

def train(
        model: nn.Module,
        dataloader: DataLoader,
        criterion: nn.L1Loss,
        optimizer: torch.optim.Adam,
        metrics: MetricsList,
        epoch: int,
        writer: SummaryWriter,
        args,
        device='cuda'
    ):
    # Batch Training Loop
    model.train()
    metrics.reset()
    epoch_loss = 0.0
    progress_description = f"Epoch [{epoch+1}/{args.epochs}]"
    progress_unit = f"batches"
    with tqdm(dataloader, desc=progress_description, unit=progress_unit) as pbar:
        for (lr, gt, scale) in pbar:
            optimizer.zero_grad()

            lr, gt = lr.to(device=device), gt.to(device=device)
            output = model(lr, scale)
            loss = criterion(output, gt)
            metrics(output, gt)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            pbar.set_postfix(loss=loss.item())
    
    # Logging into Tensorboard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    for metric in metrics.metrics:
        writer.add_scalar(f'{metric.name}/train', metric.get_value(), epoch)

    writer.add_images('Images/train/input', lr[0:2], epoch)
    writer.add_images('Images/train/output', output[0:2], epoch)
    writer.add_images('Images/train/gt', gt[0:2], epoch)


@torch.no_grad()
def valid(
        model: nn.Module,
        dataloader: DataLoader,
        metrics: MetricsList,
        epoch: int,
        writer: SummaryWriter,
        args,
        device='cuda'
    ):
    model.eval()
    metrics.reset()
    for (lr, gt, scale) in dataloader:
        lr, gt = lr.to(device=device), gt.to(device=device)
        output = model(lr, scale)
        metrics(output, gt)
    
    # Logging into Tensorboard
    for metric in metrics.metrics:
        writer.add_scalar(f'{metric.name}/valid', metric.get_value(), epoch)
    
    writer.add_images('Images/valid/input', lr[0:2], epoch)
    writer.add_images('Images/valid/output', output[0:2], epoch)
    writer.add_images('Images/valid/gt', gt[0:2], epoch)

def main():
    args = parse_args()
    device = 'cuda'
    
    # Load model
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
    ).to(device=device)

    # Load data
    transforms = T.Compose([
        CustomRandomHorizontalFlip(),
        CustomRandomVerticalFlip(),
        CustomRandomRotation(degrees=180),
    ])
    dataset_train = DIV2K(args.data_dir, phase='train', preload=False, transforms=transforms)
    dataset_valid = DIV2K(args.data_dir, phase='valid', preload=False)
    sampler_train = ScaleBatchSampler(len(dataset_train), args.batch_size)
    dataloader_train = DataLoader(
        dataset_train,
        batch_sampler=sampler_train,
        num_workers=4
    )
    dataloader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )

    # Metrics
    metrics_train = MetricsList(PSNR(data_range=1.), SSIM(data_range=1.), CustomDists(), CustomLPIPS())
    metrics_valid = MetricsList(PSNR(data_range=1.), SSIM(data_range=1.), CustomDists(), CustomLPIPS())

    # Hyperparameters
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr/(args.warmup_epochs + 1), total_iters=args.warmup_epochs)
    lr_decay_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_steps, gamma=args.decay_factor)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [lr_warmup_scheduler, lr_decay_scheduler],
        [args.warmup_epochs]
    )

    # Tensorboard logger
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Initialize best_psnr and best_epoch
    best_psnr = 0.0
    best_epoch = 0

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    for e in range(args.epochs):
        train(
            model,
            dataloader_train,
            criterion,
            optimizer,
            metrics_train,
            e,
            writer,
            args,
            device=device
        )
        valid(
            model,
            dataloader_valid,
            metrics_valid,
            e,
            writer,
            args,
            device=device
        )

        # Save checkpoints
        if e % args.save_interval == 0:
            checkpoint_path = os.path.join(args.checkpoint_dir, f'checkpoint_{e}.pth')
            torch.save(model.state_dict(), checkpoint_path)

        # Save best model
        current_psnr = metrics_valid.metrics[0].get_value()
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_epoch = e
            best_model_path = os.path.join(args.checkpoint_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)

        # Early stopping
        if e - best_epoch > args.patience:
            print(f"Early stopping at epoch {e} as no improvement in PSNR for {args.patience} epochs")
            break

        # Update learning rate
        lr_scheduler.step()

if __name__ == '__main__':
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    main()