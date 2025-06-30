# Basic imports
import os
import torch.utils.tensorboard
import yaml
import argparse
import random
from tqdm import tqdm

import torchvision
from torchvision.transforms import Compose

import torch
from torch.utils.data import DataLoader

from DISTS_pytorch import DISTS
from lpips import LPIPS

# Custom imports
from data import DIV2K, Sentinel2Processed, ScaleBatchSampler
from models import EDSR, GausSat
from data.metrics import MetricsList, PSNR, CustomSSIM, CustomLPIPS, CustomDISTS

# Warning supresion
import warnings
warnings.filterwarnings("ignore", category=UserWarning, message="The parameter 'pretrained' is deprecated")
warnings.filterwarnings("ignore", category=UserWarning, message="Arguments other than a weight enum")
warnings.filterwarnings("ignore", category=UserWarning, message="To copy construct from a tensor")
warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")

import logging
logging.getLogger("lpips").setLevel(logging.ERROR)


def check_args(args):
    # Dataset
    if not args.dataset: raise argparse.ArgumentError(None, 'You MUST provide a path to the dataset folder')

    # Backbone
    if args.resblocks <= 0: raise argparse.ArgumentError(None, 'Resblocks must be positive')
    if args.backbone_features <= 0: raise argparse.ArgumentError(None, 'Backbone features must be positive')

    # Model
    if args.channels <= 0: raise argparse.ArgumentError(None, 'Channels must be positive')
    if args.model_features <= 0: raise argparse.ArgumentError(None, 'Model features must be positive')
    if args.window_size <= 0: raise argparse.ArgumentError(None, 'Window size must be positive')
    if args.num_heads <= 0: raise argparse.ArgumentError(None, 'Number of heads must be positive')
    if args.gaussian_interaction_blocks <= 0: raise argparse.ArgumentError(None, 'Gaussian Interaction Blocks must be positive')
    if args.raster_ratio <= 0 or args.raster_ratio > 1: raise argparse.ArgumentError(None, 'Raster ratio must be a number between 0 and 1')
    if args.gaussian_density <= 0: raise argparse.ArgumentError(None, 'Gaussian density must be positive')
    if args.mlp_ratio <= 0: raise argparse.ArgumentError(None, 'MLP ratio must be positive')

    # Training
    if args.epochs <= 0: raise argparse.ArgumentError(None, 'Epochs must be positive')
    if args.batch_size <= 0: raise argparse.ArgumentError(None, 'Batch size must be positive')
    if args.lr <= 0: raise argparse.ArgumentError(None, 'Learning rate must be positive')
    if args.warmup_iterations < 0: argparse.ArgumentError(None, 'Warmup iterations must be >= 0')
    if args.decay_factor <= 0: argparse.ArgumentError(None, 'Decay factor must be positive')
    if args.patience <= 0: argparse.ArgumentError(None, 'Patience must be positive')
    if args.save_interval <= 0: argparse.ArgumentError(None, 'Save interval must be positive')

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

    # Config
    parser.add_argument('--config', type=str, default=None, help="YAML configuration file with argument values.")

    # Dataset
    parser.add_argument('--dataset', type=str, help="Path to the dataset.")
    parser.add_argument('--seed', type=int, default=42, help="Seed to reproduce experiments in the randomly generated patches from the dataset.")

    # Backbone
    parser.add_argument('--resblocks', type=int, default=16, help="Number of Residual Blocks in the backbone.")
    parser.add_argument('--backbone_features', type=int, default=64, help="Embedding size of the backbone.")

    # Model
    parser.add_argument('--name', type=str, default='Model', help="Name of the current model")
    parser.add_argument('--channels', type=int, default=3, help="Number of channels of input images.")
    parser.add_argument('--model_features', type=int, default=180, help="Embedding size of the model.")
    parser.add_argument('--window_size', type=int, default=12, help="Window size.")
    parser.add_argument('--num_heads', type=int, default=4, help="Number of attention heads.")
    parser.add_argument('--gaussian_interaction_blocks', type=int, default=6, help="Number of Gaussian Interaction Blocks.")
    parser.add_argument('--raster_ratio', type=float, default=0.1, help="Raster ratio of the Gaussian Rasterizer.")
    parser.add_argument('--gaussian_density', type=int, default=16, help="Density of gaussians.")
    parser.add_argument('--mlp_ratio', type=float, default=4., help="Decreasing ratio of MLPs.")

    # Training
    parser.add_argument('--epochs', type=int, default=100, help="Number of epochs.")
    parser.add_argument('--batch_size', type=int, default=4, help="Batch size per iteration.")
    parser.add_argument('--pretrained_backbone', type=str, help="Path to the weights of the specified backbone.")
    parser.add_argument('--lr', type=float, default=2e-4, help="Initial learning rate (after warmup if specified).")
    parser.add_argument('--warmup_iterations', type=int, default=2000, help="Number of warmup iterations.")
    parser.add_argument('--decay_steps', type=int, nargs='*', default=[250000, 400000, 450000, 475000], help="Iterations at which the learning rate decays by a given factor.")
    parser.add_argument('--decay_factor', type=float, default=0.5, help="Decay factor of the learning rate.")
    parser.add_argument('--gclip_type', type=str, default=None, help="Gradient clipping to use.")
    parser.add_argument('--gclip_value', type=float, default=0., help="Gradient clipping value.")
    parser.add_argument('--valid_interval', type=int, default=1, help="Validation interval.")
    parser.add_argument('--patience', type=int, default=10, help="Early stopping patience.")
    parser.add_argument('--checkpoint', type=str, default='checkpoints', help="Root directory to store checkpoints.")
    parser.add_argument('--save_interval', type=int, default=1, help="Checkpointing interval.")
    parser.add_argument('--resume', type=str, default=None, help="Checkpoint file to resume training from.")
    parser.add_argument('--log', type=str, default='logs', help="Root directory to store logs.")

    tmp_args = parser.parse_args()

    # YAML file
    yaml_args = parse_yaml(tmp_args.config)

    if yaml_args:
        parser.set_defaults(**yaml_args)

    args = parser.parse_args()
    
    check_args(args)

    return args

def generate_dataset(args, transforms):
    if os.path.basename(args.dataset) == 'DIV2K':
        train_dataset = DIV2K(args.dataset, phase='train', transforms=transforms, seed=args.seed)
        val_dataset = DIV2K(args.dataset, phase='valid', transforms=transforms, seed=args.seed)
        return train_dataset, val_dataset
    elif os.path.basename(args.dataset) == 'Sentinel-2':
        mode = 'rgb' if args.channels == 3 else 'ms'
        train_dataset = Sentinel2Processed(args.dataset, mode=mode, phase='train', transforms=transforms, seed=args.seed)
        val_dataset = Sentinel2Processed(args.dataset, mode=mode, phase='valid', transforms=transforms, seed=args.seed)
        return train_dataset, val_dataset

def save_tensor(t, name):
    torch.save(t, f"investigation/{name}.pt")

def train(epoch, dataloader, model, criterion, optimizer, lr_scheduler, scaler, gclipping, metrics, writer, args, device):
    mean_loss = 0.
    global_iter = epoch * len(dataloader)

    model.train()
    metrics.reset()
    
    with tqdm(dataloader, desc=f"Epoch[{epoch+1}/{args.epochs}]", unit=f"batches") as pbar:
        for i, (lr, gt, scale) in enumerate(pbar):
            current_iter = global_iter + i
            current_lr = optimizer.param_groups[0]['lr']
            
            optimizer.zero_grad(set_to_none=True)
            
            lr, gt = lr.to(device=device), gt.to(device=device)

            with torch.amp.autocast(device):
                output = model(lr, scale[0].item())
                loss = criterion(output, gt)
                isNan = torch.isnan(loss)
                if torch.any(isNan):
                    # save_tensor(lr[torch.any(isNan, dim=0)], "lr")
                    # writer.add_image('Image/train/lr_anomaly', lr[torch.any(isNan, dim=0)], global_iter)

                    # save_tensor(gt[torch.any(isNan, dim=0)], "gt")
                    # writer.add_image('Image/train/gt_anomaly', gt[torch.any(isNan, dim=0)], global_iter)
                    
                    # save_tensor(output[torch.any(isNan, dim=0)], "output")
                    # writer.add_image('Image/train/output_anomaly', output[torch.any(isNan, dim=0)], global_iter)

                    raise ValueError(f"NaN appeared at iteration {global_iter} (epoch {epoch}). Size of related tensors are {lr.shape=}, {gt.shape=}, {output.shape=}, {loss.shape=}.\
                                     NaNs in\nLr image: {torch.any(torch.isnan(lr), dim=0)}\nGt image: {torch.any(torch.isnan(gt), dim=0)}\nOutput image: {torch.any(torch.isnan(output), dim=0)}\nLoss: {torch.any(torch.isnan(loss), dim=0)}")
            
            scaler.scale(loss).backward()

            if gclipping:
                scaler.unscale_(optimizer)
                gclipping(model.parameters(), args.gclip_value)
            
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            mean_loss += loss.item()

            with torch.no_grad():
                metrics(output, gt)
                writer.add_scalar('Loss/LR', current_lr, current_iter)
                writer.add_scalar('Loss/iter', loss.item(), current_iter)
                pbar.set_postfix(loss=loss.item())
    
    channel_idx = random.randint(0, args.channels - 4) if args.channels > 3 else 0

    writer.add_images('Images/train/input', torchvision.utils.make_grid(lr[:min(args.batch_size, 4), channel_idx:channel_idx + 3], nrow=2).unsqueeze(0), epoch)
    writer.add_images('Images/train/groundtruth', torchvision.utils.make_grid(gt[:min(args.batch_size, 4), channel_idx:channel_idx + 3], nrow=2).unsqueeze(0), epoch)
    writer.add_images('Images/train/output', torchvision.utils.make_grid(output[:min(args.batch_size, 4), channel_idx:channel_idx + 3], nrow=2).unsqueeze(0), epoch)
    writer.add_scalar('Loss/train', mean_loss/len(dataloader), epoch)
    for metric in metrics:
        writer.add_scalar(f'{metric.name}/train', metric.get_value(), epoch)

@torch.no_grad()
def valid(epoch, dataloader, model, criterion, metrics, writer, args, device):
    mean_loss = 0.

    model.eval()
    metrics.reset()
    for (lr, gt, scale) in tqdm(dataloader, desc=f"Validation", unit="batches"):
        lr, gt = lr.to(device=device), gt.to(device=device)
        output = model(lr, scale[0].item())

        mean_loss += criterion(output, gt)
        metrics(output, gt)
    
    channel_idx = random.randint(0, args.channels - 4) if args.channels > 3 else 0

    writer.add_images('Images/valid/input', torchvision.utils.make_grid(lr[:min(args.batch_size, 4), channel_idx:channel_idx + 3], nrow=2).unsqueeze(0), epoch)
    writer.add_images('Images/valid/groundtruth', torchvision.utils.make_grid(gt[:min(args.batch_size, 4), channel_idx:channel_idx + 3], nrow=2).unsqueeze(0), epoch)
    writer.add_images('Images/valid/output', torchvision.utils.make_grid(output[:min(args.batch_size, 4), channel_idx:channel_idx + 3], nrow=2).unsqueeze(0), epoch)
    writer.add_scalar('Loss/valid', mean_loss/len(dataloader), epoch)
    for metric in metrics:
        writer.add_scalar(f'{metric.name}/valid', metric.get_value(), epoch)

def main():
    if not torch.cuda.is_available():
        raise Exception("CUDA is not available. This model needs CUDA")
    
    args = parse_args()

    # Variables
    device = 'cuda'
    start_epoch = 0
    best_metric = 0.
    best_epoch = 0
    
    # Transforms
    transforms = Compose([
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.RandomHorizontalFlip()
    ])

    # Dataset
    train_dataset, valid_dataset = generate_dataset(args, transforms)
    
    train_sampler = ScaleBatchSampler(len(train_dataset), args.batch_size, num_scales=4, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, num_workers=4)

    valid_sampler = ScaleBatchSampler(len(valid_dataset), args.batch_size, num_scales=4, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_sampler=valid_sampler, num_workers=4)

    # Backbone
    backbone = EDSR(args.channels, args.resblocks, args.backbone_features)
    if args.pretrained_backbone:
        backbone.load_state_dict(torch.load(args.pretrained_backbone), strict=False)
    
    # Model
    model = GausSat(backbone, args.model_features, args.window_size, args.num_heads, args.gaussian_interaction_blocks, args.channels, 
                  raster_ratio=args.raster_ratio, m=args.gaussian_density, mlp_ratio=args.mlp_ratio).to(device=device)
    opt_model = torch.compile(model, mode="reduce-overhead")
    
    # Metrics
    my_dists = DISTS().to(device=device)
    my_lpips = LPIPS(net='alex').to(device=device)
    train_metrics = MetricsList(PSNR(), CustomSSIM(args.channels), CustomLPIPS(my_lpips), CustomDISTS(my_dists))
    valid_metrics = MetricsList(PSNR(), CustomSSIM(args.channels), CustomLPIPS(my_lpips), CustomDISTS(my_dists))

    # Training Hyperparameters
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler()
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=args.lr/args.warmup_iterations, total_iters=2000),
         torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.decay_steps, gamma=args.decay_factor)],
        [args.warmup_iterations]
    )
    gradient_clipping = None
    if args.gclip_type:
        if args.gclip_type == 'norm':
            gradient_clipping = torch.nn.utils.clip_grad_norm_
        elif args.gclip_type == 'value':
            gradient_clipping = torch.nn.utils.clip_grad_value_
        else:
            raise ValueError(f"Unknown gradient clipping type: {args.gclip_type}")
    
    # Check root dirs
    os.makedirs(args.checkpoint, exist_ok=True)
    os.makedirs(args.log, exist_ok=True)

    # New run name
    run_num = max(len(os.listdir(args.checkpoint)), len(os.listdir(args.log))) + 1
    run = f"run{run_num}"

    # Resume training
    if args.resume:
        if not os.path.isfile(args.resume): raise FileNotFoundError(f"Checkpoint {args.resume} not found.")
        
        # Get run name
        run = f"{os.path.basename(os.path.dirname(args.resume))}_resumed"

        # Load checkpoint
        chkpt = torch.load(args.resume)
        start_epoch = chkpt['epoch'] + 1 # +1 because 'epoch' is the last done epoch
        best_metric = chkpt['best_metric']
        best_epoch = chkpt['best_epoch']
        model.load_state_dict(chkpt['model'])
        optimizer.load_state_dict(chkpt['optimizer'])
        lr_scheduler.load_state_dict(chkpt['lr_scheduler'])

    # Create dirs
    os.makedirs(os.path.join(args.checkpoint, run))
    os.makedirs(os.path.join(args.log, run))

    # Logs
    writer = torch.utils.tensorboard.SummaryWriter(log_dir=os.path.join(args.log, run))
    writer.add_text('Arguments', '\n'.join([f"{key}: {value}" for key, value in vars(args).items()]))

    torch.autograd.set_detect_anomaly(True)

    print(f"Training {model.__class__.__name__} with {sum(p.numel() for p in model.parameters())} parameters from epoch {start_epoch}")
    for e in range(start_epoch, args.epochs):
        train(e, train_dataloader, opt_model, criterion, optimizer, lr_scheduler, scaler, gradient_clipping, train_metrics, writer, args, device)

        if e % args.valid_interval == 0:
            valid(e, valid_dataloader, opt_model, criterion, valid_metrics, writer, args, device)

            # Save best model
            if valid_metrics.metrics[0].get_value() > best_metric:
                best_metric = valid_metrics.metrics[0].get_value()
                best_epoch = e
                torch.save({
                    'epoch': e,
                    'best_metric': best_metric,
                    'best_epoch': best_epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                }, os.path.join(args.checkpoint, run, 'best_checkpoint.pt'))
            

        # Save checkpoint
        if e % args.save_interval == 0:
            torch.save({
                'epoch': e,
                'best_metric': best_metric,
                'best_epoch': best_epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
            }, os.path.join(args.checkpoint, run, 'checkpoint.pt'))

        # Early stopping
        if e - best_epoch > args.patience and e > ((args.decay_steps[3] // len(train_dataloader)) + 50):
            print(f"Early stopping at epoch {e} as there's been no improvement in {valid_metrics.metrics[0].name} for {args.patience} epochs")
            break
    
    writer.add_text('Training Finished', f"Training completed after {e+1} epochs. Best {valid_metrics.metrics[0].name}: {best_metric:.4f} at epoch {best_epoch}")
    writer.close()

    # Copy saved checkpoint to official weights folder
    weights = torch.load(os.path.join(args.checkpoint, run, 'best_checkpoint.pt'))['model']
    torch.save({'model': weights}, os.path.join('weights', f'{args.name}.pt'))

if __name__ == "__main__":
    # os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
    main()
