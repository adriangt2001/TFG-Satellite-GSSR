import os
import torch
import torchvision.io as io
import random

class DIV2K(torch.utils.data.Dataset):
    def __init__(self, path, phase='train', transforms=None, preload=True):
        self.path = path
        self.phase = phase
        self.transforms = transforms
        self.preload = preload

        self.load_data()

    
    def load_data(self):
        HR_path = os.path.join(self.path, f'DIV2K_{self.phase}_HR')
        LR_bicubic_X2_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_bicubic', 'X2')
        LR_bicubic_X3_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_bicubic', 'X3')
        LR_bicubic_X4_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_bicubic', 'X4')
        LR_unknown_X2_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_unknown', 'X2')
        LR_unknown_X3_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_unknown', 'X3')
        LR_unknown_X4_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_unknown', 'X4')

        # Store paths for on-demand loading
        self.paths = {
            'HR': [],
            'LR_bicubic_X2': [],
            'LR_bicubic_X3': [],
            'LR_bicubic_X4': [],
            'LR_unknown_X2': [],
            'LR_unknown_X3': [],
            'LR_unknown_X4': []
        }

        if self.preload:
            data_HR = []
            data_LR_bicubic_X2 = []
            data_LR_bicubic_X3 = []
            data_LR_bicubic_X4 = []
            data_LR_unknown_X2 = []
            data_LR_unknown_X3 = []
            data_LR_unknown_X4 = []

        for file in os.listdir(HR_path):
            HR_img_path = os.path.join(HR_path, file)
            LR_img_bicubic_X2_path = os.path.join(LR_bicubic_X2_path, file)
            LR_img_bicubic_X3_path = os.path.join(LR_bicubic_X3_path, file)
            LR_img_bicubic_X4_path = os.path.join(LR_bicubic_X4_path, file)
            LR_img_unknown_X2_path = os.path.join(LR_unknown_X2_path, file)
            LR_img_unknown_X3_path = os.path.join(LR_unknown_X3_path, file)
            LR_img_unknown_X4_path = os.path.join(LR_unknown_X4_path, file)

            # Store paths for all images
            self.paths['HR'].append(HR_img_path)
            self.paths['LR_bicubic_X2'].append(LR_img_bicubic_X2_path)
            self.paths['LR_bicubic_X3'].append(LR_img_bicubic_X3_path)
            self.paths['LR_bicubic_X4'].append(LR_img_bicubic_X4_path)
            self.paths['LR_unknown_X2'].append(LR_img_unknown_X2_path)
            self.paths['LR_unknown_X3'].append(LR_img_unknown_X3_path)
            self.paths['LR_unknown_X4'].append(LR_img_unknown_X4_path)

            if self.preload:
                img = io.read_image(HR_img_path, io.ImageReadMode.RGB).float() / 255.0
                data_HR.append(img)
                img = io.read_image(LR_img_bicubic_X2_path, io.ImageReadMode.RGB).float() / 255.0
                data_LR_bicubic_X2.append(img)
                img = io.read_image(LR_img_bicubic_X3_path, io.ImageReadMode.RGB).float() / 255.0
                data_LR_bicubic_X3.append(img)
                img = io.read_image(LR_img_bicubic_X4_path, io.ImageReadMode.RGB).float() / 255.0
                data_LR_bicubic_X4.append(img)
                img = io.read_image(LR_img_unknown_X2_path, io.ImageReadMode.RGB).float() / 255.0
                data_LR_unknown_X2.append(img)
                img = io.read_image(LR_img_unknown_X3_path, io.ImageReadMode.RGB).float() / 255.0
                data_LR_unknown_X3.append(img)
                img = io.read_image(LR_img_unknown_X4_path, io.ImageReadMode.RGB).float() / 255.0
                data_LR_unknown_X4.append(img)

        if self.preload:
            self.data_HR = torch.stack(data_HR)
            data_LR_bicubic_X2 = torch.stack(data_LR_bicubic_X2)
            data_LR_bicubic_X3 = torch.stack(data_LR_bicubic_X3)
            data_LR_bicubic_X4 = torch.stack(data_LR_bicubic_X4)
            data_LR_unknown_X2 = torch.stack(data_LR_unknown_X2)
            data_LR_unknown_X3 = torch.stack(data_LR_unknown_X3)
            data_LR_unknown_X4 = torch.stack(data_LR_unknown_X4)

            self.data_LR = [
                data_LR_bicubic_X2,
                data_LR_bicubic_X3,
                data_LR_bicubic_X4,
                data_LR_unknown_X2,
                data_LR_unknown_X3,
                data_LR_unknown_X4
            ]
    
    def __len__(self):
        # Total length is 6x the number of images (6 different scales for each image)
        if self.preload:
            return len(self.data_HR) * 6
        else:
            return len(self.paths['HR']) * 6
    
    def __getitem__(self, idx):
        # Parse the index to get the actual index and scale part
        # Format: idx = real_idx * 6 + scale_part
        scale_part = idx % 6
        real_idx = idx // 6
        
        scale = scale_part % 3 + 2  # Maps 0,3->2, 1,4->3, 2,5->4
        
        if self.preload:
            gt = self.data_HR[real_idx]
            lr = self.data_LR[scale_part][real_idx]
        else:
            # Load on-demand from file paths
            if scale_part == 0:
                lr_path = self.paths['LR_bicubic_X2'][real_idx]
            elif scale_part == 1:
                lr_path = self.paths['LR_bicubic_X3'][real_idx]
            elif scale_part == 2:
                lr_path = self.paths['LR_bicubic_X4'][real_idx]
            elif scale_part == 3:
                lr_path = self.paths['LR_unknown_X2'][real_idx]
            elif scale_part == 4:
                lr_path = self.paths['LR_unknown_X3'][real_idx]
            else:  # scale_part == 5
                lr_path = self.paths['LR_unknown_X4'][real_idx]
            
            gt = io.read_image(self.paths['HR'][real_idx], io.ImageReadMode.RGB).float() / 255.0
            lr = io.read_image(lr_path, io.ImageReadMode.RGB).float() / 255.0

        if self.transforms:
            lr, gt = self.transforms(lr, gt)
        
        return lr, gt, scale


class ScaleBatchSampler(torch.utils.data.Sampler):
    """
    Batch sampler that ensures all samples in a batch have the same scale,
    but randomizes which scale is used between batches.
    """
    def __init__(self, dataset_size, batch_size, num_scales=6, shuffle=True):
        self.base_dataset_size = dataset_size // num_scales
        self.batch_size = batch_size
        self.num_scales = num_scales
        self.shuffle = shuffle
        
    def __iter__(self):
        # Create indices for each scale_part
        scale_batches = []
        
        for scale_part in range(self.num_scales):
            indices = list(range(scale_part, self.base_dataset_size * 6, 6))
            if self.shuffle:
                random.shuffle(indices)
            
            # Group into batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i:i + self.batch_size]
                if batch:  # Only add non-empty batches
                    scale_batches.append(batch)
        
        # Shuffle the order of batches if requested
        if self.shuffle:
            random.shuffle(scale_batches)
            
        # Yield each batch
        for batch in scale_batches:
            yield batch
    
    def __len__(self):
        return (self.base_dataset_size * self.num_scales + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = DIV2K('/home/msiau/data/tmp/agarciat/DIV2K_processed', phase='train', preload=False)
    batch_size = 16
    sampler = ScaleBatchSampler(len(dataset), batch_size)

    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4
    )

    # In your training loop
    for batch_idx, (lr, gt, scale) in enumerate(dataloader):
        # Now all images in the batch have the same scale
        # lr and gt are properly batched tensors
        # scale is the same for all images in this batch
        print(f"Batch {batch_idx}, Scale: {scale[0]}")
        print(f"LR shape: {lr.shape}, GT shape: {gt.shape}")
