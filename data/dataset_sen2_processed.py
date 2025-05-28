import os
from typing import Literal
import random
import torch
from tqdm import tqdm

class Sentinel2Processed(torch.utils.data.Dataset):
    def __init__(self, path, mode: Literal['rgb', 'ms'] = 'ms', phase: Literal['train', 'val', 'test'] ='train', transforms=None, seed=None):
        if seed:
            random.seed(seed)
        
        self.mode = mode
        self.phase = phase
        self.transforms = transforms
        path = os.path.join(path, phase, 'processed_data')

        self.images = self.load_data(path)
    
    def load_data(self, path):
        files = os.listdir(path)
        images = []
        for file in tqdm(files, desc=f"Loading {self.phase} dataset", unit="file", ascii=True):
            img_path = os.path.join(path, file)
            img = torch.load(img_path)
            images.append(img)
        
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if self.phase == 'test':
            img = self.images[idx]

            if self.mode == 'rgb':
                img = img[:3]
            
            if self.transforms:
                img = self.transforms(img)

            return img
        else:
            scale_factor = idx % 10 + 1
            img_idx = idx // 10 
            img = self.images[img_idx]

            # Crop random s48xs48 patch of the image
            patch_size = scale_factor * 48
            col_start = random.randint(0, img.shape[-2] - patch_size)
            row_start = random.randint(0, img.shape[-1] - patch_size)
            img = img[:, col_start:col_start+patch_size, row_start:row_start+patch_size]
            
            if self.mode == 'rgb':
                img = img[:3]

            img = img.unsqueeze(0)

            if self.transforms:
                img = self.transforms(img)
            
            # Generate downsampled version of the image
            lr = torch.nn.functional.interpolate(img, size=48, mode='bicubic', antialias=True).squeeze()
            img = img.squeeze()

            return lr, img, float(scale_factor)

class ScaleBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset_size, batch_size, num_scales=4, shuffle=True):
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.num_scales = num_scales
        self.shuffle = shuffle
    
    def __iter__(self):
        # Pick one of the 800 images, which will be sampled for this batch
        # Group images in batches
        batches = []
        indices = range(self.dataset_size)

        for idx in indices:
            # Process index in the form [image index concat scale factor]
            scale_factor = random.uniform(0, 4)

            # Group into batches
            batch = [int(idx * 10 + scale_factor) for _ in range(self.batch_size)]
            batches.append(batch)

        random.shuffle(batches)

        for batch in batches:
            yield batch
    
    def __len__(self):
        return self.dataset_size

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    dataset = Sentinel2Processed('/home/msiau/data/tmp/agarciat/Sentinel-2', phase='val')
    print(dataset[0][0].min())
    batch_size = 2
    sampler = ScaleBatchSampler(len(dataset), batch_size)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4
    )

    for (lr, gt, scale) in tqdm(dataloader, ascii=True):
        pass
