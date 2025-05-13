import os
import random
import torch
import torchvision.io as io
from tqdm import tqdm


class DIV2K(torch.utils.data.Dataset):
    def __init__(self, path, phase='train', transforms=None, seed=None):
        if seed:
            random.seed(seed)
        
        self.phase = phase
        # self.path = path
        self.transforms = transforms
        path = os.path.join(path, f'DIV2K_{phase}_HR')

        self.images = self.load_data(path)
    
    def load_data(self, path):
        files = os.listdir(path)
        images = []
        for file in tqdm(files, desc=f"Loading {self.phase} dataset", unit="file", ascii=True):
            img_path = os.path.join(path, file)
            img = io.decode_image(img_path, io.ImageReadMode.RGB).float() / 255.0
            images.append(img)
        
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # What if I just give an image and let the training handle everything else?

        ###### Option 1: Just return the asked image
        # img = self.images[idx]
        # if self.transforms:
        #     img = self.transforms(img)
        # return img
    
        ###### Option 2: Generate the input as well using the idx parameter as scale factor
        # scale_factor = idx % 4 + 1
        # img = self.images[random.randint(0, len(self.images))]

        # # Crop random s48xs48 patch of the image
        # patch_size = scale_factor * 48
        # col_start = random.randint(0, img.shape[-2] - patch_size)
        # row_start = random.randint(0, img.shape[-1] - patch_size)
        # img = img[:, col_start:col_start+patch_size, row_start:row_start+patch_size].unsqueeze(0)

        # if self.transforms:
        #     img = self.transforms(img)
        
        # # Generate downsampled version of the image
        # lr = torch.nn.functional.interpolate(img, size=(48, 48), mode='bicubic', align_corners=False).squeeze()

        # return lr, img, float(scale_factor)

        ###### Option 3: Get index and scale factor from the idx parameter
        # The scale factor is codified inside the idx and randomly generated
        # Extract scale factor, get the expected image index and proceed as option2
        scale_factor = idx % 10 + 1
        img_idx = idx // 10 
        img = self.images[img_idx]

        # Crop random s48xs48 patch of the image
        patch_size = scale_factor * 48
        col_start = random.randint(0, img.shape[-2] - patch_size)
        row_start = random.randint(0, img.shape[-1] - patch_size)
        img = img[:, col_start:col_start+patch_size, row_start:row_start+patch_size].unsqueeze(0)

        if self.transforms:
            img = self.transforms(img)
        
        # Generate downsampled version of the image
        lr = torch.nn.functional.interpolate(img, size=48, mode='bicubic', align_corners=False).squeeze()
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
    dataset = DIV2K('/home/msiau/data/tmp/agarciat/DIV2K', phase='valid')
    batch_size = 4
    sampler = ScaleBatchSampler(len(dataset), batch_size)
    dataloader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=4
    )

    for (lr, gt, scale) in tqdm(dataloader):
        pass
