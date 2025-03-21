import os
import torch
import torchvision.io as io

import random

class DIV2K(torch.utils.data.Dataset):
    def __init__(self, path, phase='train', transforms=None):
        self.path = path
        self.phase = phase
        self.transforms = transforms

        self.load_data()

    
    def load_data(self):
        HR_path = os.path.join(self.path, f'DIV2K_{self.phase}_HR')
        LR_bicubic_X2_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_bicubic', 'X2')
        LR_bicubic_X3_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_bicubic', 'X3')
        LR_bicubic_X4_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_bicubic', 'X4')
        LR_unknown_X2_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_unknown', 'X2')
        LR_unknown_X3_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_unknown', 'X3')
        LR_unknown_X4_path = os.path.join(self.path, f'DIV2K_{self.phase}_LR_unknown', 'X4')

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
        return len(self.data_HR)
    
    def __getitem__(self, idx):
        data_part = random.choice(range(6))
        scale = data_part % 3 + 2
        
        gt = self.data_HR[idx]
        lr = self.data_LR[data_part][idx]

        if self.transforms:
            lr, gt = self.transforms(lr, gt)
        
        return lr, gt, torch.tensor([scale], dtype=torch.int32)

if __name__ == '__main__':
    dataset = DIV2K('/home/msiau/data/tmp/agarciat/DIV2K_processed', phase='train')
    print(len(dataset))
