import os
# os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES'] = '6'

import argparse

import torch

from data import Sen2Dataset
from models import HFN_Default

class Sen2Normalization(object):
    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image / 10000.0


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str, help="Root path to the dataset.")
    parser.add_argument('weights20', type=str, help="Weights path for HFN_20.")
    parser.add_argument('weights60', type=str, help="Weights path for HFN_60.")

    return parser.parse_args()

@torch.no_grad()
def inference(model20, model60, device, args):
    model20.eval()
    model60.eval()

    for subset in os.listdir(args.dataset):
        set_path = os.path.join(args.dataset, subset)
        processed_path = os.path.join(set_path, 'processed_data')

        rasters = list(filter(lambda x: 'Raster' in x, os.listdir(set_path)))
        for raster in rasters:
            raster_path = os.path.join(set_path, raster)

            dataset = Sen2Dataset(
                raster_path,
                [10, 20, 60],
                60,
                transform=Sen2Normalization()
            )

            for x, _ in dataset:
                # 20m to 10m
                y_pred_20 = model20([x[0].unsqueeze(0).to(device), x[1].unsqueeze(0).to(device)])
                x1 = [torch.cat([x[0].to(device), y_pred_20.squeeze()], dim=0), x[2]]

                # 60m to 10m
                y_pred_60 = model60([x1[0].unsqueeze(0).to(device), x1[1].unsqueeze(0).to(device)])
                x2 = torch.cat([x1[0], y_pred_60.squeeze()], dim=0)

                name_img = f"{(len(os.listdir(processed_path))):04}.pt"

                torch.save(x2.squeeze().cpu(), os.path.join(processed_path, name_img))

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args = parse_args()

    # Model
    model20 = HFN_Default(
        6,
        4
    ).to(device)
    model20.load_state_dict(torch.load(args.weights20))

    model60 = HFN_Default(
        2,
        4+6
    ).to(device)
    model60.load_state_dict(torch.load(args.weights60))

    

if __name__ == '__main__':
    device = torch.cuda.is_available()
    
    main()