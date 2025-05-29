import torch
from pytorch_msssim import SSIM

class Metric:
    def __init__(self, name):
        self.name = name
        self.value = 0.0
        self.num_calls = 0

    def get_value(self):
        return self.value / self.num_calls
    
    def reset(self):
        self.value = 0.0
        self.num_calls = 0
    
    def __call__(self, img1, img2):
        if not torch.is_tensor(img1):
            img1 = torch.tensor(img1)
        if not torch.is_tensor(img2):
            img2 = torch.tensor(img2)
        return img1, img2
    
    def __str__(self):
        return f"{self.name}: {self.get_value():.2f}"

class PSNR(Metric):
    def __init__(self):
        super().__init__('PSNR')

    @torch.no_grad()
    def __call__(self, img1, img2):
        img1, img2 = super().__call__(img1, img2)
        data_range = img2.max() - img2.min()
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            single_value = 1000
            self.value += single_value
        else:
            single_value = 10 * torch.log10((torch.tensor(data_range ** 2)) / mse).item()
            self.value += single_value
        self.num_calls += 1
        return single_value

class CustomSSIM(Metric):
    def __init__(self, channels):
        super().__init__('SSIM')
        self.channels = channels

    @torch.no_grad()
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor):
        img1, img2 = super().__call__(img1, img2)

        data_range = img2.max() - img2.min()
        single_value = SSIM(data_range=data_range, size_average=True, channel=self.channels)(img1, img2).item()
        self.value += single_value
        self.num_calls += 1
        return single_value

class CustomDists(Metric):
    def __init__(self, dists):
        super().__init__('DISTS')
        self.d = dists.eval()
    
    @torch.no_grad()
    def __call__(self, img1, img2):
        img1, img2 = super().__call__(img1, img2)

        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        single_value = 0
        single_value = max(self.d(img1[:, :3], img2[:, :3], batch_average=True).item(), 0)
        self.value += single_value
        self.num_calls += 1
        return single_value

class CustomLPIPS(Metric):
    def __init__(self, my_lpips):
        super().__init__('LPIPS')
        self.lpips = my_lpips.eval()
    
    @torch.no_grad()
    def __call__(self, img1, img2):
        img1, img2 = super().__call__(img1, img2)

        # Convert to range expected by LPIPS
        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1

        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        single_value = self.lpips(img1[:, :3], img2[:, :3]).mean().item()
        self.value += single_value
        self.num_calls += 1
        return single_value

class MetricsList:
    def __init__(self, *metrics):
        self.metrics = metrics

    def get_values(self):
        return [metric.get_value() for metric in self.metrics]

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def __getitem__(self, idx):
        return self.metrics[idx]

    def __call__(self, img1, img2):
        for metric in self.metrics:
            metric(img1, img2)
        return self.metrics

    def __str__(self):
        return '\n'.join([str(metric) for metric in self.metrics])

if __name__ == '__main__':
    import os
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '6'
    print("Metrics")
    metric = MetricsList(PSNR(), CustomSSIM(), CustomDists(device='cuda'), CustomLPIPS(device='cuda'))
    metric2 = MetricsList(PSNR(), CustomSSIM(), CustomDists(device='cuda'), CustomLPIPS(device='cuda'))
    img1 = torch.rand(5, 3, 256, 256, device='cuda')
    img2 = torch.rand(5, 3, 256, 256, device='cuda')
    metric(img1, img2)
    print(metric)