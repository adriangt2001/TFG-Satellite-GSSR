import torch
from skimage.metrics import structural_similarity as ssim
from DISTS_pytorch import DISTS
import lpips

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
    def __init__(self, data_range=1.0):
        super().__init__('PSNR')
        self.data_range = data_range

    def __call__(self, img1, img2):
        img1, img2 = super().__call__(img1, img2)
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            single_value = 1000
            self.value += single_value
        else:
            single_value = 10 * torch.log10((torch.tensor(self.data_range ** 2)) / mse).item()
            self.value += single_value
        self.num_calls += 1
        return single_value

class SSIM(Metric):
    def __init__(self, data_range=1.0):
        super().__init__('SSIM')
        self.data_range = data_range

    def __call__(self, img1: torch.Tensor, img2: torch.Tensor):
        if img1.is_cuda:
            img1 = img1.cpu()
        if img2.is_cuda:
            img2 = img2.cpu()
        
        img1_np = img1.detach().numpy()
        img2_np = img2.detach().numpy()

        if len(img1_np.shape) == 4:
            batch_size = img1_np.shape[0]
            batch_ssim = 0.0
            for i in range(batch_size):
                # Move channels to last dimension for skimage
                batch_ssim += ssim(img1_np[i], img2_np[i], 
                                  data_range=self.data_range,
                                  channel_axis=0)
            single_value = batch_ssim / batch_size
        elif len(img1_np.shape) == 3:
            single_value = ssim(img1_np, img2_np, 
                               data_range=self.data_range,
                               channel_axis=0)
        else:  # height, width (grayscale)
            single_value = ssim(img1_np, img2_np, 
                               data_range=self.data_range)
        
        self.value += single_value
        self.num_calls += 1
        return single_value

class CustomDists(Metric):
    def __init__(self):
        super().__init__('DISTS')
        self.d = DISTS()
    
    def __call__(self, img1, img2):
        img1, img2 = super().__call__(img1, img2)
        single_value = max(self.d(img1, img2, batch_average=True), 0)
        self.value += single_value
        self.num_calls += 1
        return single_value

class CustomLPIPS(Metric):
    def __init__(self, net='alex'):
        super().__init__('LPIPS')
        self.lpips = lpips.LPIPS(net=net)
    
    def __call__(self, img1, img2):
        img1, img2 = super().__call__(img1, img2)

        img1 = img1 * 2 - 1
        img2 = img2 * 2 - 1

        if len(img1.shape) == 3:
            img1 = img1.unsqueeze(0)
            img2 = img2.unsqueeze(0)

        single_value = self.lpips(img1, img2).mean().item()
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

    def __call__(self, img1, img2):
        for metric in self.metrics:
            metric(img1, img2)
        return self.metrics

    def __str__(self):
        return '\n'.join([str(metric) for metric in self.metrics])

if __name__ == '__main__':
    print("Metrics")
    D = DISTS()
    metric = MetricsList(PSNR(), SSIM(), CustomDists(), CustomLPIPS())
    img1 = torch.rand(5, 3, 256, 256)
    img2 = torch.rand(5, 3, 256, 256)
    metric(img1, img1)
    print(metric)