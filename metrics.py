import torch

class PSNR:
    def __init__(self, data_range=1.0):
        self.name = 'PSNR'
        self.value = 0.0
        self.num_calls = 0
        self.data_range = data_range
    
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
        
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            single_value = 1000
            self.value += single_value
        else:
            single_value = 10 * torch.log10((torch.tensor(self.data_range ** 2)) / mse).item()
            self.value += single_value
        self.num_calls += 1
        return single_value

    def __str__(self):
        return f"{self.name}: {self.value/self.num_calls:.2f}"

class Metric:
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
    metric = Metric()
    img1 = torch.rand(3, 256, 256)
    img2 = torch.rand(3, 256, 256)
    metric(img1, img2)
    print(metric)