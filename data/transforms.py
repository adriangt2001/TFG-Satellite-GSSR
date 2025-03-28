import torch
import random

class CustomRandomHorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img1, img2):
        if random.random() < 0.5:
            img1 = img1.flip(-1)
            img2 = img2.flip(-1)
        return img1, img2

class CustomRandomVerticalFlip(object):
    def __init__(self):
        pass

    def __call__(self, img1, img2):
        if random.random() < 0.5:
            img1 = img1.flip(-2)
            img2 = img2.flip(-2)
        return img1, img2

class CustomRandomRotation(object):
    def __init__(self, degrees):
        pass

    def __call__(self, img1, img2):
        if random.random() < 0.5:
            angle = random.choice([1, 2, 3])
            img1 = torch.rot90(img1, angle, [1, 2])
            img2 = torch.rot90(img2, angle, [1, 2])
        return img1, img2
