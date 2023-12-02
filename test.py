import os

import torch

import torch
from torchvision.transforms import InterpolationMode
from torchvision import transforms

image = torch.load(r"C:/Users/Usuario\PycharmProjects\aaiv\ClothingProject\data\labels\processed_pixel_labels/0006.pt")
print(image.shape)
image = image.permute(2,1,0)
t = transforms.Resize((864, 576), interpolation=InterpolationMode.NEAREST)
print(t(image).shape)