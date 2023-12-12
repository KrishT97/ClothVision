import os

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode, transforms
from numpy import int64
from PIL import Image
import random

import LabelDecoder


class CustomClothingDataset(Dataset):
    def __init__(self, root_dir, input_dir='images/colored_labels',
                 labels_dir='labels/processed_pixel_labels',
                 transform_input=transforms.Compose([
                     transforms.Resize((816, 576), interpolation=InterpolationMode.NEAREST_EXACT),
                     # size mayor de las imagenes, para todas iguales sin perder información, esta ligeramente reducido para ser divisible por 16 (necesario para U-net
                     transforms.Lambda(lambda x: x / 255),  # reescalo de 0 a 1
                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # normalizo
                 ]),
                 transform_labels=transforms.Compose([
                     transforms.Resize((816, 576), interpolation=InterpolationMode.NEAREST_EXACT),
                 ]),
                 transform_augmentation=transforms.RandomHorizontalFlip(p=1)
                 ):

        self.root_dir = root_dir
        self.input_dir = input_dir
        self.labels_dir = labels_dir
        self.transform_input = transform_input
        self.transform_labels = transform_labels
        self.transform_augmentation = transform_augmentation
        self.image_files = os.listdir(os.path.join(root_dir, self.input_dir))
        self.label_files = os.listdir(os.path.join(root_dir, self.labels_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        input = read_image(os.path.join(self.root_dir, self.input_dir, self.image_files[idx]))
        label = torch.load(os.path.join(self.root_dir, self.labels_dir, self.label_files[idx])).permute(2, 0,
                                                                                                        1)  # el preprocesado de labels se llevo a cabo anteriormente
        if self.transform_input:
            input = self.transform_input(input)
        if self.transform_labels:
            label = self.transform_labels(label)
        if self.transform_augmentation and random.choice([True, False]):
            return self.transform_augmentation(input), self.transform_augmentation(label)
        return input, label

dataloader = DataLoader(CustomClothingDataset("./data/"), batch_size=1, shuffle=True)

class_counts = {}
decoder = LabelDecoder.LabelDecoder("./data/class_dict.csv", "one_hot_encoder.joblib")
for x,y in dataloader:
    clases = decoder.decode_labels(y[0])
    for label in clases[0][1]:
        if label in class_counts:
            class_counts[label] += 1
        else:
            class_counts[label] = 1

print(class_counts)
