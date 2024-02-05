import os

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from models.MobileNetV3 import MobileNetV3
from models.ResNet101 import ResNet101
from models.ResNet50 import ResNet50
from models.Unet import Unet
from train_utils.CustomClothingDataset import CustomClothingDataset
from train_utils.EarlyStopper import EarlyStopper
from trainers.ModelTrainer import ModelTrainer

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

BATCH_SIZE = 8
EPOCHS = 32
DATASET_DIR = './data/'
MODELS_DIR = 'models/'
RESULTS_DIR = "./results/"
LEARNING_RATE = 0.01
IMAGE_SIZE = (496, 304)


def train():
    device = get_device()
    train_loader, val_loader, test_loader = define_data_loaders()
    loss_fn = torch.nn.CrossEntropyLoss()
    unet_trainer = get_unet_trainer(device, loss_fn, test_loader, train_loader, val_loader)
    save_results(unet_trainer, "unet")


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def define_data_loaders():
    dataset = define_dataset()
    test_dataset, train_dataset, val_dataset = divide_dataset(dataset)
    return (DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)), (
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)), (
        DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False))


def define_dataset():
    return CustomClothingDataset(root_dir=DATASET_DIR,
                                 transform_input=transforms.Compose([
                                     transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR,
                                                       antialias=True),
                                     transforms.Lambda(lambda x: x / 255),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),
                                 ]),
                                 transform_labels=transforms.Compose([
                                     transforms.Resize(IMAGE_SIZE,
                                                       interpolation=InterpolationMode.NEAREST_EXACT),
                                 ]))


def divide_dataset(dataset):
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return test_dataset, train_dataset, val_dataset


def get_unet_trainer(device, loss_fn, test_loader, train_loader, val_loader):
    return train_model(device, loss_fn, Unet().to(device), test_loader, train_loader, val_loader)


def get_mobile_net_v3_trainer(device, loss_fn, test_loader, train_loader, val_loader):
    return train_model(device, loss_fn, MobileNetV3(59).to(device), test_loader, train_loader, val_loader)


def get_resnet_50__trainer(device, loss_fn, test_loader, train_loader, val_loader):
    return train_model(device, loss_fn, ResNet50(59).to(device), test_loader, train_loader, val_loader)


def get_resnet_101_trainer(device, loss_fn, test_loader, train_loader, val_loader):
    return train_model(device, loss_fn, ResNet101(59).to(device), test_loader, train_loader, val_loader)


def train_model(device, loss_fn, model, test_loader, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
    trainer = ModelTrainer(model, optimizer, loss_fn, EPOCHS, EarlyStopper(patience=7), device, scheduler)
    trainer.train_and_val(train_loader, val_loader)
    trainer.draw_results()
    trainer.test(test_loader)
    return trainer


def save_results(trainer, model_name):
    torch.save(trainer.model, RESULTS_DIR + "/" + model_name + ".pth")
    trainer.result.to_csv(RESULTS_DIR + "/" + model_name + ".csv", index=False)
