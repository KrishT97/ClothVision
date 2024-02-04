from pathlib import Path

from torchvision import transforms
from torchvision.io import read_image
from torchvision.transforms import InterpolationMode
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from collections import OrderedDict
import joblib

from LabelManagement.LabelEncoder import LabelEncoder
from models.Unet import Unet
from CustomClothingDataset import CustomClothingDataset
from trainers.ModelTrainer import ModelTrainer
from trainers.TransferLearningTrainer import TransferLearningTrainer
from EarlyStopper import EarlyStopper
from models.Resnet101 import Resnet101
from models.MobileNetV3 import MobileNetV3
from models.ResNet50 import ResNet50
from LabelManagement.LabelDecoder import LabelDecoder

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

BATCH_SIZE = 8
EPOCHS = 32
DATASET_DIR = './data/'
MODELS_DIR = 'models/'
RESULTS_DIR = "./results/"
LEARNING_RATE = 0.01
IMAGE_SIZE = (496, 304)


def preprocess_data():
    Path("./data/labels/processed_pixel_labels").mkdir(parents=True, exist_ok=True)
    LabelEncoder("./data/labels/pixel_level_labels_colored", "./data/labels/processed_pixel_labels/",
                 "./data/class_dict.csv", "./one_hot_encoder.joblib").preprocess()


def train():
    device = get_device()
    train_loader, val_loader, test_loader = define_data_loaders()
    loss_fn = torch.nn.CrossEntropyLoss()
    unet_trainer = train_unet(device, loss_fn, test_loader, train_loader, val_loader)
    save_results(unet_trainer, "unet")


def get_encoder(classes):
    if not encoder_exists():
        encoder = OneHotEncoder(sparse=False)
        encoder.fit_transform(classes["class_name"].values.reshape(-1, 1))
        joblib.dump(encoder, "one_hot_encoder.joblib")
        return encoder
    return joblib.load('one_hot_encoder.joblib')


def encoder_exists():
    return os.path.isfile("./one_hot_encoder.joblib")


def build_metadata_df():
    classes = pd.read_csv(DATASET_DIR + "/class_dict.csv")
    encoder = get_encoder(classes)
    codes = encoder.fit_transform(classes["class_name"].values.reshape(-1, 1)).astype(np.uint8)
    encoded_df = pd.concat([classes, pd.Series(list(codes[:]))], axis=1).rename(columns={0: "One-hot"}, inplace=False)
    encoded_df["array_value"] = encoded_df[["r", "g", "b"]].apply(lambda x: np.array([x.r, x.g, x.b], dtype=np.uint8),
                                                                  axis=1)
    encoded_df["tensor_value"] = encoded_df[["r", "g", "b"]].apply(
        lambda x: torch.tensor([x.r, x.g, x.b], dtype=torch.uint8), axis=1)
    return encoded_df


def get_encoder_decoder_dicts(encoded_df):
    encoded_df["byte_value"] = encoded_df["array_value"].apply(lambda x: x.tobytes())
    array_to_one_hot = encoded_df[["byte_value", "One-hot"]].set_index('byte_value').to_dict()["One-hot"]
    return array_to_one_hot, {v.tobytes(): k for k, v in array_to_one_hot.items()}


def define_data_loaders():
    dataset = define_dataset()
    test_dataset, train_dataset, val_dataset = divide_dataset(dataset)
    return (DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)), (
        DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)), (
        DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False))


def divide_dataset(dataset):
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    return test_dataset, train_dataset, val_dataset


def define_dataset():
    dataset = CustomClothingDataset(root_dir=DATASET_DIR,
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
    return dataset


def save_results(trainer, model_name):
    torch.save(trainer.model, RESULTS_DIR + "/" + model_name + ".pth")
    trainer.result.to_csv(RESULTS_DIR + "/" + model_name + ".csv", index=False)


def train_unet(device, loss_fn, test_loader, train_loader, val_loader):
    model = Unet().to(device)
    unet_trainer = train_model(device, loss_fn, model, test_loader, train_loader, val_loader)
    return unet_trainer


def train_model(device, loss_fn, model, test_loader, train_loader, val_loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1, verbose=True)
    trainer = ModelTrainer(model, optimizer, loss_fn, EPOCHS, EarlyStopper(patience=7), device, scheduler)
    trainer.train_and_val(train_loader, val_loader)
    trainer.draw_results()
    trainer.test(test_loader)
    return trainer


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
