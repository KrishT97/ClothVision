import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.transforms import InterpolationMode

from LabelManagement.LabelDecoder import LabelDecoder
from models.Unet import Unet
from torchvision import transforms

IMAGE_SIZE = (496, 304)


def test_with_image(image_file):
    def proccess_model_output(model, x):
        out = model(x.unsqueeze(0))["out"].argmax(1).squeeze(0)
        return torch.zeros_like(x).to(get_device()).scatter_(0, out.unsqueeze(0), 1.)

    transformation = get_test_transformation()
    predict, classes_predicted = LabelDecoder(class_dict_path="./data/class_dict.csv",
                                              encoder_path='one_hot_encoder.joblib').decode_labels(
        proccess_model_output(load_unet_model(), transformation(image_file).to(get_device())).cpu())
    draw_results(classes_predicted, transformation(image_file), predict)


def load_unet_model():
    model = Unet().to(get_device())
    model.load_state_dict(torch.load("./results/MobileNetV3.pth"))
    return model


def get_test_transformation():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=InterpolationMode.BILINEAR, antialias=True),
        transforms.Lambda(lambda x: x / 255),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def get_device():
    return (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )


def draw_results(classes_predicted, input, predict):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.title("Original")
    undo_transform = get_inverse_transform()
    input = undo_transform(input)
    plt.imshow(np.transpose(input.int(), (1, 2, 0)))
    plt.subplot(1, 3, 2)
    plt.imshow(predict)
    plt.title("Predict")
    print("Predicted classes:", classes_predicted)
    plt.show()


def get_inverse_transform():
    return transforms.Compose([
        transforms.Normalize(
            mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.225],
            std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Lambda(lambda x: x * 255)
    ])
