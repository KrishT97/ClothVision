import torchvision.models.segmentation as segmentation
import torch.nn as nn


class ResNet50Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50Segmentation, self).__init__()
        self.model = segmentation.deeplabv3_resnet50(pretrained=True, progress=True)
        # Ajustar la penúltima capa si se quisiera(fc) para que sea "entrenable", el in_features se extrae
        # in_features = self.model.classifier[-2].in_features
        # self.model.classifier[-2] = nn.Conv2d(in_features, num_classes, kernel_size=1)
        # se podría coger in_features_last = self.model.classifier[-1].in_features y meterlo donde pone 256,
        # lo mismo es
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)


class MobileNetV3Segmentation(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3Segmentation, self).__init__()
        # Large en el sentido de complejidad computacional
        self.model = segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)
        # Ajustar la penúltima capa si se quisiera tambien
        # in_features = self.model.classifier[-2].in_channels
        # self.model.classifier[-2] = nn.Conv2d(in_features, num_classes, kernel_size=1)

        # 1280 canales entrada predefinidos en la última capa, canales especificadas por clases como salida

        self.model.classifier[-1] = nn.Conv2d(1280, num_classes, kernel_size=1)

    def forward(self, x):
        return self.model(x)
