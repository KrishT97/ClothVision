from torch import nn as nn
from torchvision.models import segmentation as segmentation


class MobileNetV3(nn.Module):
    def __init__(self, num_classes):
        super(MobileNetV3, self).__init__()
        # Large en el sentido de complejidad computacional

        self.model = segmentation.deeplabv3_mobilenet_v3_large(pretrained=True, progress=True)

        self.model.classifier[-4] = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.classifier[-3] = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.classifier[-2] = nn.ReLU()
        self.model.classifier[-1] = nn.Conv2d(256, num_classes, kernel_size=(1, 1), stride=(1, 1))


    def forward(self, x):
        return self.model(x)

