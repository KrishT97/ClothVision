from torch import nn as nn
from torchvision.models import segmentation as segmentation


class Resnet101(nn.Module):
    def __init__(self, num_classes):
        super(Resnet101, self).__init__()

        self.model = segmentation.fcn_resnet101(pretrained=True, progress=True)

        self.model.classifier[-5] = nn.Conv2d(2048, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        self.model.classifier[-4] = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.model.classifier[-3] = nn.ReLU()
        self.model.classifier[-2] = nn.Dropout(p=0.1, inplace=False)
        self.model.classifier[-1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))


    def forward(self, x):
        return self.model(x)

print(sum(p.numel() for p in Resnet101(5).parameters() ))
