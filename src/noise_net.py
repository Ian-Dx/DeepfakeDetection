from collections import OrderedDict

import torch
from dxtorchutils.ImageClassification.models.ResNet import ResNet18
from torch.nn import *
import torch.nn.functional as F
from constrained_conv import NormalizedConv2d


class NoiseNet(Module):
    def __init__(self):
        super(NoiseNet, self).__init__()
        self.down_sample = Sequential(
            OrderedDict([
                ("conv", NormalizedConv2d(1, 64, 3, 1, 1)),
                ("tanh", Tanh()),
                ("bn", BatchNorm2d(64)),
                ("pool", MaxPool2d(2))
            ])
        )

        self.conv = NormalizedConv2d(64, 128, 3, 1, 1)

        self.up_sample = Sequential(
            OrderedDict([
                ("conv", NormalizedConv2d(128, 3, 3, 1, 1)),
                ("tanh", Tanh()),
                ("bn", BatchNorm2d(3)),
            ])
        )

        self.features = ResNet18()


    def forward(self, input):
        h, w = input.shape[-2:]
        # (n, 3, 224, 224)
        x = self.down_sample(input)
        # (n, 64, 112, 112)
        x = self.conv(x)
        # (n, 128, 112, 112)
        x = F.interpolate(x, (h, w), None, "bilinear", True)
        # (n, 128, 224, 224)
        x = self.up_sample(x)
        # (n, 3, 224, 224)
        x = input - x
        # (n, 3, 224, 224)
        output = self.features(x)
        # (n, 1000)

        return output
