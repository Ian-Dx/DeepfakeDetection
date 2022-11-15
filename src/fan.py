from collections import OrderedDict
import torch
import torch.nn as nn
from constrained_conv import ConstrainedConv2d


class FAN(nn.Module):
    def __init__(self, num_classes):
        super(FAN, self).__init__()
        self.constrained_layer = ConstrainedConv2d(1, 3, 5, 1, scaling_rate=100)

        self.hierarchical_feature_extraction = nn.Sequential(
            OrderedDict([
                ("conv0", conv_bn_tanh_pool(3, 96, 7, 2, 3)),
                ("conv1", conv_bn_tanh_pool(96, 64, 5, 1, 2)),
                ("conv2", conv_bn_tanh_pool(64, 64, 5, 1, 2))
            ])
        )

        self.cross_feature_maps_learning = conv_bn_tanh_pool(64, 128, 1, 1, 0, True)

        self.classification_layer = nn.Sequential(
            OrderedDict([
                ("fc0", fc_tanh(128*7*7, 200)),
                ("fc1", fc_tanh(200, 200)),
                ("fc2", fc_tanh(200, num_classes))
            ])
        )

    def forward(self, input):
        x = self.constrained_layer(input)
        x = self.hierarchical_feature_extraction(x)
        x = self.cross_feature_maps_learning(x)
        x = x.view(x.shape[0], -1)
        output = self.classification_layer(x)

        return output


def conv_bn_tanh_pool(in_channels, out_channels, kernel_size, stride, padding, avg_pool=False):
    if avg_pool:
        return nn.Sequential(
            OrderedDict([
                ("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
                ("bn", nn.BatchNorm2d(out_channels)),
                ("tanh", nn.Tanh()),
                ("pool", nn.AvgPool2d(2))
            ])
        )
    return nn.Sequential(
        OrderedDict([
            ("conv", nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)),
            ("bn", nn.BatchNorm2d(out_channels)),
            ("tanh", nn.Tanh()),
            ("pool", nn.MaxPool2d(2))
        ])
    )


def fc_tanh(in_channels, out_channels):
    return nn.Sequential(
        OrderedDict([
            ("fc", nn.Linear(in_channels, out_channels)),
            ("tanh", nn.Tanh())
        ])
    )

