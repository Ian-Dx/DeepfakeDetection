import torch.nn as nn
import torch
import numpy as np


class ConstrainedConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            scaling_rate: int = 1
    ):
        super(ConstrainedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )

        self.scaling_rate = scaling_rate
        self.center = kernel_size // 2

        self.in_channels = in_channels
        constrained_weight = self.get_constrained_weight()
        self.weight = nn.Parameter(constrained_weight, requires_grad=True)


    def forward(self, input):
        if torch.cuda.is_available():
            constrained_weight = self.get_constrained_weight().cuda()
        else:
            constrained_weight = self.get_constrained_weight()

        self.weight = nn.Parameter(constrained_weight, requires_grad=True)

        return self._conv_forward(input, self.weight)



    def get_constrained_weight(self):
        return torch.from_numpy(
            np.array(
                [
                    [
                        self.constrain_weight(in_filter_weight)
                        for in_filter_weight in out_filter_weight
                    ]
                    for out_filter_weight in self.weight
                ]
            )
        )


    def constrain_weight(self, in_filter_weight):
        # *2 是为了归一的时候不出现0
        # if torch.min(in_filter_weight) < 0:
        #     in_filter_weight = in_filter_weight - torch.min(in_filter_weight)

        # 归一化
        weight_total = (torch.sum(in_filter_weight) - in_filter_weight[self.center][self.center]) / self.scaling_rate
        weight = in_filter_weight / weight_total
        weight[self.center][self.center] = -self.scaling_rate * self.in_channels

        if torch.cuda.is_available():
            return weight.cpu().data.numpy()
        else:
            return weight.data.numpy()


class NormalizedConv2d(nn.Conv2d):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size,
            stride=1,
            padding=0,
            dilation=1,
            groups: int = 1,
            bias: bool = True,
            padding_mode: str = 'zeros',
            scaling_rate: int = 1
    ):
        super(NormalizedConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode
        )

        self.scaling_rate = scaling_rate
        self.center = kernel_size // 2

        normalized_weight = self.get_normalized_weight()
        self.weight = nn.Parameter(normalized_weight, requires_grad=True)


    def forward(self, input):
        if torch.cuda.is_available():
            normalized_weight = self.get_normalized_weight().cuda()
        else:
            normalized_weight = self.get_normalized_weight()

        self.weight = nn.Parameter(normalized_weight , requires_grad=True)

        return self._conv_forward(input, self.weight)

    def normalized_weight(self, in_filter_weight):
        # *2 是为了归一的时候不出现0
        # if torch.min(in_filter_weight) < 0:
        #     in_filter_weight = in_filter_weight - 2 * torch.min(in_filter_weight)

        # 归一化
        weight_total = torch.sum(in_filter_weight) / self.scaling_rate
        weight = in_filter_weight / weight_total

        if torch.cuda.is_available():
            return weight.cpu().data.numpy()
        else:
            return weight.data.numpy()


    def get_normalized_weight(self):
        return torch.from_numpy(
            np.array(
                [
                    [
                        self.normalized_weight(in_filter_weight)
                        for in_filter_weight in out_filter_weight
                    ]
                    for out_filter_weight in self.weight
                ]
            )
        )

