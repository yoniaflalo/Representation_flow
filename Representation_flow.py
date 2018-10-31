import torch.nn as nn
from resnet import *
from TVL1OF import *


class Representation_flow(nn.Module):
    def __init__(self, num_classes, use_resnet_50=False, num_iter=20):
        super(Representation_flow, self).__init__()
        self.num_classes = num_classes
        if use_resnet_50:
            self.res_net_start = resnet50_2nd_layer(num_classes)
            self.res_net_end = resnet50_end(num_classes)
            in_channels = 512
        else:
            self.res_net_start = resnet34_2nd_layer(num_classes)
            self.res_net_end = resnet34_end(num_classes)
            in_channels = 128
        self.TVL1OF = TVL1OF(num_iter=num_iter)

        out_channels = 32
        self.reduction_channel = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.expension_channel = nn.Conv2d(in_channels=4 * out_channels, out_channels=in_channels, kernel_size=3)
        self.conv_flow = nn.Conv2d(in_channels=2 * out_channels, out_channels=2 * out_channels, kernel_size=3)

    def forward(self, x1, x2, x3, x4):
        resnet1 = self.res_net_start(x1)
        resnet2 = self.res_net_start(x2)
        resnet3 = self.res_net_start(x3)
        resnet4 = self.res_net_start(x4)
        resnet1 = self.reduction_channel(resnet1)
        resnet2 = self.reduction_channel(resnet2)
        resnet3 = self.reduction_channel(resnet3)
        resnet4 = self.reduction_channel(resnet4)

        flow11 = self.TVL1OF(resnet1, resnet2)
        flow12 = self.TVL1OF(resnet3, resnet4)
        s = flow11.shape
        flow11_ = flow11.view([s[0], s[1] * s[2], s[3], s[4]])
        flow12_ = flow12.view([s[0], s[1] * s[2], s[3], s[4]])
        convflow1 = self.conv_flow(flow11_)
        convflow2 = self.conv_flow(flow12_)
        flow2 = self.TVL1OF(convflow1, convflow2)
        s = flow2.shape
        o = flow2.view([s[0], s[1] * s[2], s[3], s[4]])
        o = self.expension_channel(o)
        o = self.res_net_end(o)

        return o
