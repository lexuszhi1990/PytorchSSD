
import torch
import torch.nn as nn
import math


def conv_bn(inp, oup, kernel_size=3, stride=1, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=3, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )

def conv_1x1_bn(inp, oup, kernel_size=1, stride=1, padding=0):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        self.use_res_connect = self.stride == 1 and inp == oup
        self.conv = nn.Sequential(
            # pw
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


v1 = Variable(torch.randn(1, 3, 224, 224), requires_grad=True)
conv_bn_1 = conv_bn(3, 128, stride=3)

from src.symbol.RefineSSD_mobilenet_v2 import MobileNetV2_Base
base = MobileNetV2_Base(width_mult=0.5)
base._initialize_weights_from_scratch()
import torch
from torch.autograd import Variable
v1 = Variable(torch.randn(1, 3, 224, 224), requires_grad=True)

v2 = Variable(torch.randn(1, 3, 320, 320), requires_grad=True)

v2 = Variable(torch.randn(1, 3, 512, 512), requires_grad=True)

import time
start_ = time.time()
y2 = base(v2)
print("forward cost %.3f" % (time.time() - start_))

from src.symbol.RefineSSD_mobilenet_v2 import make_dot
g = make_dot(y.mean())
g.save()
# g = make_dot(y.mean(), params=dict(base.named_parameters()))

