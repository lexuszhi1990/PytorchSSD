import sys
sys.path.append('.')

import torch
from torch.autograd import Variable
from src.symbol.RefineSSD_mobilenet_v2 import RefineSSDMobileNet
from src.symbol.RefineSSD_mobilenet_v2_1 import RefineSSDMobileNetV1
from src.prior_box import PriorBox
from src.config import config

cfg = config.list['v6']

# 19:6
# 4:3

model = RefineSSDMobileNetV1(num_classes=81, base_channel_num=128, width_mult=1, use_refine=True)
model.initialize_weights()
# v2 = Variable(torch.randn(1, 3, 960, 540), requires_grad=True)
v2 = Variable(torch.randn(1, 3, 512, 288), requires_grad=True)
# v2 = Variable(torch.randn(1, 3, 128*4, 128*3), requires_grad=True)
y = model(v2)

