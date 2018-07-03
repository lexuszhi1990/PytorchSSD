import sys
sys.path.append('.')

import torch
from torch.autograd import Variable
from src.symbol.RefineSSD_ResNeXt import RefineSSDSEResNeXt
from src.prior_box import PriorBox
from src.config import config

cfg = config.list['v6']

# 19:6
# 4:3

model = RefineSSDSEResNeXt(use_refine=True)
# model.initialize_weights()
v2 = Variable(torch.randn(1, 3, 320, 320), requires_grad=True)
# v2 = Variable(torch.randn(1, 3, 1920, 1080), requires_grad=True)
# v2 = Variable(torch.randn(1, 3, 960, 540), requires_grad=True)
# v2 = Variable(torch.randn(1, 3, 480, 270), requires_grad=True)
# v2 = Variable(torch.randn(1, 3, 512, 288), requires_grad=True)
# v2 = Variable(torch.randn(1, 3, 128*4, 128*3), requires_grad=True)
y = model(v2)

