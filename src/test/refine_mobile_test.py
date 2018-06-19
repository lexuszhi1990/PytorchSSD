import time
import torch
from torch.autograd import Variable

from src.symbol.RefineSSD_vgg import build_net
from src.symbol.RefineSSD_mobilenet_v2 import RefineSSDMobileNet


v2 = Variable(torch.randn(1, 3, 320, 320), requires_grad=True)
model = RefineSSDMobileNet(shape=320, num_classes=2, use_refine=True)
y = model(v2)

