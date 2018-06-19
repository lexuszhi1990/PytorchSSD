import time
import torch
from torch.autograd import Variable

from src.symbol.RefineSSD_vgg import RefineSSD

model = RefineSSD(size=320, num_classes=81, use_refine=True)
# model.base._initialize_weights_from_scratch()
v2 = Variable(torch.randn(1, 3, 320, 320), requires_grad=True)
y = model(v2)

