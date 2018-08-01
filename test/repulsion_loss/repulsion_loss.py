# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils.box_utils import IoG, decode_new


class RepulsionLoss(nn.Module):

    def __init__(self, variance=[0.1, 0.2], sigma=0.):
        super(RepulsionLoss, self).__init__()
        self.variance = variance
        self.sigma = sigma

    # TODO
    def smoothln(self, x, sigma=0.):
        pass

    def forward(self, loc_data, ground_data, prior_data):

        decoded_boxes = decode_new(loc_data, Variable(prior_data.data, requires_grad=False), self.variance)
        iog = IoG(ground_data, decoded_boxes)
        # sigma = 1
        # loss = torch.sum(-torch.log(1-iog+1e-10))
        # sigma = 0
        loss = torch.sum(iog)
        return loss
