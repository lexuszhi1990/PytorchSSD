# -*- coding: utf-8 -*-
import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import L2Norm
from .mobilenet_v2 import conv_bn, conv_1x1_bn, InvertedResidual

def build_mobile_net_v2(first_channel_num=32, data_dim=3, width_mult=1., n_class=81):

    interverted_residual_setting = [
        # t, c, n, s
        [1, 16, 1, 1],
        [6, 24, 2, 2],
        [6, 32, 3, 2],
        [6, 64, 4, 2],
        [6, 96, 3, 1],
        [6, 160, 3, 2],
        [6, 320, 1, 1],
    ]

    # building first layer
    input_channel = int(first_channel_num * width_mult)
    features = [conv_bn(data_dim, input_channel, stride=2)]
    # building inverted residual blocks
    for t, c, n, s in interverted_residual_setting:
        output_channel = int(c * width_mult)
        for i in range(n):
            if i == 0:
                features.append(InvertedResidual(input_channel, output_channel, s, t))
                print(len(features))
            else:
                features.append(InvertedResidual(input_channel, output_channel, 1, t))
            input_channel = output_channel

    return features

class RefineSSDMobileNet(nn.Module):
    def __init__(self, shape, num_classes, use_refine=True, width_mult=1.):
        super(RefineSSDMobileNet, self).__init__()
        self.num_classes = num_classes
        self.shape = shape
        self.base_mbox = 3
        self.use_refine = use_refine

        self.base = nn.ModuleList(build_mobile_net_v2(width_mult=width_mult))
        self.L2Norm_4_3 = L2Norm(512, 10)
        self.L2Norm_5_3 = L2Norm(512, 8)

        self.last_layer_trans = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
            )
        self.extras = nn.Sequential(
                nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
                nn.ReLU(inplace=True)
            )
        self.arm_loc = nn.ModuleList([
                nn.Conv2d(512, 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(1024, 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(512, 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
            ])
        self.arm_conf = nn.ModuleList([
            nn.Conv2d(512, 2*self.base_mbox, kernel_size=3, stride=1, padding=1), \
            nn.Conv2d(512, 2*self.base_mbox, kernel_size=3, stride=1, padding=1), \
            nn.Conv2d(1024, 2*self.base_mbox, kernel_size=3, stride=1, padding=1), \
            nn.Conv2d(512, 2*self.base_mbox, kernel_size=3, stride=1, padding=1), \
        ])

        self.odm_loc = nn.ModuleList([
            nn.Conv2d(256, 4*self.base_mbox, kernel_size=3, stride=1, padding=1), \
            nn.Conv2d(256, 4*self.base_mbox, kernel_size=3, stride=1, padding=1), \
            nn.Conv2d(256, 4*self.base_mbox, kernel_size=3, stride=1, padding=1), \
            nn.Conv2d(256, 4*self.base_mbox, kernel_size=3, stride=1, padding=1), \
        ])
        self.odm_conf = nn.ModuleList([
            nn.Conv2d(256, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1), \
            nn.Conv2d(256, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1), \
            nn.Conv2d(256, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1), \
            nn.Conv2d(256, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1), \
        ])
        self.trans_layers = nn.ModuleList([
            nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
            nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)),
        ])
        self.up_layers = nn.ModuleList([
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2, padding=0),
        ])
        self.latent_layrs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        ])

    def forward(self, x):
        arm_sources = list()
        arm_loc_list = list()
        arm_conf_list = list()
        obm_loc_list = list()
        obm_conf_list = list()
        obm_sources = list()

        print(len(self.base))
        import pdb
        pdb.set_trace()

