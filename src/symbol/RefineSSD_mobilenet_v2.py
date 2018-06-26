# -*- coding: utf-8 -*-

import os
import math
import logging
from collections import OrderedDict
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import L2Norm
from src.utils import kaiming_weights_init, xavier, load_weights

def build_mobile_net_v2(width_mult=1., data_dim=3, first_channel_num=32):

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
            else:
                features.append(InvertedResidual(input_channel, output_channel, 1, t))
            input_channel = output_channel

    return features


def conv_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
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
            nn.Conv2d(inp, inp * expand_ratio, 1, 1, 0, bias=True),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # dw
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=True),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pw-linear
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=True),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class RefineSSDMobileNet(nn.Module):
    def __init__(self, num_classes=81, width_mult=1., base_channel_num=128, use_refine=True):
        super(RefineSSDMobileNet, self).__init__()
        self.num_classes = num_classes
        self.base_channel_num = base_channel_num
        self.use_refine = use_refine

        self.base_channel_list = [ int(width_mult*num) for num in [32, 64, 160, 320] ]
        self.base_mbox = 3
        self.base = nn.ModuleList(build_mobile_net_v2(width_mult=width_mult))
        if self.use_refine:
            self.arm_loc = nn.ModuleList([
                    nn.Conv2d(self.base_channel_list[0], 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.base_channel_list[1], 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.base_channel_list[2], 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.base_channel_list[3], 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
                ])
            self.arm_conf = nn.ModuleList([
                    nn.Conv2d(self.base_channel_list[0], 2*self.base_mbox, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.base_channel_list[1], 2*self.base_mbox, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.base_channel_list[2], 2*self.base_mbox, kernel_size=3, stride=1, padding=1),
                    nn.Conv2d(self.base_channel_list[3], 2*self.base_mbox, kernel_size=3, stride=1, padding=1),
                ])
        self.odm_loc = nn.ModuleList([
                nn.Conv2d(self.base_channel_num, 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.base_channel_num, 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.base_channel_num, 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.base_channel_num, 4*self.base_mbox, kernel_size=3, stride=1, padding=1),
            ])
        self.odm_conf = nn.ModuleList([
                nn.Conv2d(self.base_channel_num, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.base_channel_num, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.base_channel_num, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.base_channel_num, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1),
            ])

        self.appended_layer = nn.Sequential(
                nn.Conv2d(self.base_channel_list[3], self.base_channel_list[3] * 2, kernel_size=1, stride=1, padding=0),
                nn.ReLU6(inplace=True),
                nn.Conv2d(self.base_channel_list[3] * 2, self.base_channel_list[3], kernel_size=3, stride=2, padding=1),
                nn.ReLU6(inplace=True)
            )
        self.trans_layers = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(self.base_channel_list[0], self.base_channel_num, kernel_size=3, stride=1, padding=1),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1),
                ),

                nn.Sequential(
                    nn.Conv2d(self.base_channel_list[1], self.base_channel_num, kernel_size=3, stride=1, padding=1),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1),
                ),

                nn.Sequential(
                    nn.Conv2d(self.base_channel_list[2], self.base_channel_num, kernel_size=3, stride=1, padding=1),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1),
                ),

                nn.Sequential(
                    nn.Conv2d(self.base_channel_list[3], self.base_channel_num, kernel_size=3, stride=1, padding=1),
                    nn.ReLU6(inplace=True),
                    nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1),
                )
            ])
        self.up_layers = nn.ModuleList([
                nn.ConvTranspose2d(self.base_channel_list[3], self.base_channel_num, kernel_size=3, stride=1, padding=1),
                nn.ConvTranspose2d(self.base_channel_num, self.base_channel_num, kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose2d(self.base_channel_num, self.base_channel_num, kernel_size=2, stride=2, padding=0),
                nn.ConvTranspose2d(self.base_channel_num, self.base_channel_num, kernel_size=2, stride=2, padding=0),
            ])
        self.latent_layrs = nn.ModuleList([
                nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1),
            ])

    def initialize_base_weights(self):
        for sq in self.base.modules():
            for m in sq.modules():
                if isinstance(m, nn.Conv2d):
                    n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                    m.weight.data.normal_(0, math.sqrt(2. / n))
                    if m.bias is not None:
                        m.bias.data.zero_()
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()
                elif isinstance(m, nn.Linear):
                    n = m.weight.size(1)
                    m.weight.data.normal_(0, 0.01)
                    m.bias.data.zero_()

    def initialize_weights(self, ckpt_path=None):
        if ckpt_path and Path(ckpt_path).exists():
            state_dict = torch.load(ckpt_path)
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                head = k[:7]
                if head == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v
            self.load_state_dict(new_state_dict)
            logging.info("load weights from %s" % ckpt_path)
        else:
            self.initialize_base_weights()
            self.appended_layer.apply(kaiming_weights_init)
            self.trans_layers.apply(kaiming_weights_init)
            self.up_layers.apply(kaiming_weights_init)
            self.latent_layrs.apply(kaiming_weights_init)
            self.odm_loc.apply(kaiming_weights_init)
            self.odm_conf.apply(kaiming_weights_init)
            if self.use_refine:
                self.arm_loc.apply(kaiming_weights_init)
                self.arm_conf.apply(kaiming_weights_init)

    def base_forward(self, x, steps=[]):
        arm_sources = []
        for k in range(len(self.base)):
            x = self.base[k](x)
            # print(x.shape)
            if k in steps:
                arm_sources.append(x)

        return x, arm_sources

    def forward(self, x, inference=False):
        arm_sources = list()
        arm_loc_list = list()
        arm_conf_list = list()
        obm_loc_list = list()
        obm_conf_list = list()
        obm_sources = list()
        trans_layer_list = list()
        arm_loc_result, arm_conf_result, obm_loc_result, obm_conf_result = None, None, None, None

        # 1, 2, 4, 7, 11, 14, 17,
        base_output, arm_sources = self.base_forward(x, [4, 7, 14])
        output = self.appended_layer(base_output)
        arm_sources.append(output)
        # print([x.shape for x in arm_sources])

        if self.use_refine:
            for (a_s, a_l, a_c) in zip(arm_sources, self.arm_loc, self.arm_conf):
                arm_loc_list.append(a_l(a_s).permute(0, 2, 3, 1).contiguous())
                arm_conf_list.append(a_c(a_s).permute(0, 2, 3, 1).contiguous())
            arm_loc_temp = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)
            arm_conf_temp = torch.cat([o.view(o.size(0), -1) for o in arm_conf_list], 1)
            arm_loc_result = arm_loc_temp.view(arm_loc_temp.size(0), -1, 4)
            if inference:
                arm_conf_result = nn.Softmax(-1)(arm_conf_temp.view(-1, 2))
            else:
                arm_conf_result = arm_conf_temp.view(arm_conf_temp.size(0), -1, 2)

        for (a_s, t_l) in zip(arm_sources, self.trans_layers):
            trans_layer_list.append(t_l(a_s))
        trans_layer_list.reverse()
        # print([x.shape for x in trans_layer_list])

        for (t_l, u_l, l_l) in zip(trans_layer_list, self.up_layers, self.latent_layrs):
            output = F.relu6(l_l(F.relu6(u_l(output) + t_l, inplace=True)), inplace=True)
            obm_sources.append(output)
        # print([x.shape for x in obm_sources])

        obm_sources.reverse()
        for (x, op_loc, op_cls) in zip(obm_sources, self.odm_loc, self.odm_conf):
            obm_loc_list.append(op_loc(x).permute(0, 2, 3, 1).contiguous())
            obm_conf_list.append(op_cls(x).permute(0, 2, 3, 1).contiguous())
        obm_loc = torch.cat([o.view(o.size(0), -1) for o in obm_loc_list], 1)
        obm_conf = torch.cat([o.view(o.size(0), -1) for o in obm_conf_list], 1)

        obm_loc_result = obm_loc.view(obm_loc.size(0), -1, 4)
        if inference:
            obm_conf_result = nn.Softmax(-1)(obm_conf.view(obm_conf.size(0), -1, self.num_classes))
        else:
            obm_conf_result = obm_conf.view(obm_conf.size(0), -1, self.num_classes)

        return (arm_loc_result, arm_conf_result, obm_loc_result, obm_conf_result)
