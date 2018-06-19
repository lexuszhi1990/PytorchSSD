# -*- coding: utf-8 -*-

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.loss import L2Norm



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
                print(len(features)-1)
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

class MobileNetV2Base(nn.Module):
    def __init__(self, width_mult=1., data_dim=3, first_channel_num=32):
        super(MobileNetV2Base, self).__init__()

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

        self.features = nn.ModuleList(features)

    def forward(self, x, steps=[]):
        arm_sources = []
        for k in range(len(self.features)):
            x = self.features[k](x)
            if k in steps:
                arm_sources.append(x)

        return x, arm_sources

    def initialize_weights(self):
        for m in self.modules():
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


class RefineSSDMobileNet(nn.Module):
    def __init__(self, shape, num_classes, use_refine=True, width_mult=1.):
        super(RefineSSDMobileNet, self).__init__()
        self.num_classes = num_classes
        self.shape = shape
        self.base_mbox = 3
        self.base_channel_num = 64
        self.use_refine = use_refine

        self.base = MobileNetV2Base(width_mult=width_mult)

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
                nn.Conv2d(320, 512, kernel_size=1, stride=1, padding=0),
                nn.ReLU6(inplace=True),
                nn.Conv2d(512, self.base_channel_num, kernel_size=3, stride=2, padding=1),
                nn.ReLU6(inplace=True)
            )
        self.trans_layers = nn.ModuleList([
                nn.Sequential(nn.Conv2d(24, self.base_channel_num, kernel_size=3, stride=1, padding=1),
                nn.ReLU6(inplace=True),
                nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1)),

                nn.Sequential(nn.Conv2d(32, self.base_channel_num, kernel_size=3, stride=1, padding=1),
                nn.ReLU6(inplace=True),
                nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1)),

                nn.Sequential(nn.Conv2d(96, self.base_channel_num, kernel_size=3, stride=1, padding=1),
                nn.ReLU6(inplace=True),
                nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1)),

                nn.Sequential(nn.Conv2d(320, self.base_channel_num, kernel_size=3, stride=1, padding=1),
                nn.ReLU6(inplace=True),
                nn.Conv2d(self.base_channel_num, self.base_channel_num, kernel_size=3, stride=1, padding=1)),
            ])
        self.up_layers = nn.ModuleList([
                nn.ConvTranspose2d(self.base_channel_num, self.base_channel_num, kernel_size=2, stride=2, padding=0),
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
        for sq in self.base.mudules():
            import pdb
            pdb.set_trace()
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


    def forward(self, x):
        arm_sources = list()
        arm_loc_list = list()
        arm_conf_list = list()
        obm_loc_list = list()
        obm_conf_list = list()
        obm_sources = list()
        trans_layer_list = list()

        # 1, 2, 4, 7, 11, 14, 17,
        base_output, arm_sources = self.base(x, [2, 4, 11, 17])
        output = self.appended_layer(base_output)

        print([x.shape for x in arm_sources])
        for (a_s, t_l) in zip(arm_sources, self.trans_layers):
            trans_layer_list.append(t_l(a_s))
        trans_layer_list.reverse()

        print([x.shape for x in trans_layer_list])
        for (t_l, u_l, l_l) in zip(trans_layer_list, self.up_layers, self.latent_layrs):
            output = F.relu6(l_l(F.relu6(u_l(output) + t_l, inplace=True)), inplace=True)
            obm_sources.append(output)

        print([x.shape for x in obm_sources])
        obm_sources.reverse()
        for (x, op_loc, op_cls) in zip(obm_sources, self.odm_loc, self.odm_conf):
            obm_loc_list.append(op_loc(x).permute(0, 2, 3, 1).contiguous())
            obm_conf_list.append(op_cls(x).permute(0, 2, 3, 1).contiguous())
        obm_loc = torch.cat([o.view(o.size(0), -1) for o in obm_loc_list], 1)
        obm_conf = torch.cat([o.view(o.size(0), -1) for o in obm_conf_list], 1)

        output = (
            obm_loc.view(obm_loc.size(0), -1, 4),  # loc preds
            obm_conf.view(obm_conf.size(0), -1, self.num_classes)
            # nn.Softmax()(obm_conf.view(-1, self.num_classes)),  # conf preds
            # nn.Softmax()(obm_conf.view(obm_conf.size(0), -1, self.num_classes))
        )

        return output
