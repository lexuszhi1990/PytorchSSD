# -*- coding: utf-8 -*-

import math
from collections import OrderedDict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.weight_init import kaiming_weights_init, xavier
import logging

class SEScale(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEScale, self).__init__()
        self.fc1 = nn.Conv2d(channel, reduction, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(reduction, channel, kernel_size=1, padding=0)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.fc1(x)
        x = F.relu(x, inplace=True)
        x = self.fc2(x)
        x = F.sigmoid(x)
        return x


class Bottleneck(nn.Module):
    """
    SE-ResNeXt 50 BottleNeck
    """
    def __init__(self, i_ch, o_ch, stride=1, groups=32):
        """
        :param i_ch: input tensor channels
        :param o_ch: output tensor channels in each stage
        :param stride: the 3x3 kernel in a stage may set to 2
                       only in each group of stages' 1st stage
        """
        super(Bottleneck, self).__init__()
        s_ch = o_ch // 2  # stage channel
        self.conv1 = nn.Conv2d(i_ch, s_ch, kernel_size=1, padding=0, stride=1,      bias=False)
        self.bn1 = nn.BatchNorm2d(s_ch)
        self.conv2 = nn.Conv2d(s_ch, s_ch, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)
        self.bn2 = nn.BatchNorm2d(s_ch)
        self.conv3 = nn.Conv2d(s_ch, o_ch, kernel_size=1, padding=0, stride=1,      bias=False)
        self.bn3 = nn.BatchNorm2d(o_ch)
        self.scale = SEScale(o_ch)
        self.relu = nn.ReLU6(inplace=True)
        self.shortcut = nn.Sequential()  # empty Sequential module returns the original input

        if stride != 1 or i_ch != o_ch:  # tackle input/output size/channel mismatch during shortcut add
            self.shortcut = nn.Sequential(
                nn.Conv2d(i_ch, o_ch, kernel_size=1, padding=0, stride=stride, bias=False),
                nn.BatchNorm2d(o_ch)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        out = self.scale(out)*out + self.shortcut(x)
        out = self.relu(out)

        return out


class LateralBlock(nn.Module):
    """
    Feature Pyramid LateralBlock
    """
    def __init__(self, c_planes, base_num=256):
        super(LateralBlock, self).__init__()
        self.lateral = nn.Conv2d(c_planes, base_num, 1)
        self.smooth = nn.Conv2d(base_num, base_num, 3, padding=1)
        self.lateral_bn = nn.BatchNorm2d(base_num)

    def forward(self, c, p):
        """
        :param c: c layer before conv 1x1
        :param p: p layer before upsample
        :return: conv3x3( conv1x1(c) + upsample(p) )
        """
        _, _, H, W = c.size()
        c = self.lateral(c)
        p = F.upsample(p, scale_factor=2, mode='nearest')
        # p = F.upsample(p, size=(H, W), mode='bilinear')
        p = p[:, :, :H, :W] + c
        p = self.smooth(p)
        p = self.lateral_bn(p)
        return p


class SEResNeXtFPN(nn.Module):
    def __init__(self, num_list, base_channel_num=256):
        super(SEResNeXtFPN, self).__init__()
        self.c_num = base_channel_num

        # bottom up layers/stages
        # 3 -> (7x7 x 64) -> 64 -> BN -> ReLU -> MaxPool ->64
        self.conv1 = self._make_conv1()
        self.conv2_x = self._make_stage(self.c_num//2, self.c_num, num_list[0], stride=1)
        self.conv3_x = self._make_stage(self.c_num, self.c_num*2,  num_list[1], stride=2)
        self.conv4_x = self._make_stage(self.c_num*2, self.c_num*4, num_list[2], stride=2)
        self.conv5_x = self._make_stage(self.c_num*4, self.c_num*8, num_list[3], stride=2)
        # top down layers
        self.layer_p5 = nn.Conv2d(self.c_num*8, self.c_num, 1)
        self.layer_p4 = LateralBlock(self.c_num*4, base_num=self.c_num)  # takes p5 and c4 (1024)
        self.layer_p3 = LateralBlock(self.c_num*2, base_num=self.c_num)
        self.layer_p2 = LateralBlock(self.c_num, base_num=self.c_num)

    def _make_conv1(self):
        return nn.Sequential(
            nn.Conv2d(3, 32, 7, stride=2, padding=3, bias=False),  # (224-7+2*3) // 2 +1 = 112
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True),

            # shrink to 1/4 of original size
            # nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU6(inplace=True),

            nn.Conv2d(64, self.c_num//2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.c_num//2),
            nn.ReLU6(inplace=True),
        )

    def _make_stage(self, i_ch, o_ch, num_blocks, stride=1):
        """
        making conv_2, conv_3, conv_4, conv_5
        :param i_ch: channels of input  tensor
        :param o_ch: channels of output tensor
        :param num_blocks: repeats of bottleneck
        :param stride: stride of the 3x3 conv layer of each bottleneck
        :return:
        """
        layers = []
        layers.append(Bottleneck(i_ch, o_ch, stride))  # only the first stage in the module need stride=2
        for i in range(1, num_blocks):
            layers.append(Bottleneck(o_ch, o_ch))
        return nn.Sequential(*layers)

    def initialize_base_weights(self):
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

    def forward(self, x):
        c1 = self.conv1(x)
        c2 = self.conv2_x(c1)
        c3 = self.conv3_x(c2)
        c4 = self.conv4_x(c3)
        c5 = self.conv5_x(c4)

        p5 = self.layer_p5(c5)
        p4 = self.layer_p4(c4, p5)
        p3 = self.layer_p3(c3, p4)
        p2 = self.layer_p2(c2, p3)

        latent_feat = [c2, c3, c4, c5]
        final_feat = [p2, p3, p4, p5]
        return latent_feat, final_feat


class RefineSSDSEResNeXt(nn.Module):
    def __init__(self, cfg):
        super(RefineSSDSEResNeXt, self).__init__()

        self.num_classes = cfg.num_classes
        self.base_channel_num = cfg.base_channel_num
        self.use_refine = cfg.use_refine
        self.base_mbox = 2 * len(cfg.aspect_ratios[0]) + 1
        self.base_channel_list = [ int(self.base_channel_num * scale) for scale in [1, 2, 4, 8] ]

        self.feature_net = SEResNeXtFPN([3, 4, 6, 3], base_channel_num=self.base_channel_num)
        if self.use_refine:
            self.arm_loc = nn.ModuleList([
                nn.Conv2d(self.base_channel_list[0], 4*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(self.base_channel_list[1], 4*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(self.base_channel_list[2], 4*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(self.base_channel_list[3], 4*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
            ])
            self.arm_conf = nn.ModuleList([
                nn.Conv2d(self.base_channel_list[0], 2*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(self.base_channel_list[1], 2*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(self.base_channel_list[2], 2*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
                nn.Conv2d(self.base_channel_list[3], 2*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
            ])
        self.odm_loc = nn.ModuleList([
            nn.Conv2d(self.base_channel_num, 4*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(self.base_channel_num, 4*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(self.base_channel_num, 4*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(self.base_channel_num, 4*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
        ])
        self.odm_conf = nn.ModuleList([
            nn.Conv2d(self.base_channel_num, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(self.base_channel_num, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(self.base_channel_num, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(self.base_channel_num, self.num_classes*self.base_mbox, kernel_size=3, stride=1, padding=1, bias=False),
        ])

    def initialize_weights(self, ckpt_path=None):
        if ckpt_path and Path(ckpt_path).exists():
            state_dict = torch.load(ckpt_path, lambda storage, loc: storage)

            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                head = k[:7]
                if head == 'module.':
                    name = k[7:] # remove `module.`
                else:
                    name = k
                new_state_dict[name] = v

            self.load_state_dict(new_state_dict)
            # self.load_state_dict(torch.load(ckpt_path))
            logging.info("load weights from %s" % ckpt_path)
        else:
            self.feature_net.initialize_base_weights()
            self.odm_loc.apply(kaiming_weights_init)
            self.odm_conf.apply(kaiming_weights_init)
            if self.use_refine:
                self.arm_loc.apply(kaiming_weights_init)
                self.arm_conf.apply(kaiming_weights_init)
            logging.info("init weights from scratch")

    def forward(self, x, inference=False):
        latent_feat, final_feat = self.feature_net(x)

        arm_loc_result, arm_conf_result = None, None
        arm_loc_list = list()
        arm_conf_list = list()
        obm_loc_list = list()
        obm_conf_list = list()

        if self.use_refine:
            for i in range(0, 4):
                arm_loc_list.append(self.arm_loc[i](latent_feat[i]).permute(0, 2, 3, 1).contiguous())
                arm_conf_list.append(self.arm_conf[i](latent_feat[i]).permute(0, 2, 3, 1).contiguous())
            arm_loc_temp = torch.cat([o.view(o.size(0), -1) for o in arm_loc_list], 1)
            arm_conf_temp = torch.cat([o.view(o.size(0), -1) for o in arm_conf_list], 1)
            arm_loc_result = arm_loc_temp.view(arm_loc_temp.size(0), -1, 4)
            if inference:
                arm_conf_result = nn.Softmax(-1)(arm_conf_temp.view(-1, 2))
            else:
                arm_conf_result = arm_conf_temp.view(arm_conf_temp.size(0), -1, 2)

        for i in range(0, 4):
            obm_loc_list.append(self.odm_loc[i](final_feat[i]).permute(0, 2, 3, 1).contiguous())
            obm_conf_list.append(self.odm_conf[i](final_feat[i]).permute(0, 2, 3, 1).contiguous())
        obm_loc = torch.cat([o.view(o.size(0), -1) for o in obm_loc_list], 1)
        obm_conf = torch.cat([o.view(o.size(0), -1) for o in obm_conf_list], 1)
        obm_loc_result = obm_loc.view(obm_loc.size(0), -1, 4)
        if inference:
            obm_conf_result = nn.Softmax(-1)(obm_conf.view(obm_conf.size(0), -1, self.num_classes))
        else:
            obm_conf_result = obm_conf.view(obm_conf.size(0), -1, self.num_classes)

        return (arm_loc_result, arm_conf_result, obm_loc_result, obm_conf_result)
