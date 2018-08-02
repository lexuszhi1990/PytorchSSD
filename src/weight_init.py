# -*- coding: utf-8 -*-

import torch.nn.init as init

def xavier(param):
    init.xavier_uniform(param)

# initialize newly added layers' weights with kaiming_normal method
def kaiming_weights_init(m):
    for key in m.state_dict():
        if key.split('.')[-1] == 'weight':
            if 'conv' in key:
                init.kaiming_normal(m.state_dict()[key], mode='fan_out')
            if 'bn' in key:
                m.state_dict()[key][...] = 1
        elif key.split('.')[-1] == 'bias':
            m.state_dict()[key][...] = 0
