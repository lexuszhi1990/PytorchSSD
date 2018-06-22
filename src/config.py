# -*- coding: utf-8 -*-

from easydict import EasyDict as edict

config = edict()
config.coco = edict()
config.voc = edict()

####################################################
#           voc config
####################################################

config.voc.root_path = '/mnt/dataset/VOCdevkit'
config.voc.train_sets = [('2012', 'train')]
config.voc.val_sets = [('2012', 'val')]
config.voc.base_stepvalues = [120, 150, 200]
config.voc.img_dim = 320
config.voc.rgb_std = (1,1,1)
config.voc.rgb_means = (104,117,123)
config.voc.augment_ratio = 0.2
config.voc.num_classes = 21

# RFB CONFIGS
config.voc.dimension_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [30, 60, 111, 162, 213, 264],

    'max_sizes': [60, 111, 162, 213, 264, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}
config.voc.dimension_320 = {
    'feature_maps': [40, 20, 10, 5],

    'min_dim': 320,

    'steps': [8, 16, 32, 64],

    'min_sizes': [32, 64, 128, 256],

    'max_sizes': [],

    'aspect_ratios': [[2], [2], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

config.voc.dimension_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],

    'min_dim': 512,

    'steps': [8, 16, 32, 64, 128, 256, 512],

    'min_sizes': [35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8],

    'max_sizes': [76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

####################################################
#           coco config
####################################################

config.coco.root_path = '/mnt/dataset/coco'
if False:
    config.coco.train_sets = 'train2017'
    config.coco.val_sets = 'val2017'
    config.coco.num_classes = 81
else:
    config.coco.train_sets = 'person_train2017'
    config.coco.val_sets = 'person_val2017'
    config.coco.num_classes = 2
config.coco.img_dim = 320
config.coco.rgb_std = (1,1,1)
config.coco.rgb_means = (104,117,123)
config.coco.augment_ratio = 0.2
config.coco.base_stepvalues = [90, 120, 140]

config.coco.dimension_300 = {
    'feature_maps': [38, 19, 10, 5, 3, 1],

    'min_dim': 300,

    'steps': [8, 16, 32, 64, 100, 300],

    'min_sizes': [21, 45, 99, 153, 207, 261],

    'max_sizes': [45, 99, 153, 207, 261, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

config.coco.dimension_512 = {
    'feature_maps': [64, 32, 16, 8, 4, 2, 1],

    'min_dim': 512,

    'steps': [8, 16, 32, 64, 128, 256, 512],

    'min_sizes': [20.48, 51.2, 133.12, 215.04, 296.96, 378.88, 460.8],

    'max_sizes': [51.2, 133.12, 215.04, 296.96, 378.88, 460.8, 542.72],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}

config.coco.dimension_mobile_300 = {
    'feature_maps': [19, 10, 5, 3, 2, 1],

    'min_dim': 300,

    'steps': [16, 32, 64, 100, 150, 300],

    'min_sizes': [45, 90, 135, 180, 225, 270],

    'max_sizes': [90, 135, 180, 225, 270, 315],

    'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,
}


config.coco.dimension_320 = {
    'feature_maps': [40, 20, 10, 5],

    'min_dim': 320,

    'steps': [8, 16, 32, 64],

    'min_sizes': [32, 64, 128, 256],

    'max_sizes': [],

    'aspect_ratios': [[2], [2], [2], [2]],

    'variance': [0.1, 0.2],

    'clip': True,

    'width_mult': 1,

    'base_channel_num': 128,

    'use_refine': True


}
