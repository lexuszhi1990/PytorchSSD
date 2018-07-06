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

if True:
    config.coco.train_sets = 'train2017'
    config.coco.val_sets = 'val2017'
    config.coco.num_classes = 81
else:
    config.coco.train_sets = 'person_train2017'
    config.coco.val_sets = 'person_val2017'
    config.coco.num_classes = 2

config.coco.root_path = '/mnt/dataset/coco'
config.coco.img_dim = 320
config.coco.rgb_std = (1,1,1)
config.coco.rgb_means = (104,117,123)
config.coco.augment_ratio = 0.2

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

    'width_mult': 0.5,

    'base_channel_num': 128,
}

config.list = {
    "v1" : {
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
        'use_refine': True,
    },
    "v2" : {

        'train_sets' : 'person_train2017',
        'val_sets' : 'person_val2017',
        'num_classes' : 2,

        'feature_maps': [40, 20, 10, 5],
        'min_dim': 320,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,

        'width_mult': 1.,
        'base_channel_num': 128,
        'use_refine': False,
        'batch_size': 64,
        'num_workers': 8,
        'shape': 320,
        'base_lr': 1e-3,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'max_epoch': 200,

        'top_k': 200,
        'confidence_thresh': 0.01,
        'nms_thresh': 0.45,
    },
    "v3" : {

        'train_sets' : 'person_train2017',
        'val_sets' : 'person_val2017',
        'num_classes' : 2,

        'feature_maps': [40, 20, 10, 5],
        'min_dim': 320,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,

        'width_mult': 0.75,
        'base_channel_num': 128,
        'use_refine': False,
        'batch_size': 192,
        'num_workers': 36,
        'shape': 320,
        'base_lr': 4e-2,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'max_epoch': 200,

        'top_k': 200,
        'confidence_thresh': 0.01,
        'nms_thresh': 0.45,
    },
    "v4" : {

        'train_sets' : 'person_train2017',
        'val_sets' : 'person_val2017',
        'num_classes' : 2,

        'feature_maps': [40, 20, 10, 5],
        'min_dim': 320,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,

        'width_mult': 0.75,
        'base_channel_num': 128,
        'use_refine': True,
        'batch_size': 192,
        'num_workers': 36,
        'shape': 320,
        'base_lr': 4e-2,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'max_epoch': 200,

        'top_k': 200,
        'confidence_thresh': 0.01,
        'nms_thresh': 0.45,
    },
    "v5" : {

        'train_sets' : 'person_train2017',
        'val_sets' : 'person_val2017',
        'num_classes' : 2,

        'feature_maps': [40, 20, 10, 5],
        'min_dim': 320,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        # 'aspect_ratios': [[2, 3], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,

        'width_mult': 0.5,
        'base_channel_num': 128,
        'use_refine': True,
        # 'batch_size': 256,
        # 'num_workers': 64,
        'batch_size': 4,
        'num_workers': 2,
        'shape': 320,
        'base_lr': 8e-2,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'max_epoch': 201,

        'top_k': 100,
        'confidence_thresh': 0.15,
        # 'top_k': 200,
        # 'confidence_thresh': 0.01,
        'nms_thresh': 0.45,
    },
    "v6" : {

        'train_sets' : 'train2017',
        'val_sets' : 'val2017',
        'num_classes' : 81,

        'feature_maps': [40, 20, 10, 5],
        'min_dim': 320,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[2], [2], [2], [2]],
        # 'aspect_ratios': [[2, 3], [2], [2], [2]],
        'variance': [0.1, 0.2],
        'clip': True,

        'width_mult': 1.,
        'base_channel_num': 256,
        'use_refine': True,
        'batch_size': 96,
        'num_workers': 32,
        'shape': 320,
        'base_lr': 4e-2,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'max_epoch': 201,

        # 'top_k': 100,
        # 'confidence_thresh': 0.15,
        'top_k': 200,
        'confidence_thresh': 0.01,
        'nms_thresh': 0.45,
    },

    # resnet config
    "r1" : {

        'train_sets' : 'person_train2017',
        'val_sets' : 'person_val2017',
        'num_classes' : 2,

        'feature_maps': [40, 20, 10, 5],
        'min_dim': 320,
        'steps': [8, 16, 32, 64],
        'min_sizes': [32, 64, 128, 256],
        'max_sizes': [],
        'aspect_ratios': [[3], [3], [3], [3]],
        'variance': [0.1, 0.2],
        'clip': True,

        'width_mult': 1.,
        'base_channel_num': 256,
        'use_refine': True,
        'batch_size': 128,
        'num_workers': 8,
        'shape': 320,
        # 'base_lr': 3e-2,
        'base_lr': 0.0005,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'max_epoch': 201,

        'top_k': 200,
        'confidence_thresh': 0.15,
        # 'top_k': 200,
        # 'confidence_thresh': 0.01,
        'nms_thresh': 0.45,
    },

    "v-test" : {
        # 'shape': [320, 320], # [width, height]
        # 'feature_maps': [[40, 40], [20, 20], [10, 10], [5, 5]],
        # 'steps': [[8, 8], [16, 16], [32, 32], [64, 64]],
        # 'aspect_ratios': [[2], [2], [2], [2]],
        # 'min_sizes': [[32, 32], [64, 64], [128, 128], [256, 256]],

        'feature_maps': [[64, 36], [32, 18], [16, 9], [8, 5]],
        'shape': [512, 288], # [width, height]
        'steps': [[8, 8], [16, 16], [32, 32], [64, 56]],
        'aspect_ratios': [[2, 3], [2, 3], [2, 3], [2, 3]],
        'min_sizes': [[32, 24], [64, 48], [128, 96], [256, 196]],
        'max_sizes': [],
        'variance': [0.1, 0.2],
        'clip': True,
        # 'min_dim': 320,

        'width_mult': 0.75,
        'base_channel_num': 128,
        'use_refine': True,
        'batch_size': 192,
        'num_workers': 36,
        'base_lr': 4e-2,
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'gamma': 0.1,
        'max_epoch': 200,

        'top_k': 200,
        'confidence_thresh': 0.01,
        'nms_thresh': 0.45,
    },
}
