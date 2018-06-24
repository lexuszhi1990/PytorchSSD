# -*- coding: utf-8 -*-

import sys
import os
import time
import logging
import cv2
import numpy as np
import argparse
import pickle
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torch.nn.init as init
from torch.autograd import Variable
import torch.utils.data as data

from src.config import config
from src.data.data_augment import detection_collate, BaseTransform, preproc
from src.data.coco import COCODet
from src.loss import RefineMultiBoxLoss, MultiBoxLoss
from src.detector import Detector
from src.prior_box import PriorBox
from src.utils.nms_wrapper import nms
from src.utils import setup_logger, kaiming_weights_init
from src.utils.args import get_args
from src.utils.timer import Timer

from src.symbol.RefineSSD_vgg import build_net
from src.symbol.RefineSSD_mobilenet_v2 import RefineSSDMobileNet

if __name__ == '__main__':

    args = get_args()
    workspace = args.workspace
    shape = int(args.shape)
    image_path = args.eval_img
    ckpt_path = args.ckpt_path
    top_k = args.top_k
    nms_thresh = args.nms_thresh
    confidence_thresh = args.confidence_thresh

    gpu_ids = [int(i) for i in args.gpu_ids]
    enable_cuda = args.cuda and torch.cuda.is_available() and len(gpu_ids) > 0
    if enable_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        cudnn.benchmark = True

    setup_logger(workspace)

    basic_conf = config.coco
    module_cfg = basic_conf.list[args.config_id]
    val_trainsform = BaseTransform(module_cfg['shape'], basic_conf.rgb_means, basic_conf.rgb_std, (2, 0, 1))
    priorbox = PriorBox(module_cfg)
    priors = Variable(priorbox.forward(), volatile=True)
    detector = Detector(basic_conf.num_classes, top_k=module_cfg['top_k'], conf_thresh=module_cfg['confidence_thresh'], nms_thresh=module_cfg['nms_thresh'], variance=module_cfg['variance'])

    net = RefineSSDMobileNet(shape, basic_conf.num_classes, base_channel_num=module_cfg['base_channel_num'], width_mult=module_cfg['width_mult'], use_refine=module_cfg['use_refine'])
    net.initialize_weights(ckpt_path)
    if enable_cuda and len(gpu_ids) > 0:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)
        net.cuda()

    img = cv2.imread(image_path)
    x = Variable(val_trainsform(img).unsqueeze(0), volatile=True)
    if enable_cuda:
        x = x.cuda()
    basic_scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    # _t['im_detect'].tic()
    forward_starts = time.time()
    out = net(x=x, inference=True)  # forward pass
    # detect_time = _t['im_detect'].toc()
    detect_time = time.time() - forward_starts

    forward_starts = time.time()
    out = net(x=x, inference=True)  # forward pass
    # detect_time = _t['im_detect'].toc()
    detect_time = time.time() - forward_starts

    print(detect_time)

    _t['misc'].tic()
    arm_loc, arm_conf, odm_loc, odm_conf = out
    output = detector.forward((odm_loc, odm_conf), priors, (arm_loc, arm_conf))
    nms_time = _t['misc'].toc()

    output_np = output.cpu().numpy()

    for class_id in range(1, basic_conf.num_classes):
        cls_outut = output_np[class_id]
        dets = cls_outut[cls_outut[:, 1] > 0.25]
        dets[:, 2:6] = np.floor(dets[:, 2:6] * basic_scale)
        for det in dets:
            cls_id, score, left, top, right, bottom = det
            img_det = cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 0), 1)
            img_det = cv2.putText(img_det, '%d:%.3f'%(class_id, score), (int(left), int(top)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    cv2.imwrite("./test_v4.png", img_det)
    logging.info('im_detect: %s, detect_time:%.3fs nms_time:%.3fs'%(image_path, detect_time, nms_time))
