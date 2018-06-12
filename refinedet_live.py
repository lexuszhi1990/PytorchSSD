# -*- coding: utf-8 -*-

import sys
import os
import time
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
from src.symbol.RefineSSD_vgg import build_net
from src.loss import RefineMultiBoxLoss
from src.detection import Detect
from src.prior_box import PriorBox
from src.utils import str2bool
from src.utils.nms_wrapper import nms
from src.utils.timer import Timer


class refinedet_wrapper(object):
    """docstring for refinedet_wrapper"""
    def __init__(self):
        data_shape = 320
        num_classes = 2
        resume_net_path = './ckpt/Refine_vgg_COCO_epoches_250.pth'
        module_cfg = config.coco.dimension_320
        rgb_std = (1,1,1)
        rgb_means = (104, 117, 123)

        self.net = build_net(data_shape, num_classes, use_refine=True)
        # https://pytorch.org/docs/master/torch.html?highlight=load#torch.load
        # state_dict = torch.load(resume_net_path)
        state_dict = torch.load(resume_net_path, lambda storage, loc: storage)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        self.net.load_state_dict(new_state_dict)
        priorbox = PriorBox(module_cfg)
        self.priors = Variable(priorbox.forward(), volatile=True)
        self.detector = Detect(num_classes, 0, module_cfg, object_score=0.01)
        self.val_trainsform = BaseTransform(self.net.size, rgb_means, rgb_std, (2, 0, 1))

    def analyse(self, orig_img):
        x = Variable(val_trainsform(orig_img).unsqueeze(0), volatile=True)
        if enable_cuda:
            x = x.cuda()

        _t = {'im_detect': Timer(), 'misc': Timer()}
        _t['im_detect'].tic()
        arm_loc, arm_conf, odm_loc, odm_conf = self.net(x=x, test=True)
        boxes, scores = self.detector.forward((odm_loc, odm_conf), priors, (arm_loc, arm_conf))
        detect_time = _t['im_detect'].toc()
        print("forward time: %fs" % (detect_time))
        boxes = boxes[0]
        scores=scores[0]
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        all_boxes = [[] for _ in range(num_classes)]
        for class_id in range(1, num_classes):
            inds = np.where(scores[:, class_id] > 0.95)[0]
            c_scores = scores[inds, class_id]
            c_bboxes = boxes[inds]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)
            keep = nms(c_dets, 0.45, force_cpu=True)
            all_boxes[class_id] = c_dets[keep, :]

        for class_id in range(1, num_classes):
            for det in all_boxes[class_id]:
                left, top, right, bottom, score = det
                orig_img = cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 0), 1)
                orig_img = cv2.putText(orig_img, '%d:%.3f'%(class_id, score), (int(left), int(top)+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
        # cv2.imwrite("./test_3.png", img_det)

        return orig_img

