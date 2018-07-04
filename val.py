# -*- coding: utf-8 -*-

import sys
import os
import time
import logging
import numpy as np
import argparse
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from src.config import config
from src.data.data_augment import BaseTransform
from src.data.coco import COCODet
from src.data.voc import VOCDetection, AnnotationTransform
from src.detector import Detector, DetectorScratch
from src.prior_box import PriorBox
from src.utils import load_weights
from src.utils.args import get_args
from src.utils.timer import Timer
from src.utils.nms_wrapper import nms

from src.symbol.RefineSSD_vgg import RefineSSDVGG
from src.symbol.RefineSSD_mobilenet_v2 import RefineSSDMobileNet


def val(net, detector, priors, num_classes, val_dataset, transform, save_folder, ckpt_path=None, enable_cuda=False, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        net = load_weights(net, ckpt_path)

    # dump predictions and assoc. ground truth to text file for now
    num_images = len(val_dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    for i in range(num_images):
        img = val_dataset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0), volatile=True)
        if enable_cuda:
            x = x.cuda()
        basic_scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]

        _t['im_detect'].tic()
        out = net(x=x, inference=True)  # forward pass
        detect_time = _t['im_detect'].toc()

        _t['misc'].tic()
        arm_loc, arm_conf, odm_loc, odm_conf = out
        output = detector.forward((odm_loc, odm_conf), priors, (arm_loc, arm_conf))
        output_np = output.cpu().numpy()
        nms_time = _t['misc'].toc()

        for j in range(1, num_classes):
            results = output_np[j]
            mask = results[:, 1] >= 0.05
            results = results[mask]
            if len(results) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue

            scores = results[:, 1]
            boxes = results[:, 2:6] * basic_scale
            c_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} detect_time:{:.3f}s nms_time:{:.3f}s'.format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    val_dataset.evaluate_detections(all_boxes, save_folder)


def val_scratch(net, detector, priors, num_classes, val_dataset, transform, save_folder, ckpt_path=None, enable_cuda=False, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    if ckpt_path is not None and os.path.exists(ckpt_path):
        net = load_weights(net, ckpt_path)

    # dump predictions and assoc. ground truth to text file for now
    num_images = len(val_dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    for i in range(num_images):
        img = val_dataset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0), volatile=True)
        if enable_cuda:
            x = x.cuda()
        basic_scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]

        _t['im_detect'].tic()
        out = net(x=x, inference=True)  # forward pass
        arm_loc, arm_conf, odm_loc, odm_conf = out
        boxes, scores = detector.forward((odm_loc, odm_conf), priors, (arm_loc,arm_conf))
        detect_time = _t['im_detect'].toc()
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        scale = np.array([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        boxes *= scale

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(np.float32, copy=False)

            # from src.utils.nms.py_cpu_nms import py_cpu_nms
            # keep = py_cpu_nms(c_dets, 0.45)
            keep = nms(c_dets, 0.45, force_cpu=enable_cuda)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} detect_time:{:.3f}s nms_time:{:.3f}s'.format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    val_dataset.evaluate_detections(all_boxes, save_folder)

if __name__ == '__main__':

    args = get_args()
    workspace = args.workspace
    shape = int(args.shape)
    dataset = args.dataset.upper()
    prefix = args.prefix
    ckpt_path = args.ckpt_path
    top_k = args.top_k
    nms_thresh = args.nms_thresh
    confidence_thresh = args.confidence_thresh

    gpu_ids = [int(i) for i in args.gpu_ids]
    enable_cuda = args.cuda and torch.cuda.is_available() and len(gpu_ids) > 0
    if enable_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        cudnn.benchmark = True

    if dataset == "COCO":
        basic_conf = config.coco
    elif dataset == "VOC":
        basic_conf = config.voc
    else:
        raise RuntimeError("not support dataset %s" % (dataset))

    root_path, val_sets, num_classes, img_dim, rgb_means, rgb_std, augment_ratio = basic_conf.root_path, basic_conf.val_sets, basic_conf.num_classes, basic_conf.img_dim, basic_conf.rgb_means, basic_conf.rgb_std, basic_conf.augment_ratio

    module_cfg = config.list[args.config_id]
    val_trainsform = BaseTransform(module_cfg['shape'], rgb_means, rgb_std, (2, 0, 1))
    priorbox = PriorBox(module_cfg)
    priors = Variable(priorbox.forward(), volatile=True)
    detector = Detector(num_classes, top_k=module_cfg['top_k'], conf_thresh=module_cfg['confidence_thresh'], nms_thresh=module_cfg['nms_thresh'], variance=module_cfg['variance'])

    net = RefineSSDMobileNet(num_classes, base_channel_num=module_cfg['base_channel_num'], width_mult=module_cfg['width_mult'], use_refine=module_cfg['use_refine'])

    if dataset == "VOC":
        val_dataset = VOCDetection(root_path, val_sets, None, AnnotationTransform())
    elif dataset == "COCO":
        val_dataset = COCODet(root_path, val_sets, None)

    val(net, detector, priors, num_classes, val_dataset, val_trainsform, workspace, ckpt_path=ckpt_path, enable_cuda=enable_cuda, max_per_image=300, thresh=0.005)
    # val(net, detector, priors, num_classes, val_dataset, val_trainsform, workspace, ckpt_path=ckpt_path, enable_cuda=enable_cuda, max_per_image=300, thresh=0.005)

    # resume_net_path = '/mnt/ckpt/pytorchSSD/Refine_vgg_320/v1/refineDet-model-50.pth'
    # resume_net_path = 'workspace/v2/refineDet-model-280.pth'
    # resume_net_path = '/mnt/ckpt/pytorchSSD/Refine_vgg_320/refinedet_vgg_0516/Refine_vgg_COCO_epoches_250.pth'
    # resume_net_path = '/mnt/ckpt/pytorchSSD/Refine_mobilenet/scratch-v2/refineDet-model-50.pth'

