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
import torchvision.transforms as transforms
import torch.nn.init as init
import torch.utils.data as data
from torch.autograd import Variable

from src.config import config
from src.data.data_augment import detection_collate, BaseTransform, preproc
from src.data.coco import COCODet
from src.data.voc import VOCDetection, AnnotationTransform
from src.loss import RefineMultiBoxLoss
# from src.detection import Detect
from src.detector import Detector
from src.prior_box import PriorBox
from src.utils import str2bool, load_weights
from src.utils.nms_wrapper import nms
from src.utils.timer import Timer

from src.symbol.RefineSSD_vgg import build_net
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
        output = detector.forward((odm_loc, odm_conf), priors.data, (arm_loc, arm_conf))
        output_np = output.cpu().numpy()
        nms_time = _t['misc'].toc()

        for j in range(1, num_classes):
            results = output_np[output_np[:, 0] == j]
            if len(results) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue

            scores = results[:, 1]
            boxes = results[:, 2:6] * basic_scale
            c_dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
            all_boxes[j][i] = c_dets

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} detect_time:{:.3f}s nms_time:{:.3f}s'.format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    val_dataset.evaluate_detections(all_boxes, save_folder)


parser = argparse.ArgumentParser(
    description='Refined SSD val')
parser.add_argument('--workspace', default='./workspace')
parser.add_argument('--shape', default='320', help='320 or 512 input size.')
parser.add_argument('--dataset', default='COCO', help='VOC or COCO dataset')
parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda to train model')
parser.add_argument('--gpu_ids', nargs='+', default=[], help='gpu id')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
parser.add_argument('--resume', default=False, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0, type=int, help='resume iter for retraining')
parser.add_argument('--max_epoch', default=300, type=int, help='max epoch for retraining')
parser.add_argument('--save_frequency', default=10, type=int, help='epoch for saving ckpt')
parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--basenet', default='./weights/pretrained/vgg16_reducedfc.pth', help='pretrained base model')
parser.add_argument('--ckpt_path', type=str, help='pretrained base model')

parser.add_argument('--warm_epoch', default=1,
                    type=int, help='max epoch for retraining')
parser.add_argument('--date', default='0327')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--test_frequency',default=10)
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')

if __name__ == '__main__':

    args = parser.parse_args()
    workspace = args.workspace
    batch_size = args.batch_size
    shape = int(args.shape)
    dataset = args.dataset.upper()
    base_lr = args.lr
    warm_epoch = args.warm_epoch
    max_epoch = args.max_epoch
    resume = args.resume
    resume_epoch = args.resume_epoch
    momentum = args.momentum
    weight_decay = args.weight_decay
    gamma = args.gamma
    num_workers = args.num_workers
    save_frequency = args.save_frequency
    basenet = args.basenet
    ckpt_path = args.ckpt_path
    enable_visdom = args.visdom
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

    root_path, train_sets, val_sets, num_classes, img_dim, rgb_means, rgb_std, augment_ratio = basic_conf.root_path, basic_conf.train_sets, basic_conf.val_sets, basic_conf.num_classes, basic_conf.img_dim, basic_conf.rgb_means, basic_conf.rgb_std, basic_conf.augment_ratio

    module_cfg = getattr(basic_conf, "dimension_%d"%(int(shape)))
    priorbox = PriorBox(module_cfg)
    priors = Variable(priorbox.forward(), volatile=True)
    detector = Detector(num_classes, variance=module_cfg['variance'])
    net = RefineSSDMobileNet(shape, num_classes, use_refine=True)
    val_trainsform = BaseTransform(shape, rgb_means, rgb_std, (2, 0, 1))

    if dataset == "VOC":
        val_dataset = VOCDetection(root_path, val_sets, None, AnnotationTransform())
    elif dataset == "COCO":
        val_dataset = COCODet(root_path, val_sets, None)

    val(net, detector, priors, num_classes, val_dataset, val_trainsform, workspace, ckpt_path=ckpt_path, enable_cuda=enable_cuda, max_per_image=300, thresh=0.005)

    # resume_net_path = '/mnt/ckpt/pytorchSSD/Refine_vgg_320/v1/refineDet-model-50.pth'
    # resume_net_path = 'workspace/v2/refineDet-model-280.pth'
    # resume_net_path = '/mnt/ckpt/pytorchSSD/Refine_vgg_320/refinedet_vgg_0516/Refine_vgg_COCO_epoches_250.pth'
    # resume_net_path = '/mnt/ckpt/pytorchSSD/Refine_mobilenet/scratch-v2/refineDet-model-50.pth'

