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

parser = argparse.ArgumentParser(
    description='Receptive Field Block Net Training')
parser.add_argument('-v', '--version', default='Refine_vgg',
                    help='Refine_vgg')
parser.add_argument('-s', '--size', default='320',
                    help='320 or 512 input size.')
parser.add_argument('-d', '--dataset', default='VOC',
                    help='VOC or COCO dataset')
parser.add_argument(
    '--basenet', default='/mnt/lvmhdd1/zuoxin/ssd_pytorch_models/vgg16_reducedfc.pth', help='pretrained base model')
#parser.add_argument(
#    '--basenet', default='/mnt/lvmhdd1/zuoxin/ssd_pytorch_models/mb.pth', help='pretrained base model')
parser.add_argument('--jaccard_threshold', default=0.5,
                    type=float, help='Min Jaccard index for matching')
parser.add_argument('-b', '--batch_size', default=32,
                    type=int, help='Batch size for training')
parser.add_argument('--num_workers', default=8,
                    type=int, help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True,
                    type=bool, help='Use cuda to train model')
parser.add_argument('--gpu_ids', nargs='+', default=[], help='gpu id')
parser.add_argument('--lr', '--learning-rate',
                    default=1e-3, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')

parser.add_argument('--resume_net', default=False, help='resume net for retraining')
parser.add_argument('--resume_epoch', default=0,
                    type=int, help='resume iter for retraining')

parser.add_argument('-max','--max_epoch', default=300,
                    type=int, help='max epoch for retraining')
parser.add_argument('-we','--warm_epoch', default=1,
                    type=int, help='max epoch for retraining')
parser.add_argument('--weight_decay', default=5e-4,
                    type=float, help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1,
                    type=float, help='Gamma update for SGD')
parser.add_argument('--log_iters', default=True,
                    type=bool, help='Print the loss at each iteration')
parser.add_argument('--save_folder', default='/mnt/lvmhdd1/zuoxin/ssd_pytorch_models/refine/',
                    help='Location to save checkpoint models')
parser.add_argument('--date',default='0327')
parser.add_argument('--save_frequency',default=10)
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--test_frequency',default=10)
parser.add_argument('--visdom', default=False, type=str2bool, help='Use visdom to for loss visualization')
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')

if __name__ == '__main__':

    args = parser.parse_args()
    save_folder = os.path.join(args.save_folder, args.version+'_'+args.size, args.date)
    enable_cuda = args.cuda and torch.cuda.is_available()
    if enable_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        cudnn.benchmark = True
    num_classes = 81
    data_shape = 320
    resume_net_path = './ckpt/Refine_vgg_COCO_epoches_250.pth'

    net = build_net(data_shape, num_classes, use_refine=True)
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
    net.load_state_dict(new_state_dict)

    module_cfg = config.coco.dimension_320
    rgb_std = (1,1,1)
    rgb_means = (104, 117, 123)
    priorbox = PriorBox(module_cfg)
    priors = Variable(priorbox.forward(), volatile=True)
    detector = Detect(num_classes, 0, module_cfg, object_score=0.01)
    val_trainsform = BaseTransform(net.size, rgb_means, rgb_std, (2, 0, 1))
    img = cv2.imread('./samples/ebike-three.jpg')
    x = Variable(val_trainsform(img).unsqueeze(0), volatile=True)
    if enable_cuda:
        x = x.cuda()

    _t = {'im_detect': Timer(), 'misc': Timer()}
    _t['im_detect'].tic()
    arm_loc, arm_conf, odm_loc, odm_conf = net(x=x, test=True)
    boxes, scores = detector.forward((odm_loc, odm_conf), priors, (arm_loc, arm_conf))
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

    img_det = img.copy()
    for class_id in range(1, num_classes):
        for det in all_boxes[class_id]:
            left, top, right, bottom, score = det
            img_det = cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 0), 1)
            img_det = cv2.putText(img_det, '%d:%.3f'%(class_id, score), (int(left), int(top)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

    cv2.imwrite("./test_3.png", img_det)
