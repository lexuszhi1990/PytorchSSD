# -*- coding: utf-8 -*-

import sys
import os
import time
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
import torch.utils.data as data
from torch.autograd import Variable

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

parser.add_argument('--warm_epoch', default=1,
                    type=int, help='max epoch for retraining')
parser.add_argument('--date', default='0327')
parser.add_argument('--retest', default=False, type=bool,
                    help='test cache results')
parser.add_argument('--test_frequency',default=10)
parser.add_argument('--send_images_to_visdom', type=str2bool, default=False, help='Sample a random image from each 10th batch, send it to visdom after augmentations step')


def val(net, detector, priors, testset, num_classes, transform, save_folder, enable_cuda=False, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)

    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    for i in range(num_images):
        img = testset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0), volatile=True)
        if enable_cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        out = net(x=x, test=True)  # forward pass
        arm_loc, arm_conf, odm_loc, odm_conf = out
        boxes, scores = detector.forward((odm_loc,odm_conf), priors, (arm_loc,arm_conf))
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores=scores[0]
        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)

            keep = nms(c_dets, 0.45, force_cpu=True)
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
    testset.evaluate_detections(all_boxes, save_folder)

if __name__ == '__main__':

    args = parser.parse_args()
    workspace = args.workspace
    batch_size = args.batch_size
    shape = args.shape
    dataset = args.dataset
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

    root_path, train_sets, val_sets = basic_conf.root_path,  basic_conf.train_sets, basic_conf.val_sets
    num_classes, img_dim, rgb_means, rgb_std, augment_ratio = basic_conf.num_classes, basic_conf.img_dim, basic_conf.rgb_means, basic_conf.rgb_std, basic_conf.augment_ratio
    module_cfg = getattr(basic_conf, "dimension_%d"%(int(shape)))

    resume_net_path = 'workspace/v1/refineDet-model-210.pth'
    # resume_net_path = 'workspace/v2/refineDet-model-160.pth'
    # resume_net_path = '/mnt/ckpt/pytorchSSD/Refine_vgg_320/refinedet_vgg_0516/Refine_vgg_COCO_epoches_250.pth'

    net = build_net(int(shape), num_classes, use_refine=True)
    state_dict = torch.load(resume_net_path)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        head = k[:7]
        if head == 'module.':
            name = k[7:] # remove `module.`
        else:
            name = k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    print(net)

    priorbox = PriorBox(module_cfg)
    priors = Variable(priorbox.forward(), volatile=True)
    detector = Detect(num_classes, 0, module_cfg, object_score=0.01)
    val_dataset = COCODet(root_path, val_sets, None)
    val_trainsform = BaseTransform(net.size, rgb_means, rgb_std, (2, 0, 1))
    val(net, detector, priors, val_dataset, num_classes, val_trainsform, workspace, enable_cuda=enable_cuda, max_per_image=300, thresh=0.005)
