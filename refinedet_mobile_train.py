# -*- coding: utf-8 -*-

import sys
import os
import time
import numpy as np
import argparse
import pickle
import logging
from pathlib import Path

import visdom
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from src.config import config
from src.data.data_augment import detection_collate, BaseTransform, preproc
from src.data.coco import COCODet
from src.data.voc import VOCDetection, AnnotationTransform
from src.loss import RefineMultiBoxLoss, MultiBoxLoss, RepulsionLoss
from src.detector import Detector
from src.prior_box import PriorBox
from src.utils.args import get_args
from src.utils import setup_logger, kaiming_weights_init
from src.utils.timer import Timer

from src.utils.nms_wrapper import nms

from src.symbol.RefineSSD_vgg import build_net
from src.symbol.RefineSSD_mobilenet_v2 import RefineSSDMobileNet
from refinedet_mobile_val import val

def train(workspace, train_dataset, val_dataset, val_trainsform, priors, detector, base_channel_num, width_mult, use_refine,batch_size, num_workers, shape, base_lr, momentum, weight_decay, gamma, max_epoch=200, resume=False, resume_epoch=0, save_frequency=10, enable_cuda=False, gpu_ids=[], enable_visdom=False, prefix='refinedet_model'):

    if enable_visdom:
        viz = visdom.Visdom()

    workspace_path = Path(workspace)
    if not workspace_path.exists():
        workspace_path.mkdir(parents=True)
    val_results_path = workspace_path.joinpath('ss_predict')
    if not val_results_path.exists():
        val_results_path.mkdir(parents=True)
    log_file_path = workspace_path.joinpath("train-%s" % (shape))
    setup_logger(log_file_path.as_posix())

    net = RefineSSDMobileNet(num_classes, base_channel_num=base_channel_num, width_mult=width_mult, use_refine=use_refine)
    net.initialize_weights()
    logging.info(net)

    if enable_cuda and len(gpu_ids) > 0:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)

    timer = Timer()
    mean_odm_loss_c, mean_odm_loss_l, mean_arm_loss_c, mean_arm_loss_l = 0, 0, 0, 0
    arm_criterion = RefineMultiBoxLoss(2, overlap_thresh=0.5, neg_pos_ratio=3, enable_cuda=enable_cuda)
    odm_criterion = RefineMultiBoxLoss(num_classes, overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.001, enable_cuda=enable_cuda)
    arm_repulsion_criterion = RepulsionLoss(num_classes, overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.001, enable_cuda=enable_cuda)
    criterion = MultiBoxLoss(num_classes, overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.01, enable_cuda=enable_cuda)
    logging.info('Loading datasets...')
    train_dataset_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)

    # optimizer = optim.RMSprop(net.parameters(), lr=base_lr, alpha = 0.9, eps=1e-08, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[ i*6 for i in range(1, max_epoch//6) ], gamma=0.75)
    for epoch in range(max_epoch):
        net.train()
        scheduler.step()
        for iteration, (images, targets) in enumerate(train_dataset_loader):
            if enable_cuda:
                images = Variable(images.cuda())
                targets = [Variable(anno.cuda(), volatile=True) for anno in targets]
            else:
                images = Variable(images)
                targets = [Variable(anno, volatile=True) for anno in targets]

            timer.tic()
            arm_loc, arm_conf, odm_loc, odm_conf = net(images)
            timer.toc()

            if use_refine:
                arm_loss_l, arm_loss_c = arm_criterion((arm_loc, arm_conf), priors, targets)
                arm_repulsion_criterion((arm_loc, arm_conf), priors, targets)

            odm_loss_l, odm_loss_c = odm_criterion((odm_loc, odm_conf), priors, targets, (arm_loc, arm_conf))
            optimizer.zero_grad()
            if use_refine:
                loss = 0.2 * (arm_loss_l + arm_loss_c) + 0.8 * (odm_loss_l + odm_loss_c)
            else:
                loss = odm_loss_l + odm_loss_c
            loss.backward()
            optimizer.step()
            if use_refine:
                mean_arm_loss_c += arm_loss_c.data[0]
                mean_arm_loss_l += arm_loss_l.data[0]
            mean_odm_loss_l += odm_loss_l.data[0]
            mean_odm_loss_c += odm_loss_c.data[0]

            if iteration % save_frequency == 0:
                logging.info("[%d/%d] || total_loss: %.4f(mean_arm_loc_loss: %.4f mean_arm_cls_loss: %.4f mean_obm_loc_loss: %.4f mean_obm_cls_loss: %.4f) || Batch time: %.4f sec. || LR: %.6f" % (epoch, iteration, loss, mean_arm_loss_l/save_frequency, mean_arm_loss_c/save_frequency, mean_odm_loss_l/save_frequency, mean_odm_loss_c/save_frequency, timer.average_time, optimizer.param_groups[0]['lr']))
                timer.clear()
                mean_odm_loss_c, mean_odm_loss_l, mean_arm_loss_c, mean_arm_loss_l = 0, 0, 0, 0

        if epoch % save_frequency == 0:
            net.eval()
            save_ckpt_path = workspace_path.joinpath("%s-%d.pth" %(prefix, epoch))
            torch.save(net.state_dict(), save_ckpt_path)
            logging.info("save model to %s " % save_ckpt_path)
            val(net, detector, priors, num_classes, val_dataset, val_trainsform, val_results_path, enable_cuda=enable_cuda, max_per_image=300, thresh=0.005)
            net.train()

    final_model_path = workspace_path.joinpath("Final-refineDet-%d.pth" %(epoch)).as_posix()
    torch.save(net.state_dict(), final_model_path)
    logging.info("save final model to %s " % final_model_path)

if __name__ == '__main__':

    # v2 = Variable(torch.randn(1, 3, 320, 320), requires_grad=True)
    # model = RefineSSDMobileNet(shape=320, num_classes=2, base_channel_num=128, width_mult=0.5, use_refine=True)
    # model.initialize_weights()
    # y = model(v2)
    # import pdb
    # pdb.set_trace()

    args = get_args()
    workspace = args.workspace
    batch_size = args.batch_size
    dataset = args.dataset.upper()
    prefix = args.prefix
    resume = args.resume
    resume_epoch = args.resume_epoch
    save_frequency = args.save_frequency
    enable_visdom = args.visdom

    # shape = args.shape
    # base_lr = args.lr
    # momentum = args.momentum
    # weight_decay = args.weight_decay
    # gamma = args.gamma
    # num_workers = args.num_workers
    # basenet = args.basenet
    # ckpt_path = args.ckpt_path
    # top_k = args.top_k
    # nms_thresh = args.nms_thresh
    # confidence_thresh = args.confidence_thresh

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

    module_cfg = config.list[args.config_id]
    val_trainsform = BaseTransform(module_cfg['shape'], rgb_means, rgb_std, (2, 0, 1))
    priorbox = PriorBox(module_cfg)
    priors = Variable(priorbox.forward(), volatile=True)
    detector = Detector(num_classes, top_k=module_cfg['top_k'], conf_thresh=module_cfg['confidence_thresh'], nms_thresh=module_cfg['nms_thresh'], variance=module_cfg['variance'])

    if dataset == "VOC":
        train_dataset = VOCDetection(root_path, train_sets, preproc(img_dim, rgb_means, rgb_std, augment_ratio), AnnotationTransform())
        val_dataset = VOCDetection(root_path, val_sets, None, AnnotationTransform())
    elif dataset == "COCO":
        train_dataset = COCODet(root_path, train_sets, preproc(img_dim, rgb_means, rgb_std, augment_ratio))
        val_dataset = COCODet(root_path, val_sets, None)

    train(workspace, train_dataset, val_dataset, val_trainsform, priors, detector, module_cfg['base_channel_num'], module_cfg['width_mult'], module_cfg['use_refine'], module_cfg['batch_size'], module_cfg['num_workers'], module_cfg['shape'], module_cfg['base_lr'], module_cfg['momentum'], module_cfg['weight_decay'], module_cfg['gamma'], module_cfg['max_epoch'], resume, resume_epoch, save_frequency, enable_cuda, gpu_ids, enable_visdom, prefix)
