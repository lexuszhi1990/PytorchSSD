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
import torch.nn.init as init
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

from src.config import config
from src.data.data_augment import detection_collate, BaseTransform, preproc
from src.data.coco import COCODet
from src.data.voc import VOCDetection, AnnotationTransform
from src.loss import RefineMultiBoxLoss
from src.detection import Detect
from src.prior_box import PriorBox
from src.utils import str2bool, setup_logger
from src.utils.nms_wrapper import nms
from src.utils.timer import Timer

from src.symbol.RefineSSD_vgg import build_net
from src.symbol.RefineSSD_mobilenet_v2 import RefineSSDMobileNet

parser = argparse.ArgumentParser(
    description='Refined SSD train')
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

def train(workspace, train_dataset, val_dataset, module_cfg, batch_size, shape, base_lr, momentum, weight_decay, gamma, gpu_ids, enable_cuda, max_epoch, resume, resume_epoch, num_workers, save_frequency, basenet, enable_visdom):

    if enable_visdom:
        viz = visdom.Visdom()

    priorbox = PriorBox(module_cfg)
    detector = Detect(num_classes, 0, module_cfg, object_score=0.01)
    priors = Variable(priorbox.forward(), volatile=True)

    workspace_path = Path(workspace)
    if not workspace_path.exists():
        workspace_path.mkdir(parents=True)
    val_results_path = workspace_path.joinpath('ss_predict')
    if not val_results_path.exists():
        val_results_path.mkdir(parents=True)
    log_file_path = workspace_path.joinpath("train-%s" % (shape))
    setup_logger(log_file_path.as_posix())

    net = RefineSSDMobileNet(320, num_classes, use_refine=True)
    logging.info(net)
    if not resume:

        def xavier(param):
            init.xavier_uniform(param)

        def weights_init(m):
            for key in m.state_dict():
                if key.split('.')[-1] == 'weight':
                    if 'conv' in key:
                        init.kaiming_normal(m.state_dict()[key], mode='fan_out')
                    if 'bn' in key:
                        m.state_dict()[key][...] = 1
                elif key.split('.')[-1] == 'bias':
                    m.state_dict()[key][...] = 0

        logging.info('Initializing weights...')
        # initialize newly added layers' weights with kaiming_normal method
        if Path(basenet).exists():
            base_weights = torch.load(basenet)
            logging.info('Loading base network...')
            net.base.load_state_dict(base_weights)
        else:
            net.base.initialize_weights()

        net.appended_layer.apply(weights_init)
        net.trans_layers.apply(weights_init)
        net.up_layers.apply(weights_init)
        net.latent_layrs.apply(weights_init)
        net.arm_loc.apply(weights_init)
        net.arm_conf.apply(weights_init)
        net.odm_loc.apply(weights_init)
        net.odm_conf.apply(weights_init)
    else:
        # load resume network
        # resume_path = os.path.join(save_folder,version+'_'+dataset + '_epoches_'+ str(resume_epoch) + '.pth')
        logging.info('Loading resume network',resume_path)
        state_dict = torch.load(resume_path)
        # create new OrderedDict that does not contain `module.`
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            head = k[:7]
            if head == 'module.':
                name = k[7:] # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        net.load_state_dict(new_state_dict)
    if enable_cuda and len(gpu_ids) > 0:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)

    timer = Timer()
    mean_odm_loss_c = 0
    mean_odm_loss_l = 0
    mean_arm_loss_c = 0
    mean_arm_loss_l = 0
    log_interval = 50
    arm_criterion = RefineMultiBoxLoss(2, overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.5, enable_cuda=enable_cuda)
    odm_criterion = RefineMultiBoxLoss(num_classes, overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.5, enable_cuda=enable_cuda)
    logging.info('Loading datasets...')
    train_dataset_loader = data.DataLoader(train_dataset, batch_size,
                                           shuffle=True,
                                           num_workers=num_workers,
                                           collate_fn=detection_collate)

    optimizer = optim.RMSprop(net.parameters(), lr=base_lr, alpha = 0.9, eps=1e-08, momentum=momentum, weight_decay=weight_decay)
    # optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[ i for i in range(1, 30) ], gamma=0.8)
    for epoch in range(1, max_epoch):

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
            arm_loss_l, arm_loss_c = arm_criterion((arm_loc, arm_conf), priors, targets)
            odm_loss_l, odm_loss_c = odm_criterion((odm_loc, odm_conf), priors, targets, (arm_loc, arm_conf), False)
            mean_arm_loss_l += arm_loss_l.data[0]
            mean_arm_loss_c += arm_loss_c.data[0]
            mean_odm_loss_l += odm_loss_l.data[0]
            mean_odm_loss_c += odm_loss_c.data[0]

            if epoch <= 100:
                loss = arm_loss_l + arm_loss_c
                # loss = 0.8*arm_loss_l + 0.8*arm_loss_c + 0.2*odm_loss_l + 0.2*odm_loss_c
            elif epoch > 100 and epoch < 200:
                loss = odm_loss_l + odm_loss_c
            else:
                loss = 0.2*arm_loss_l + 0.2*arm_loss_c + 0.8*odm_loss_l + 0.8*odm_loss_c

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if iteration % log_interval == 0:
                logging.info("[%d/%d] || total_loss: %.4f(mean_arm_loc_loss: %.4f mean_arm_cls_loss: %.4f mean_obm_loc_loss: %.4f mean_obm_cls_loss: %.4f) || Batch time: %.4f sec. || LR: %.6f" % (epoch, iteration, loss, mean_arm_loss_l/log_interval, mean_arm_loss_c/log_interval, mean_odm_loss_l/log_interval, mean_odm_loss_c/log_interval, timer.average_time, optimizer.param_groups[0]['lr']))
                timer.clear()
                mean_odm_loss_c = 0
                mean_odm_loss_l = 0
                mean_arm_loss_c = 0
                mean_arm_loss_l = 0

        if epoch % save_frequency == 0:
            save_ckpt_path = workspace_path.joinpath("refineDet-model-%d.pth" %(epoch))
            torch.save(net.state_dict(), save_ckpt_path)
            logging.info("save model to %s " % save_ckpt_path)

    torch.save(net.state_dict(), workspace_path.joinpath("Final-refineDet-%d.pth" %(epoch)))

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

    if dataset == "VOC":
        train_dataset = VOCDetection(root_path, train_sets, preproc(img_dim, rgb_means, rgb_std, augment_ratio), AnnotationTransform())
        val_dataset = VOCDetection(root_path, val_sets, None, AnnotationTransform())
    elif dataset == "COCO":
        train_dataset = COCODet(root_path, train_sets, preproc(img_dim, rgb_means, rgb_std, augment_ratio))
        val_dataset = COCODet(root_path, val_sets, None)

    train(workspace, train_dataset, val_dataset, module_cfg, batch_size, shape, base_lr, momentum, weight_decay, gamma, gpu_ids, enable_cuda, max_epoch, resume, resume_epoch, num_workers, save_frequency, basenet, enable_visdom)
