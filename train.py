# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import numpy as np

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
from src.prior_box import PriorBox, PriorBoxV1
from src.utils import setup_logger, kaiming_weights_init
from src.utils.args import get_args
from src.utils.timer import Timer
from src.utils.nms_wrapper import nms

from src.symbol.RefineSSD_vgg import RefineSSDVGG
from src.symbol.RefineSSD_mobilenet_v2 import RefineSSDMobileNet
from src.symbol.RefineSSD_mobilenet_v2_1 import RefineSSDMobileNetV1
from src.symbol.RefineSSD_ResNeXt import RefineSSDSEResNeXt
from val import val

def train(workspace, model_name, num_classes, train_dataset, val_dataset, val_trainsform, priors, detector, base_channel_num, width_mult, use_refine,batch_size, num_workers, shape, base_lr, momentum, weight_decay, gamma, max_epoch=200, resume_epoch=0, inteval=10, enable_cuda=False, gpu_ids=[], enable_visdom=False, prefix='refinedet_model'):

    if enable_visdom:
        viz = visdom.Visdom()

    workspace_path = Path(workspace)
    val_results_path = workspace_path.joinpath('validation')
    if not val_results_path.exists():
        val_results_path.mkdir(parents=True)

    module_lib = globals()[model_name]
    net = module_lib(num_classes=num_classes, base_channel_num=base_channel_num, width_mult=width_mult, use_refine=use_refine)

    ckpt_path = workspace_path.joinpath("%s-%d.pth" %(prefix, resume_epoch))
    net.initialize_weights(ckpt_path)
    logging.info(net)
    if enable_cuda and len(gpu_ids) > 0:
        net = torch.nn.DataParallel(net, device_ids=gpu_ids)

    timer = Timer()
    if use_refine:
        arm_criterion = MultiBoxLoss(2, overlap_thresh=0.5, neg_pos_ratio=3, enable_cuda=enable_cuda)
        odm_criterion = RefineMultiBoxLoss(num_classes, overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.001, enable_cuda=enable_cuda)
        arm_repulsion_criterion = RepulsionLoss(2, overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.001, enable_cuda=enable_cuda)
    else:
        odm_criterion = MultiBoxLoss(num_classes, overlap_thresh=0.5, neg_pos_ratio=3, enable_cuda=enable_cuda)
        odm_repulsion_criterion = RepulsionLoss(num_classes,    overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.001, enable_cuda=enable_cuda)
    logging.info('Loading datasets...')
    train_dataset_loader = data.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers, collate_fn=detection_collate)
    logging.info('Loading datasets is done...')

    # optimizer = optim.RMSprop(net.parameters(), lr=base_lr, alpha = 0.9, eps=1e-08, momentum=momentum, weight_decay=weight_decay)
    optimizer = optim.SGD(net.parameters(), lr=base_lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[ i*6 for i in range(1, max_epoch//6) ], gamma=0.75)
    arm_loss_l, arm_loss_c, odm_loss_l, odm_loss_c = 0., 0., 0., 0.
    for epoch in range(resume_epoch, max_epoch):
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
                arm_rep_loss = arm_repulsion_criterion((arm_loc, arm_conf), priors, targets)
                odm_loss_l, odm_loss_c = odm_criterion((odm_loc, odm_conf), priors, targets, (arm_loc, arm_conf))
                odm_rep_loss = odm_repulsion_criterion((odm_loc, odm_conf), priors, targets, (arm_loc, arm_conf))
            else:
                odm_loss_l, odm_loss_c = odm_criterion((odm_loc, odm_conf), priors, targets)

            if use_refine:
                if epoch < 50:
                    # loss = 0.5 * (arm_loss_l + arm_loss_c + arm_rep_loss) + 0.5 * (odm_loss_l + odm_loss_c + odm_rep_loss)
                    loss = 0.5 * (arm_loss_l + arm_loss_c) + 0.5 * (odm_loss_l + odm_loss_c)
                else:
                    # loss = 0.2 * (arm_loss_l + arm_loss_c + arm_rep_loss) + 0.8 * (odm_loss_l + odm_loss_c + odm_rep_loss)
                    loss = 0.2 * (arm_loss_l + arm_loss_c) + 0.8 * (odm_loss_l + odm_loss_c)
            else:
                loss = odm_loss_l + odm_loss_c
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration % inteval == 0:
                # logging.info("[%d/%d] || total_loss: %.4f(arm_loc_loss: %.4f, arm_cls_loss: %.4f, , arm_rep_loss: %.4f, odm_loss_l: %.4f, odm_loss_c: %.4f, odm_rep_loss: %.4f) || Batch time: %.4f sec. || LR: %.6f" % (epoch, iteration, loss, arm_loss_l.data[0], arm_loss_c.data[0], 0, odm_loss_l.data[0], odm_loss_c.data[0], 0, timer.average_time, optimizer.param_groups[0]['lr']))
                logging.info("[%d/%d] || total_loss: %.4f(arm_loc_loss: %.4f, arm_cls_loss: %.4f, , arm_rep_loss: %.4f, odm_loss_l: %.4f, odm_loss_c: %.4f, odm_rep_loss: %.4f) || Batch time: %.4f sec. || LR: %.6f" % (epoch, iteration, loss, 0, 0, 0, odm_loss_l.data[0], odm_loss_c.data[0], 0, timer.average_time, optimizer.param_groups[0]['lr']))
                timer.clear()

        if epoch % inteval == 0:
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
    dataset = args.dataset.upper()
    prefix = args.prefix
    resume_epoch = args.resume_epoch
    inteval = args.inteval
    enable_visdom = args.visdom
    gpu_ids = [int(i) for i in args.gpu_ids]
    enable_cuda = args.cuda and torch.cuda.is_available() and len(gpu_ids) > 0
    if enable_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        cudnn.benchmark = True

    workspace_path = Path(workspace)
    if not workspace_path.exists():
        workspace_path.mkdir(parents=True)
    log_file_path = Path(workspace).joinpath("train-%s" % (dataset))
    setup_logger(log_file_path.as_posix())

    conf = config.list[args.config_id]
    val_trainsform = BaseTransform(conf['shape'], conf['rgb_means'], conf['rgb_std'], (2, 0, 1))
    priorbox = PriorBox(conf)
    priors = Variable(priorbox.forward(), volatile=True)
    detector = Detector(conf['num_classes'], top_k=conf['top_k'], conf_thresh=conf['confidence_thresh'], nms_thresh=conf['nms_thresh'], variance=conf['variance'])

    if dataset == "VOC":
        train_dataset = VOCDetection(conf['root_path'], conf['train_sets'], preproc(conf['shape'], conf['rgb_means'], conf['rgb_std'], conf['augment_ratio']), AnnotationTransform())
        val_dataset = VOCDetection(conf['root_path'], conf['val_sets'], None, AnnotationTransform())
    elif dataset == "COCO":
        train_dataset = COCODet(conf['root_path'], conf['train_sets'], preproc(conf['shape'], conf['rgb_means'], conf['rgb_std'], conf['augment_ratio']))
        val_dataset = COCODet(conf['root_path'], conf['val_sets'], None)

    train(workspace, conf['model_name'], conf['num_classes'], train_dataset, val_dataset, val_trainsform, priors, detector, conf['base_channel_num'], conf['width_mult'], conf['use_refine'], conf['batch_size'], conf['num_workers'], conf['shape'], conf['base_lr'], conf['momentum'], conf['weight_decay'], conf['gamma'], conf['max_epoch'], resume_epoch, inteval, enable_cuda, gpu_ids, enable_visdom, prefix)
