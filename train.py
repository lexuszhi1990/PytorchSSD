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
from torch.optim.lr_scheduler import MultiStepLR

from src.config import config
from src.data.data_augment import detection_collate, BaseTransform, preproc
from src.data.coco import COCODet
from src.data.voc import VOCDetection, AnnotationTransform
from src.loss import RefineMultiBoxLoss, MultiBoxLoss, RepulsionLoss
from src.detector import Detector
from src.prior_box import PriorBox, PriorBoxV1
from src.logger import setup_logger
from src.args import get_args
from src.timer import Timer

from src.symbol.RefineSSD_mobilenet_v2 import RefineSSDMobileNet
from src.symbol.RefineSSD_ResNeXt import RefineSSDSEResNeXt
from val import val


def train(train_dataset, val_dataset, val_trainsform, priors, detector, resume_epoch, device, cfg):

    timer = Timer()
    if cfg.enable_visdom:
        viz = visdom.Visdom()

    workspace = Path(cfg.workspace)
    val_results_path = workspace.joinpath('validation')
    if not val_results_path.exists():
        val_results_path.mkdir(parents=True)
    ckpt_path = workspace.joinpath("%s-%d.pth" %(cfg.prefix, resume_epoch))

    logging.info('Initialize model %s...' % cfg.model_name)
    module_lib = globals()[cfg.model_name]
    net = module_lib(cfg=cfg)
    net.initialize_weights(ckpt_path)
    logging.info(net)
    net.to(device)
    logging.info('Initialize model %s done!' % cfg.model_name)

    if cfg.use_refine:
        arm_criterion = MultiBoxLoss(2, overlap_thresh=0.5, neg_pos_ratio=3, arm_barch=True)
        odm_criterion = RefineMultiBoxLoss(cfg.num_classes, overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.001)
        arm_repulsion_criterion = RepulsionLoss(2, overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.001)
    else:
        odm_criterion = MultiBoxLoss(cfg.num_classes, overlap_thresh=0.5, neg_pos_ratio=3, device=device)
    odm_repulsion_criterion = RepulsionLoss(cfg.num_classes,    overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.001)

    logging.info('Loading datasets...')
    train_dataset_loader = data.DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, collate_fn=detection_collate)
    logging.info('Loading datasets is done!')

    optimizer = optim.SGD(net.parameters(), lr=cfg.base_lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=[ i*6 for i in range(1, cfg.max_epoch//6) ], gamma=0.75)

    arm_loss_l, arm_loss_c, odm_loss_l, odm_loss_c = 0., 0., 0., 0.

    for epoch in range(resume_epoch+1, cfg.max_epoch):
        net.train()
        scheduler.step()
        for iteration, (images, targets) in enumerate(train_dataset_loader):

            images = images.to(device)
            targets = [x.to(device) for x in targets]
            timer.tic()
            arm_loc, arm_conf, odm_loc, odm_conf = net(images)
            timer.toc()

            if cfg.use_refine:
                arm_loss_l, arm_loss_c = arm_criterion((arm_loc, arm_conf), priors, targets)
                arm_rep_loss = arm_repulsion_criterion((arm_loc, arm_conf), priors, targets)
                odm_loss_l, odm_loss_c = odm_criterion((odm_loc, odm_conf), priors, targets, (arm_loc, arm_conf))
                odm_rep_loss = odm_repulsion_criterion((odm_loc, odm_conf), priors, targets, (arm_loc, arm_conf))
            else:
                odm_loss_l, odm_loss_c = odm_criterion((odm_loc, odm_conf), targets, priors)

            if cfg.use_refine:
                if epoch < 50:
                    loss = 0.5 * (arm_loss_l + arm_loss_c) + 0.5 * (odm_loss_l + odm_loss_c)
                else:
                    loss = 0.2 * (arm_loss_l + arm_loss_c) + 0.8 * (odm_loss_l + odm_loss_c)
            else:
                loss = odm_loss_l + odm_loss_c
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if iteration % cfg.inteval == 0:
                logging.info("[%d/%d] || total_loss: %.4f(arm_loc_loss: %.4f, arm_cls_loss: %.4f, , arm_rep_loss: %.4f, odm_loss_l: %.4f, odm_loss_c: %.4f, odm_rep_loss: %.4f) || Batch time: %.4f sec. || LR: %.6f" % (epoch, iteration, loss, arm_loss_l, arm_loss_c, 0, odm_loss_l, odm_loss_c, 0, timer.average_time, optimizer.param_groups[0]['lr']))
                timer.clear()

        if epoch % cfg.inteval == 0:
            net.eval()
            save_ckpt_path = workspace.joinpath("%s-%d.pth" %(cfg.prefix, epoch))
            torch.save(net.state_dict(), save_ckpt_path)
            logging.info("save model to %s " % save_ckpt_path)
            import pdb
            pdb.set_trace()
            val(net, detector, priors, val_dataset, val_trainsform, device, cfg=cfg)
            net.train()

    final_model_path = workspace.joinpath("Final-refineDet-%d.pth" %(epoch)).as_posix()
    torch.save(net.state_dict(), final_model_path)
    logging.info("save final model to %s " % final_model_path)

if __name__ == '__main__':

    args = get_args()
    cfg = config[args.config]
    resume_epoch = args.resume
    gpu_ids = [int(i) for i in args.gpu_ids]
    enable_cuda = torch.cuda.is_available() and len(gpu_ids) > 0
    device = torch.device("cuda:%d" % (gpu_ids[0]) if enable_cuda else "cpu")

    workspace = Path(cfg.workspace)
    if not workspace.exists():
        workspace.mkdir(parents=True)
    log_file_path = workspace.joinpath("train")
    setup_logger(log_file_path.as_posix())

    val_trainsform = BaseTransform(cfg.shape, cfg.rgb_means, cfg.rgb_std, (2, 0, 1))
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
    priors = priors.to(device)

    detector = Detector(cfg.num_classes, top_k=cfg.top_k, conf_thresh=cfg.confidence_thresh, nms_thresh=cfg.nms_thresh, variance=cfg.variance, device=device)

    if cfg.dataset.lower() == "voc":
        train_dataset = VOCDetection(cfg.root_path, cfg.train_sets, preproc(cfg.shape, cfg.rgb_means, cfg.rgb_std, cfg.augment_ratio), AnnotationTransform())
        val_dataset = VOCDetection(cfg.root_path, cfg.val_sets, None, AnnotationTransform())
    elif cfg.dataset.lower() == "coco":
        train_dataset = COCODet(cfg.root_path, cfg.train_sets, preproc(cfg.shape, cfg.rgb_means, cfg.rgb_std, cfg.augment_ratio))
        val_dataset = COCODet(cfg.root_path, cfg.val_sets, None)

    train(train_dataset, val_dataset, val_trainsform, priors, detector, resume_epoch, device, cfg)
