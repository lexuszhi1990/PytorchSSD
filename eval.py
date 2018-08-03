# -*- coding: utf-8 -*-

import cv2
import time
import logging
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from src.config import config
from src.data.data_augment import BaseTransform
from src.detector import Detector
from src.prior_box import PriorBox
from src.logger import setup_logger
from src.args import get_args
from src.timer import Timer

from src.symbol.RefineSSD_mobilenet_v2 import RefineSSDMobileNet
from src.symbol.RefineSSD_ResNeXt import RefineSSDSEResNeXt


if __name__ == '__main__':

    _t = {'im_detect': Timer(), 'misc': Timer()}
    args = get_args()
    cfg = config[args.config]
    img_path = args.img_path
    ckpt_path = args.ckpt_path
    gpu_ids = [int(i) for i in args.gpu_ids]
    enable_cuda = torch.cuda.is_available() and len(gpu_ids) > 0
    device = torch.device("cuda:%d" % (gpu_ids[0]) if enable_cuda else "cpu")

    workspace = Path(cfg.workspace)
    if not workspace.exists():
        workspace.mkdir(parents=True)
    log_file_path = workspace.joinpath("eval")
    setup_logger(log_file_path.as_posix())

    transform = BaseTransform(cfg.shape, cfg.rgb_means, cfg.rgb_std, (2, 0, 1))
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
    priors = priors.to(device)
    detector = Detector(cfg.num_classes, top_k=cfg.top_k, conf_thresh=cfg.confidence_thresh, nms_thresh=cfg.nms_thresh, variance=cfg.variance, device=device)

    module_lib = globals()[cfg.model_name]
    net = module_lib(cfg=cfg)
    net.initialize_weights(ckpt_path)
    net.to(device)
    net.eval()

    assert Path(img_path).exists(), "%s not exists" % img_path
    img = cv2.imread(img_path)
    basic_scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
    inputs = transform(img).unsqueeze(0)
    with torch.no_grad():
        inputs = torch.Tensor(inputs).to(device)
        _t['im_detect'].tic()
        out = net(x=inputs, inference=True)  # forward pass
        detect_time = _t['im_detect'].toc()

        _t['misc'].tic()
        arm_loc, arm_conf, odm_loc, odm_conf = out
        output = detector.forward((odm_loc, odm_conf), priors, (arm_loc, arm_conf))
        output_np = output.numpy()
        nms_time = _t['misc'].toc()


    for class_id in range(1, cfg.num_classes):
        cls_output = output_np[class_id]
        dets = cls_output[cls_output[:, 1] > 0.60]
        dets[:, 2:6] = np.floor(dets[:, 2:6] * basic_scale)
        for det in dets:
            cls_id, score, left, top, right, bottom = det
            img_det = cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 0), 1)
            img_det = cv2.putText(img_det, '%d:%.3f'%(class_id, score), (int(left), int(top)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            logging.info("cat: %d, score: %.4f" % (cls_id, score))

    saved_path = "%s_%s_det.png" % (Path(img_path).stem, args.config)
    cv2.imwrite(saved_path, img_det)
    logging.info('im_detect: %s, detect_time:%.3fs nms_time:%.3fs\nimage saved at %s'%(img_path, detect_time, nms_time, saved_path))


    # import pdb
    # pdb.set_trace()


    # shape = int(args.shape)
    # ckpt_path = args.ckpt_path
    # top_k = args.top_k
    # nms_thresh = args.nms_thresh
    # config_id = args.config_id
    # confidence_thresh = args.confidence_thresh

    # gpu_ids = [int(i) for i in args.gpu_ids]
    # enable_cuda = args.cuda and torch.cuda.is_available() and len(gpu_ids) > 0
    # if enable_cuda:
    #     torch.set_default_tensor_type('torch.cuda.FloatTensor')
    #     cudnn.benchmark = True

    # setup_logger(workspace)
    # _t = {'im_detect': Timer(), 'misc': Timer()}

    # conf = config.list[config_id]
    # val_trainsform = BaseTransform(conf['shape'], conf['rgb_means'], conf['rgb_std'], (2, 0, 1))
    # priorbox = PriorBox(conf)
    # priors = Variable(priorbox.forward(), volatile=True)
    # detector = Detector(conf['num_classes'], top_k=conf['top_k'], conf_thresh=conf['confidence_thresh'], nms_thresh=conf['nms_thresh'], variance=conf['variance'])

    # module_lib = globals()[conf['model_name']]
    # net = module_lib(num_classes=conf['num_classes'], base_channel_num=conf['base_channel_num'], width_mult=conf['width_mult'], use_refine=conf['use_refine'])

    # net.initialize_weights(ckpt_path)
    # if enable_cuda and len(gpu_ids) > 0:
    #     net = torch.nn.DataParallel(net, device_ids=gpu_ids)
    #     logging.info("initlaize model at gpu" + str(gpu_ids))

    # net.eval()
    # assert Path(image_path).exists(), "%s not exists" % image_path
    # img = cv2.imread(image_path)
    # x = Variable(val_trainsform(img).unsqueeze(0), volatile=True)
    # if enable_cuda:
    #     x = x.cuda()
    # basic_scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]

    # # init module
    # out = net(x=x, inference=True)

    # _t['im_detect'].tic()
    # out = net(x=x, inference=True)  # forward pass
    # detect_time = _t['im_detect'].toc()

    # _t['misc'].tic()
    # arm_loc, arm_conf, odm_loc, odm_conf = out
    # output = detector.forward((odm_loc, odm_conf), priors, (arm_loc, arm_conf))
    # nms_time = _t['misc'].toc()

    # output_np = output.cpu().numpy()

    # for class_id in range(1, conf['num_classes']):
    #     cls_outut = output_np[class_id]
    #     dets = cls_outut[cls_outut[:, 1] > 0.60]
    #     dets[:, 2:6] = np.floor(dets[:, 2:6] * basic_scale)
    #     for det in dets:
    #         cls_id, score, left, top, right, bottom = det
    #         img_det = cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 0), 1)
    #         img_det = cv2.putText(img_det, '%d:%.3f'%(class_id, score), (int(left), int(top)+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
    #         logging.info("cat: %d, score: %.4f" % (cls_id, score))

    # saved_path = "%s_%s_det.png" % (Path(image_path).stem, config_id)
    # cv2.imwrite(saved_path, img_det)
    # logging.info('im_detect: %s, detect_time:%.3fs nms_time:%.3fs\nimage saved at %s'%(image_path, detect_time, nms_time, saved_path))
