# -*- coding: utf-8 -*-

import numpy as np
import pickle
from pathlib import Path
import logging

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from src.config import config
from src.data.data_augment import BaseTransform
from src.data.coco import COCODet
from src.data.voc import VOCDetection, AnnotationTransform
from src.detector import Detector
from src.prior_box import PriorBox
from src.utils import setup_logger, load_weights
from src.utils.args import get_args
from src.utils.timer import Timer

from src.symbol.RefineSSD_vgg import RefineSSDVGG
from src.symbol.RefineSSD_mobilenet_v2 import RefineSSDMobileNet
from src.symbol.RefineSSD_ResNeXt import RefineSSDSEResNeXt

def val(net, detector, priors, num_classes, val_dataset, transform, workspace, ckpt_path=None, enable_cuda=False, max_per_image=300, thresh=0.005):

    workspace_path = Path(workspace)
    if ckpt_path is not None and Path(ckpt_path).exists:
        net = load_weights(net, ckpt_path)

    # dump predictions and ground truth
    num_images = len(val_dataset)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = workspace_path.joinpath('detections.pkl')

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
            logging.info('im_detect: {:d}/{:d} detect_time:{:.3f}s nms_time:{:.3f}s'.format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    logging.info("dump results to %s" % (det_file))

    logging.info('Evaluating detections')
    val_dataset.evaluate_detections(all_boxes, workspace)

if __name__ == '__main__':

    args = get_args()
    workspace = args.workspace
    dataset = args.dataset.upper()
    ckpt_path = args.ckpt_path

    workspace_path = Path(workspace)
    if not workspace_path.exists():
        workspace_path.mkdir(parents=True)
    log_file_path = Path(workspace).joinpath("validate-%s" % (dataset))
    setup_logger(log_file_path.as_posix())

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
    root_path, img_dim, rgb_means, rgb_std = basic_conf.root_path, basic_conf.img_dim, basic_conf.rgb_means, basic_conf.rgb_std

    module_cfg = config.list[args.config_id]
    val_trainsform = BaseTransform(module_cfg['shape'], rgb_means, rgb_std, (2, 0, 1))
    priorbox = PriorBox(module_cfg)
    priors = Variable(priorbox.forward(), volatile=True)
    detector = Detector(module_cfg['num_classes'], top_k=module_cfg['top_k'], conf_thresh=module_cfg['confidence_thresh'], nms_thresh=module_cfg['nms_thresh'], variance=module_cfg['variance'])

    module_lib = globals()[module_cfg['model_name']]
    net = module_lib(module_cfg['num_classes'], base_channel_num=module_cfg['base_channel_num'], width_mult=module_cfg['width_mult'], use_refine=module_cfg['use_refine'])
    if dataset == "VOC":
        val_dataset = VOCDetection(root_path, module_cfg['val_sets'], None, AnnotationTransform())
    elif dataset == "COCO":
        val_dataset = COCODet(root_path, module_cfg['val_sets'], None)

    val(net, detector, priors, module_cfg['num_classes'], val_dataset, val_trainsform, workspace, ckpt_path=ckpt_path, enable_cuda=enable_cuda, max_per_image=300, thresh=0.005)

    # resume_net_path = '/mnt/ckpt/pytorchSSD/Refine_vgg_320/v1/refineDet-model-50.pth'
    # resume_net_path = 'workspace/v2/refineDet-model-280.pth'
    # resume_net_path = '/mnt/ckpt/pytorchSSD/Refine_vgg_320/refinedet_vgg_0516/Refine_vgg_COCO_epoches_250.pth'
    # resume_net_path = '/mnt/ckpt/pytorchSSD/Refine_mobilenet/scratch-v2/refineDet-model-50.pth'

