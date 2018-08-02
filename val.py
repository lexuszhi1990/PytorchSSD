# -*- coding: utf-8 -*-

import numpy as np
import pickle
from pathlib import Path
import logging

import torch

from src.config import config
from src.data.data_augment import BaseTransform
from src.data.coco import COCODet
from src.data.voc import VOCDetection, AnnotationTransform
from src.detector import Detector
from src.prior_box import PriorBox
from src.logger import setup_logger
from src.args import get_args
from src.timer import Timer

from src.symbol.RefineSSD_vgg import RefineSSDVGG
from src.symbol.RefineSSD_mobilenet_v2 import RefineSSDMobileNet
from src.symbol.RefineSSD_ResNeXt import RefineSSDSEResNeXt

def val(net, detector, priors, dataset, transform, device, cfg=None):

    _t = {'im_detect': Timer(), 'misc': Timer()}
    workspace = Path(cfg.workspace)
    num_images = len(dataset)
    all_boxes = [[[] for _ in range(num_images)] for _ in range(cfg.num_classes)]
    det_file = workspace.joinpath('detections.pkl')

    for i in range(num_images):
        img = dataset.pull_image(i)
        basic_scale = [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]

        inputs = transform(img).unsqueeze(0)
        inputs = torch.Tensor(inputs).to(device)

        _t['im_detect'].tic()
        out = net(x=inputs, inference=True)  # forward pass
        detect_time = _t['im_detect'].toc()

        _t['misc'].tic()
        arm_loc, arm_conf, odm_loc, odm_conf = out
        output = detector.forward((odm_loc, odm_conf), priors, (arm_loc, arm_conf))
        output_np = output.cpu().numpy()
        nms_time = _t['misc'].toc()

        for j in range(1, cfg.num_classes):
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
        if cfg.max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1, cfg.num_classes)])
            if len(image_scores) > cfg.max_per_image:
                image_thresh = np.sort(image_scores)[-cfg.max_per_image]
                for j in range(1, cfg.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        if i % 20 == 0:
            logging.info('im_detect: {:d}/{:d} detect_time:{:.3f}s nms_time:{:.3f}s'.format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
        logging.info("dump results to %s" % (det_file))

    if dataset.image_set.find("test") == -1:
        logging.info('Evaluating detections')
        dataset.evaluate_detections(all_boxes, workspace)


if __name__ == '__main__':


    args = get_args()
    cfg = config[args.config]
    ckpt_path = args.ckpt_path
    resume_epoch = args.resume
    gpu_ids = [int(i) for i in args.gpu_ids]
    enable_cuda = torch.cuda.is_available() and len(gpu_ids) > 0
    device = torch.device("cuda:%d" % (gpu_ids[0]) if enable_cuda else "cpu")

    workspace = Path(cfg.workspace)
    if not workspace.exists():
        workspace.mkdir(parents=True)
    log_file_path = workspace.joinpath("val")
    setup_logger(log_file_path.as_posix())

    transform = BaseTransform(cfg.shape, cfg.rgb_means, cfg.rgb_std, (2, 0, 1))
    priorbox = PriorBox(cfg)
    with torch.no_grad():
        priors = priorbox.forward()
    priors = priors.to(device)

    detector = Detector(cfg.num_classes, top_k=cfg.top_k, conf_thresh=cfg.confidence_thresh, nms_thresh=cfg.nms_thresh, variance=cfg.variance, device=device)

    if cfg.dataset.lower() == "voc":
        dataset = VOCDetection(cfg.root_path, cfg.val_sets, None, AnnotationTransform())
    elif cfg.dataset.lower() == "coco":
        dataset = COCODet(cfg.root_path, cfg.val_sets, None)

    module_lib = globals()[cfg.model_name]
    net = module_lib(cfg=cfg)
    net.initialize_weights(ckpt_path)
    logging.info("load weights from %s" % (ckpt_path))
    net.to(device)
    net.eval()

    val(net, detector, priors, dataset, transform, device, cfg=cfg)
