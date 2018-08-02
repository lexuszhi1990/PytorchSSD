# -*- coding: utf-8 -*-

import argparse
from src.utils import str2bool

def get_args():
    parser = argparse.ArgumentParser(description='Refined SSD')

    # train
    parser.add_argument('--config', default='config id', type=str, help='config ID')
    parser.add_argument('--gpu_ids', nargs='+', default=[], help='gpu id')
    parser.add_argument('--cuda', action="store_true", default=False, help='Use cuda to train model')
    parser.add_argument('--resume', default=-1, type=int, help='resume iter for retraining')

    # eval
    parser.add_argument('--image_set', type=str, default='val', help='image set')
    parser.add_argument('--eval_img', type=str, default='None', help='validate image')
    parser.add_argument('--save_results', action="store_true", default=False, help='Use cuda to train model')


    # parser.add_argument('--workspace', default='./workspace')
    # parser.add_argument('--dataset', default='COCO', help='VOC or COCO dataset')
    # parser.add_argument('--batch_size', default=32, type=int, help='Batch size for training')
    # parser.add_argument('--prefix', default='refinedet_model', type=str, help='prefix for saved module name')
    # parser.add_argument('--ckpt_path', type=str, help='pretrained base model')
    # parser.add_argument('--visdom', action="store_true", default=False, help='Use visdom to for loss visualization')
    # parser.add_argument('--resume', action="store_true", default=False, help='resume net for retraining')
    # parser.add_argument('--inteval', default=10, type=int, help='epoch for saving ckpt')


    # network config
    # parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, help='initial learning rate')
    # parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    # parser.add_argument('--gamma', default=0.1, type=float, help='Gamma update for SGD')
    # parser.add_argument('--jaccard_threshold', default=0.5, type=float, help='Min Jaccard index for matching')
    # parser.add_argument('--weight_decay', default=5e-4, type=float, help='Weight decay for SGD')
    # parser.add_argument('--max_epoch', default=200, type=int, help='max epoch for retraining')

    # nms
    # parser.add_argument('--top_k', default=200, type=int, help='Number of roi used for nms')
    # parser.add_argument('--nms_thresh', default=0.45, type=float, help='nms threshold')
    # parser.add_argument('--confidence_thresh', default=0.01, type=float, help='confidence threshold for filtering the roi')



    # desperate
    # parser.add_argument('--shape', default='320', help='320 or 512 input size.')
    # parser.add_argument('--basenet', type=str, default='None', help='pretrained base model')
    # parser.add_argument('--num_workers', default=4, type=int, help='Number of workers used in dataloading')

    args = parser.parse_args()
    return args
