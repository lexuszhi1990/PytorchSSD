# -*- coding: utf-8 -*-

import json
import pickle
import logging

import cv2
import numpy as np
import os
from pathlib import Path
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import sys
sys.path.append('../../3rdparth/cocoapi/PythonAPI')
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCODet(data.Dataset):
    """VOC Detection Dataset Object

    input is image, target is annotation

    Arguments:
        root (string): filepath to VOCdevkit folder.
        image_set (string): imageset to use (eg. 'train', 'val', 'test')
        transform (callable, optional): transformation to perform on the
            input image
        target_transform (callable, optional): transformation to perform on the
            target `annotation`
            (eg: take in caption string, return tensor of word indices)
        dataset_name (string, optional): which dataset to load
            (default: 'VOC2007')
    """

    def __init__(self, root, image_set, preproc=None, target_transform=None,
                 dataset_name='COCO'):
        self.root = root
        self.cache_path = Path(self.root, 'cache')
        self.image_set = image_set
        self.preproc = preproc
        self.target_transform = target_transform
        self.name = dataset_name
        self.ids = list()
        self.annotations = list()

        self.coco_anno_path = Path(self.root, 'annotations', "instances_%s.json" % (image_set))
        assert self.coco_anno_path.exists(), "coco annotation path not exists: %s" % self.coco_anno_path

        self._COCO = COCO(self.coco_anno_path.as_posix())
        self.cats = self._COCO.loadCats(self._COCO.getCatIds())
        self._classes = tuple(['__background__'] + [c['name'] for c in self.cats])
        self.num_classes = len(self._classes)
        self._class_to_ind = dict(zip(self._classes, range(self.num_classes)))
        self._class_to_coco_cat_id = dict(zip([c['name'] for c in self.cats], self._COCO.getCatIds()))
        self.image_indexes = self._COCO.getImgIds()
        self.ids = [self.image_path_from_index(index) for index in self.image_indexes]

        if image_set.find('test') != -1:
            logging.info('test set will not load annotations!')
        else:
            self.annotations = self._load_coco_annotations()

        if not self.cache_path.exists():
            self.cache_path.mkdir()
            logging.info('create cache path %s ' % self.cache_path)

    def image_path_from_index(self, index):
        """
        Construct an image path from the image's "index" identifier.
        """
        # Example image path for index=119993:
        #   coco/images/train2017/000000119993.jpg
        # image_path = Path(self.root, 'images', self.image_set.split('_')[-1], str(index).zfill(12) + '.jpg')
        image_path = Path(self.root, 'images', self.image_set, str(index).zfill(12) + '.jpg')
        assert image_path.exists(), "%s not exists" % image_path
        return image_path.as_posix()

    def _load_coco_annotations(self):
        cache_file = self.cache_path.joinpath(self.image_set + '_gt_roidb.pkl')
        if cache_file.exists():
            with cache_file.open('rb') as fid:
                roidb = pickle.load(fid)
            logging.info('{} gt roidb loaded from {}'.format(self.image_set, cache_file))
            return roidb

        gt_roidb = [self._annotation_from_index(index)
                    for index in self.image_indexes]
        with open(cache_file, 'wb') as fid:
            pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        logging.info('wrote gt roidb to {}'.format(cache_file))

        return gt_roidb

    def _annotation_from_index(self, index):
        """
        Loads COCO bounding-box instance annotations. Crowd instances are
        handled by marking their overlaps (with all categories) to -1. This
        overlap value means that crowd "instances" are excluded from training.
        """
        im_ann = self._COCO.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self._COCO.getAnnIds(imgIds=index, iscrowd=None)
        objs = self._COCO.loadAnns(annIds)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        for obj in objs:
            x1 = np.max((0, obj['bbox'][0]))
            y1 = np.max((0, obj['bbox'][1]))
            x2 = np.min((width - 1, x1 + np.max((0, obj['bbox'][2] - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, obj['bbox'][3] - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
        objs = valid_objs
        num_objs = len(objs)

        res = np.zeros((num_objs, 5))

        # Lookup table to map from COCO category ids to our internal class
        # indices
        coco_cat_id_to_class_ind = dict([(self._class_to_coco_cat_id[cls],
                                          self._class_to_ind[cls])
                                         for cls in self._classes[1:]])

        for ix, obj in enumerate(objs):
            cls = coco_cat_id_to_class_ind[obj['category_id']]
            res[ix, 0:4] = obj['clean_bbox']
            res[ix, 4] = cls

        return res

    def __getitem__(self, index):
        img_id = self.ids[index]
        target = self.annotations[index]
        img = cv2.imread(img_id, cv2.IMREAD_COLOR)
        height, width, _ = img.shape

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.preproc is not None:
            img, target = self.preproc(img, target)

            # target = self.target_transform(target, width, height)
        # logging.info(target.shape)

        return img, target

    def __len__(self):
        return len(self.ids)

    def pull_image(self, index):
        '''Returns the original image object at index in PIL form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            PIL img
        '''
        img_id = self.ids[index]
        return cv2.imread(img_id, cv2.IMREAD_COLOR)

    def pull_tensor(self, index):
        '''Returns the original image at an index in tensor form

        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            tensorized version of img, squeezed
        '''
        to_tensor = transforms.ToTensor()
        return torch.Tensor(self.pull_image(index)).unsqueeze_(0)

    def _print_detection_eval_metrics(self, coco_eval):
        IoU_lo_thresh = 0.5
        IoU_hi_thresh = 0.95

        def _get_thr_ind(coco_eval, thr):
            ind = np.where((coco_eval.params.iouThrs > thr - 1e-5) &
                           (coco_eval.params.iouThrs < thr + 1e-5))[0][0]
            iou_thr = coco_eval.params.iouThrs[ind]
            assert np.isclose(iou_thr, thr)
            return ind

        ind_lo = _get_thr_ind(coco_eval, IoU_lo_thresh)
        ind_hi = _get_thr_ind(coco_eval, IoU_hi_thresh)
        # precision has dims (iou, recall, cls, area range, max dets)
        # area range index 0: all area ranges
        # max dets index 2: 100 per image
        precision = \
            coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
        ap_default = np.mean(precision[precision > -1])
        logging.info('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
              '~~~~'.format(IoU_lo_thresh, IoU_hi_thresh))
        logging.info('{:.1f}'.format(100 * ap_default))
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            # minus 1 because of __background__
            precision = coco_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
            ap = np.mean(precision[precision > -1])
            logging.info('{:.1f}'.format(100 * ap))

        logging.info('~~~~ Summary metrics ~~~~')
        coco_eval.summarize()

    def _do_detection_eval(self, res_file, output_dir):
        ann_type = 'bbox'
        coco_dt = self._COCO.loadRes(res_file)
        coco_eval = COCOeval(self._COCO, coco_dt)
        coco_eval.params.useSegm = (ann_type == 'segm')
        coco_eval.evaluate()
        coco_eval.accumulate()
        self._print_detection_eval_metrics(coco_eval)
        eval_file = Path(output_dir, 'detection_results.pkl')
        with eval_file.open('wb') as fid:
            pickle.dump(coco_eval, fid, pickle.HIGHEST_PROTOCOL)
        logging.info('Wrote COCO eval results to: {}'.format(eval_file))

    def _coco_results_one_category(self, boxes, cat_id):
        results = []
        for im_ind, index in enumerate(self.image_indexes):
            dets = boxes[im_ind].astype(np.float)
            if dets == []:
                continue
            scores = dets[:, -1]
            xs = dets[:, 0]
            ys = dets[:, 1]
            ws = dets[:, 2] - xs + 1
            hs = dets[:, 3] - ys + 1
            results.extend(
                [{'image_id': index,
                  'category_id': cat_id,
                  'bbox': [xs[k], ys[k], ws[k], hs[k]],
                  'score': scores[k]} for k in range(dets.shape[0])])
        return results

    def _write_coco_results_file(self, all_boxes, res_file):
        # [{"image_id": 42,
        #   "category_id": 18,
        #   "bbox": [258.15,41.29,348.26,243.78],
        #   "score": 0.236}, ...]
        results = []
        for cls_ind, cls in enumerate(self._classes):
            if cls == '__background__':
                continue
            logging.info('Collecting {} results ({:d}/{:d})'.format(cls, cls_ind, self.num_classes))
            coco_cat_id = self._class_to_coco_cat_id[cls]
            results.extend(self._coco_results_one_category(all_boxes[cls_ind], coco_cat_id))

        logging.info('Writing results json to {}'.format(res_file))
        with open(res_file, 'w') as fid:
            json.dump(results, fid)

    def evaluate_detections(self, all_boxes, output_dir):
        res_file = Path(output_dir, "detections_%s_results.json" % (self.image_set)).as_posix()
        self._write_coco_results_file(all_boxes, res_file)
        # Only do evaluation on non-test sets
        if self.image_set.find('test') == -1:
            self._do_detection_eval(res_file, output_dir)
        # Optionally cleanup results json file
