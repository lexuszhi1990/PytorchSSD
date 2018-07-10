#!/usr/bin/env python3.6
# -*- coding:utf-8 -*-

import os
import math
import json
import numpy as np
import csv
import argparse
import cv2
from pathlib import Path
import string
from random import choice
import secrets

IMAGE_DIR = 'images'
ANNO_DIR = 'annotations'
RESULTS_DIR = 'results'

CAT_LIST = ["car"]

class CocoDatasetGenerator(object):

    def __init__(self, source_dir, image_set, generate_bbox=True, generate_segm=False, generate_kp=False):

        self.source_dir = source_dir
        self.image_set = image_set

        self.generate_bbox = generate_bbox
        self.generate_segm = generate_segm
        self.generate_kp = generate_kp

        self.images = []
        self.categories = []
        self.annotations = []

    def generate_categories(self):
        for index, cat in enumerate(CAT_LIST):
            category = {}
            category['supercategory'] = cat
            category['id'] = index+1
            category['name']= cat

            category['keypoints'] = []
            category['mask_list'] = []
            category['skeleton'] = []

            self.categories.append(category)

    def get_category(self, name):
        for cat in self.categories:
            if cat['name'] == name:
                return cat
        return None

    def new_anno(self, image_id=None, category_id=None, iscrowd=0):
        return   {
                    'id': secrets.randbits(64),
                    'segmentation': [],
                    'bbox': [], 'keypoints': [],
                    'iscrowd': iscrowd,
                    'image_id': image_id,
                    'category_id': category_id
                 }

    def _build_annotation(self, img_id, anno_dict):
        annotations = []

        cat_name = 'car'
        category_id = CAT_LIST.index(cat_name) + 1
        for coord in anno_dict['coordinate'].split(';'):
            if coord == '':
                continue
            try:
                xmin, ymin, width, height = [float(x) for x in coord.split('_')]
            except Exception as e:
                import pdb
                pdb.set_trace()

            anno = self.new_anno(img_id, category_id)

            if self.generate_bbox:
                anno['bbox'] = [xmin, ymin, width, height]
                anno['area'] = width * height

            annotations.append(anno)
        return annotations

    def generate_label(self):
        data_anno_path = Path(self.source_dir, 'annotations', self.image_set + '.csv')
        if not data_anno_path.exists():
            print("path %s not exists" % data_anno_path)
            exit(-1)

        for anno_dict in csv.DictReader(data_anno_path.open('r')):
            image_name = anno_dict['name']
            ab_image_path = Path(self.source_dir, 'images', self.image_set, image_name)
            if not ab_image_path.exists():
                print("Path does not exist: {}".format(ab_image_path))
                continue
            else:
                print('processing %s %s' % (self.image_set, ab_image_path))

            image={}
            image_raw = cv2.imread(ab_image_path.as_posix())
            image['height'], image['width'], _ = image_raw.shape
            image['file_name'] = image_name
            image['id'] = image_name.split('.')[0]
            image['category'] = CAT_LIST[0]
            self.images.append(image)

            if self.image_set.find('test') < 0:
                self.annotations.extend(self._build_annotation(image['id'], anno_dict))

    def data2coco(self):
        coco_set = {}
        coco_set['images'] = self.images
        coco_set['categories'] = self.categories
        coco_set['annotations'] = self.annotations

        return coco_set

    def save(self):
        output_anno_path = Path(self.source_dir, 'annotations', "instances_%s.json" % self.image_set)
        json.dump(self.data2coco(), output_anno_path.open(mode='w+'), indent=4)
        print("save results to %s" % output_anno_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--source_dir', default=None, type=str,
                        dest='source_dir', help='The directory to save the ground truth.')
    parser.add_argument('--image_set', default=None, type=str,
                        dest='image_set', help='train|val|test')
    args = parser.parse_args()

    generator = CocoDatasetGenerator(args.source_dir, args.image_set)
    generator.generate_categories()
    generator.generate_label()
    generator.save()
