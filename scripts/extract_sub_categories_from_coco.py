import sys
sys.path.append('/mnt/dataset/coco/PythonAPI')

from pycocotools.coco import COCO
import os, sys, zipfile
import urllib.request
import shutil
import numpy as np
from pathlib import Path
import pylab
import json
import cv2
from random import randrange
import matplotlib.pyplot as plt

root_dir = '/mnt/dataset/coco'
image_set = 'test2017'
anno_file = Path(root_dir, 'annotations', "instances_%s.json" % (image_set))
coco=COCO(anno_file.as_posix())
len(coco.imgs)

data=json.load(anno_file.open('r'))
new_data={}
new_data['info']=data['info']
new_data['licenses']=data['licenses']
new_data['images']=data['images']
new_data['categories']=data['categories']
new_data['annotations']=[]

sub_cat_ids = [1]
new_annotations = []

for anno in data['annotations']:
    if anno['category_id'] in sub_cat_ids:
        new_annotations.append(anno)
print(len(new_annotations))

new_data['annotations'] = new_annotations

# json.dump(new_data, open('./new_1_instances_train2017.json','w'),indent=4) # indent=4 更加美观显示
json.dump(new_data, open('./instances_person_%s.json' % (image_set), 'w'))
