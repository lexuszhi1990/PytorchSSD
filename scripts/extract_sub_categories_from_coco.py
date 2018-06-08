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
image_set = 'train2017'
anno_file = Path(root_dir, 'annotations', "instances_%s.json" % (image_set))
coco=COCO(anno_file.as_posix())
len(coco.imgs)

data=json.load(anno_file.open('r'))
new_data={}
new_data['info']=data['info']
new_data['licenses']=data['licenses']
new_data['categories']=data['categories']
new_data['images']=[]
new_data['annotations']=[]

sub_cat_ids = [1]
new_annotations = []
new_images = []

img_ids = []
for anno in data['annotations']:
    if anno['category_id'] in sub_cat_ids:
        new_annotations.append(anno)
        img_ids.append(anno['image_id'])
print(len(new_annotations))

img_ids = set(img_ids)
for anno in data['images']:
    if anno['id'] in img_ids:
        new_images.append(anno)
print("sub images: %d" %(len(new_images)))

new_data['annotations'] = new_annotations
new_data['images'] = new_images

# json.dump(new_data, open('./new_1_instances_train2017.json','w'),indent=4) # indent=4 更加美观显示
json.dump(new_data, open('./instances_person_%s.json' % (image_set), 'w'))
