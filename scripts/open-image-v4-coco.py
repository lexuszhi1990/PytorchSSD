# -*- coding: utf-8 -*-


import cv2
from pathlib import Path


person_labels=['/m/04yx4', '/m/03bt1vf', '/m/01bl7v', '/m/05r655', '/m/01g317'] #merger for 'person'
bag_labels=['/m/0hf58v5', '/m/01940j'] #merger for 'bag'

root_dir = '/mnt/dataset/open-image-v4'
output_dir = '/data/david/open-image-v4/transfered/person'
lbl_dir= root_dir + '/tools/train-annotations-bbox_people_and_bag.csv'
