# -*- coding: utf-8 -*-

import sys
sys.path.append('.')

from src.utils.pycocotools.coco import COCO
from src.utils.pycocotools.cocoeval import COCOeval


from src.data.coco import COCODet
coco_det = COCODet('/mnt/dataset/coco', 'val2017')

# pure ssd
ssd_detection_path = 'src/test/results/refinedet_ssd_detection.json'
# coco_det._do_detection_eval(ssd_detection_path, './')
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.198
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.338
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.204
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.033
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.197
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.366
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.189
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.256
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.256
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.037
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.254
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.475


# pure mask rcnn
mask_rcnn_detection_path = 'src/test/results/mask-rcnn-detection.json'
# coco_det._do_detection_eval(mask_rcnn_detection_path, './')
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.347
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.543
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.377
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.163
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.389
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.486
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.295
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.423
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.432
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.480
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.602

# pure faster rcnn
faster_rcnn_detection_path = 'src/test/results/faster-rcnn-detection.json'
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.303
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.513
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.318
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.132
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.349
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.432
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.279
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.415
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.426
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.217
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.488
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.598


# fusion results

import json
from pathlib import Path
ssd_json = json.load(Path(ssd_detection_path).open('r'))
faster_rcnn_json = json.load(Path(faster_rcnn_detection_path).open('r'))
mask_rcnn_json = json.load(Path(mask_rcnn_detection_path).open('r'))

# ssd_json_copy = ssd_json.copy()
# ssd_json_copy.extend(mask_rcnn_json)
# detection_fusion_v1_path = "./fusion_v1.json"
# json.dump(ssd_json_copy, Path(detection_fusion_v1_path).open('w'))
# coco_det._do_detection_eval(detection_fusion_v1_path, './')
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.327
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.504
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.357
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.165
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.393
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.452
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.447
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.460
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.217
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.507
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.651

# mask-rcnn and ssd
# json_v2 = []
# for v in ssd_json:
#     if v['score'] > 0.99:
#         json_v2.append(v)
# json_v2.extend(mask_rcnn_json)
# detection_fusion_v2_path = "./fusion_v2.json"
# json.dump(json_v2, Path(detection_fusion_v2_path).open('w'))
# coco_det._do_detection_eval(detection_fusion_v2_path, './')
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.329
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.515
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.357
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.163
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.388
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.458
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.295
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.425
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.434
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.480
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.605


# mask-rcnn(fpn50) and faster-rcnn(vgg16)
# json_v3 = []
# json_v3 = faster_rcnn_json.copy()
# json_v3.extend(mask_rcnn_json)
# detection_fusion_v3_path = "./fusion_v3.json"
# json.dump(json_v3, Path(detection_fusion_v3_path).open('w'))
# coco_det._do_detection_eval(detection_fusion_v3_path, './')
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.302
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.459
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.331
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.173
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.366
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.405
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.310
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.491
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.518
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.290
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.579
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.702


# mask-rcnn(fpn50) and faster-rcnn(vgg16)
# json_v4 = []
# for v in faster_rcnn_json:
#     if v['score'] > 0.5:
#         json_v4.append(v)
# for v in mask_rcnn_json:
#     if v['score'] < 0.5:
#         json_v4.append(v)
# detection_fusion_v4_path = "./fusion_v4.json"
# json.dump(json_v4, Path(detection_fusion_v4_path).open('w'))
# coco_det._do_detection_eval(detection_fusion_v4_path, './')
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.353
# Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.556
# Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.382
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.171
# Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.396
# Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.490
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.306
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.458
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.471
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.264
# Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.524
# Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645


# mask-rcnn(fpn50) and faster-rcnn(vgg16)
json_v5 = []
for v in faster_rcnn_json:
    if v['score'] < 0.5:
        json_v5.append(v)
for v in mask_rcnn_json:
    if v['score'] > 0.5:
        json_v5.append(v)
detection_fusion_v5_path = "./fusion_v5.json"
json.dump(json_v5, Path(detection_fusion_v5_path).open('w'))
coco_det._do_detection_eval(detection_fusion_v5_path, './')
