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
# json_v5 = []
# for v in faster_rcnn_json:
#     if v['score'] < 0.5:
#         json_v5.append(v)
# for v in mask_rcnn_json:
#     if v['score'] > 0.5:
#         json_v5.append(v)
# detection_fusion_v5_path = "./fusion_v5.json"
# json.dump(json_v5, Path(detection_fusion_v5_path).open('w'))
# coco_det._do_detection_eval(detection_fusion_v5_path, './')
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.353
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.556
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.382
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.171
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.396
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.490
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.306
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.458
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.471
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.264
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.524
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.645

import numpy as np

def py_cpu_nms(dets, scores, thresh):
    """Pure Python NMS baseline."""
    dets = np.array(dets).astype(np.float)
    scores = np.array(scores).astype(np.float)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2] + x1 - 1
    y2 = dets[:, 3] + y1 - 1
    # scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        weight = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy2 - yy1 + 1)
        inter = weight * height
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

json_v6 = []
for v in faster_rcnn_json:
    if v['score'] > 0.5:
        json_v6.append(v)
json_v6.extend(mask_rcnn_json)

json_v6_res = {}
for res in json_v6:
    cls_id = res['category_id']
    if cls_id in json_v6_res.keys():
        json_v6_res[cls_id].append(res)
    else:
        json_v6_res[cls_id] = [res]

json_v6_filter = []
for cls_id in json_v6_res.keys():
    cls_bbox = [v['bbox'] for v in json_v6_res[cls_id]]
    scores = [v['score'] for v in json_v6_res[cls_id]]
    keep = py_cpu_nms(cls_bbox, scores, 0.85)
    json_v6_filter.extend([json_v6_res[cls_id][i] for i in keep])
    print("clsID: %d, total: %d, keep: %d" % (cls_id, len(json_v6_res[cls_id]), len(keep)))

detection_fusion_v6_path = "./fusion_v6.json"
with Path(detection_fusion_v6_path).open('w') as fid:
    json.dump(json_v6_filter, fid)
coco_det._do_detection_eval(detection_fusion_v6_path, './')

import pdb
pdb.set_trace()
# scores_list = np.array([v['score'] for v in json_v6_filter])
# scores_list = [v['score'] for v in json_v6_filter]

