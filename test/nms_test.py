# -*- coding: utf-8 -*-

import numpy as np

def nms_scratch(dets, scores, threshold):
    '''
        dets: Nx4. [xmin, ymin, xmax, ymax]
        scores: Nx1
        threshold: scalar, [0, 1]
    '''

    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]

    bbox_areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    scores_index = scores.argsort()[::-1]

    keep = []
    while scores_index.size > 0:
        i = 0
        keep.append(i)
        xx1 = np.maximum(dets[i, 0], dets[scores_index[1:], 0])
        yy1 = np.maximum(dets[i, 1], dets[scores_index[1:], 1])
        xx2 = np.minimum(dets[i, 2], dets[scores_index[1:], 2])
        yy2 = np.minimum(dets[i, 3], dets[scores_index[1:], 3])

        weight = np.maximum(0.0, xx2 - xx1 + 1)
        height = np.maximum(0.0, yy1 - yy1 + 1)

        inter_section = weight * height
        iou = inter_section / (bbox_areas[i] + inter_section[scores_index[i:]] - inter_section)

        indx = np.where(iou < threshold)[0]
        scores_index = scores_index[inds + 1]

