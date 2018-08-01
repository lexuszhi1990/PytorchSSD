# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

dets = [
    [170, 276, 268, 166],
    [313, 203, 134, 192],
    [334, 407, 157, 174],
    [ 94, 370, 480, 242],
    [148, 221,  93, 260],
    [122, 324, 308, 188],
    [  5, 136, 284, 198],
    [260, 144,  12, 264],
    [396, 195, 105, 208],
    [356, 273, 232, 206],
]
dets = np.array(dets).astype(np.float32)
dets[:, 2] += dets[:, 0]
dets[:, 3] += dets[:, 1]

scores = [
            0.96392,
            0.78246,
            0.56377,
            0.89394,
            0.13090,
            0.16478,
            0.93908,
            0.64257,
            0.60520,
            0.76587,
         ]
scores = np.array(scores).astype(np.float32)

def display(dets, path='nms.png'):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    dets = dets/(np.max(dets) + 5.0)
    for det in dets:
        xmin, ymin, xmax, ymax = det
        rect = plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, edgecolor=(1, 0, 0), linewidth=1)
        ax.add_patch(rect)
    plt.savefig(path, dpi=100)

display(dets, 'propose.png')

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
    areas = (x2 - x1 + 1.) * (y2 - y1 + 1.)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        width = np.maximum(xx2 - xx1 + 1., 0.)
        height = np.maximum(yy2 - yy1 + 1., 0.)
        intersection = width * height
        iou = intersection / (areas[order[1:]] + areas[i] - intersection)

        inds = np.where(iou < threshold)[0]
        order = order[inds+1]


    return keep

keep = nms_scratch(dets, scores, 0.1)
display(dets[keep], 'nms_res.png')
