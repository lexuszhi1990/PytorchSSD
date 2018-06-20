# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from .nms.cpu_nms import cpu_nms, cpu_soft_nms

try:
    from .nms.gpu_nms import gpu_nms
except Exception as e:
    print("no gpu_nms available")

# def nms(dets, thresh, force_cpu=False):
#     """Dispatch to either CPU or GPU NMS implementations."""
#
#     if dets.shape[0] == 0:
#         return []
#     if cfg.USE_GPU_NMS and not force_cpu:
#         return gpu_nms(dets, thresh, device_id=cfg.GPU_ID)
#     else:
#         return cpu_nms(dets, thresh)


def nms(dets, thresh, force_cpu=False, soft_nms=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []

    if force_cpu:
        if soft_nms:
            return cpu_soft_nms(dets, thresh, method=0)
        else:
            return cpu_nms(dets, thresh)
    else:
        return gpu_nms(dets, thresh)
