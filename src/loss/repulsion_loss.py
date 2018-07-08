# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
# from src.utils.box_utils import IoG, decode_new

from src.utils.box_utils import rep_match, decode

class RepulsionLoss(nn.Module):
    def __init__(self, num_classes=2, variance=[0.1, 0.2], sigma=0., overlap_thresh=0.5, neg_pos_ratio=3, object_score=0.01, bg_class_id=0, enable_cuda=False, filter_arm_object=False):
        super(RepulsionLoss, self).__init__()

        self.num_classes = num_classes
        self.variance = variance
        self.sigma = sigma

        self.overlap_thresh = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.object_score = object_score
        self.variance = variance
        self.enable_cuda = enable_cuda
        self.filter_arm_object = filter_arm_object
        self.bg_class_id = bg_class_id

    # TODO
    def smoothln(self, x, sigma=0.):
        pass

    def forward(self, pred_data, priors, gt_data, arm_data=(None, None)):
        """Repulsion Loss
        Args:
            pred_data (tuple): A tuple containing loc preds, conf preds
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """
        pred_loc, pred_score = pred_data
        arm_loc, arm_conf = arm_data
        num = pred_loc.size(0)
        num_priors = (priors.size(0))

        target_loc = torch.Tensor(num, num_priors, 4)
        target_score = torch.LongTensor(num, num_priors)
        for idx in range(num):
            gt_loc = gt_data[idx][:,:-1].data
            gt_cls = gt_data[idx][:,-1].data
            target_loc[idx], target_score[idx] = rep_match(self.overlap_thresh, gt_loc, gt_cls, pred_loc[idx].data, priors.data, self.variance, arm_loc)

        if self.enable_cuda:
            target_loc = target_loc.cuda()
            target_score = target_score.cuda()
        target_loc = Variable(target_loc, requires_grad=False)
        target_score = Variable(target_score, requires_grad=False)

        pos = target_score > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(pred_loc)


        import pdb
        pdb.set_trace()
        pred_loc_ = pred_loc[pos_idx].view(-1, 4)
        target_loc_ = target_loc[pos_idx].view(-1, 4)
        priors_loc_ = priors[pos_idx].view(-1, 4)

        decode(pred_loc, priors.data, self.variance)
        decode(pred_loc_, priors_loc_, variances=self.variance)
        decode(pred_loc_, Variable(priors_loc_.data, requires_grad=False), variances=self.variance)



    # def forward(self, loc_data, ground_data, prior_data):

    #     decoded_boxes = decode_new(loc_data, Variable(prior_data.data, requires_grad=False), self.variance)
    #     iog = IoG(ground_data, decoded_boxes)
    #     # sigma = 1
    #     # loss = torch.sum(-torch.log(1-iog+1e-10))
    #     # sigma = 0
    #     loss = torch.sum(iog)
    #     return loss
