import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.utils.box_utils import match, refine_match, log_sum_exp, decode

class RefineMultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching ground truth boxes
           with (default) 'prior boxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, neg_pos_ratio, object_score, variance=[0.1,0.2], bg_class_id=0, enable_cuda=False):
        super(RefineMultiBoxLoss, self).__init__()

        self.num_classes = num_classes
        self.overlap_threshold = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.object_score = object_score
        self.variance = variance
        self.enable_cuda = enable_cuda
        self.bg_class_id = bg_class_id

    def forward(self, odm_data, priors, targets, arm_data=None, filter_object=False):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
            arm_data (tuple): arm branch containg arm_loc and arm_conf
            filter_object: whether filter out the prediction according to the arm conf score
        """

        loc_data, conf_data = odm_data
        if arm_data:
            arm_loc, arm_conf = arm_data
        priors = priors.data
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            gt_loc = targets[idx][:,:-1].data
            gt_cls = targets[idx][:,-1].data > self.bg_class_id

            if self.num_classes > 2:
                gt_cls = gt_cls > self.bg_class_id

            if arm_data:
                loc_t[idx], conf_t[idx] = refine_match(self.overlap_threshold, gt_loc, gt_cls, priors, arm_loc[idx].data, self.variance)
            else:
                loc_t[idx], conf_t[idx] = match(self.overlap_threshold, gt_loc, gt_cls, priors, self.variance)

        if self.enable_cuda:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        # wrap targets
        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t,  requires_grad=False)
        if arm_data and filter_object:
            arm_conf_data = arm_conf.data[:,:,1]
            pos = conf_t > 0
            object_score_index = arm_conf_data <= self.object_score
            pos[object_score_index] = 0
        else:
            pos = conf_t > 0

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1,4)
        loc_t = loc_t[pos_idx].view(-1,4)
        loss_loc = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1,1))
        # Hard Negative Mining
        loss_c[pos] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_cls = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.data.sum()
        loss_loc /= N
        loss_cls /= N
        return loss_loc, loss_cls
