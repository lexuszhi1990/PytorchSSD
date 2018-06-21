import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from src.utils.box_utils import match, log_sum_exp

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched 'priorboxes'.
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
        super(MultiBoxLoss, self).__init__()

        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.object_score = object_score
        self.variance = variance
        self.enable_cuda = enable_cuda
        self.bg_class_id = bg_class_id

    def forward(self, pred_data, priors_data, gt_data):
        """Multibox Loss
        Args:
            pred_data (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        pred_loc, pred_score = pred_data
        num = pred_loc.size(0)
        num_priors = (priors_data.size(0))

        # match priors (default boxes) and ground truth boxes
        target_loc = torch.Tensor(num, num_priors, 4)
        target_score = torch.LongTensor(num, num_priors)
        for idx in range(num):
            gt_loc = gt_data[idx][:,:-1].data
            gt_cls = gt_data[idx][:,-1].data
            target_loc[idx], target_score[idx] = match(self.overlap_thresh, gt_loc, gt_cls, priors_data, self.variance)

        if self.enable_cuda:
            target_loc = target_loc.cuda()
            target_score = target_score.cuda()
        # wrap gt_data
        target_loc = Variable(target_loc, requires_grad=False)
        target_score = Variable(target_score,requires_grad=False)

        pos = target_score > 0
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(pred_loc)

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pred_loc_ = pred_loc[pos_idx].view(-1, 4)
        target_loc_ = target_loc[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(pred_loc_, target_loc_, size_average=False)

        # Compute max conf across batch for hard negative mining
        batch_score = pred_score.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_score) - batch_score.gather(1, target_score.view(-1, 1))
        # Hard Negative Mining
        loss_c[pos] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.neg_pos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(pred_score)
        neg_idx = neg.unsqueeze(2).expand_as(pred_score)
        filtered_pred_score = pred_score[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = target_score[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(filtered_pred_score, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N

        N = num_pos.data.sum()
        loss_l/=N
        loss_c/=N
        return loss_l, loss_c
