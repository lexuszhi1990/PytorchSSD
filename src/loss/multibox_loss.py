import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.box_utils import match

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching ground truth boxes
           with (default) 'prior boxes' that have jaccard index > threshold parameter
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


    def __init__(self, num_classes, overlap_thresh, neg_pos_ratio=3, object_score=0.01, variance=[0.1,0.2], bg_class_id=0, use_arm_barch=False, device=torch.device("cpu")):
        super(MultiBoxLoss, self).__init__()
        self.smooth_l1_loss = nn.SmoothL1Loss(size_average=False)

        self.num_classes = num_classes
        self.overlap_thresh = overlap_thresh
        self.neg_pos_ratio = neg_pos_ratio
        self.object_score = object_score
        self.variance = variance
        self.bg_class_id = bg_class_id
        self.use_arm_barch = use_arm_barch
        self.device = device

    def cross_entropy_loss(self, x, y):
        """
        x: tensor (batch_size*8732, num_classes) batch_size*8732个default box num_classes个类别
        y: tensor (batch_size*8732,) 每条数据的标签，值在0~D-1之间
        return: (batch_size*8732,)
        对所有的default box进行交叉熵的计算
        """
        xmax = x.detach().max() # 标量
        log_sum_exp = torch.log(torch.sum(torch.exp(x-xmax), 1)) + xmax # (batch_size*default_box,)
        # 这个x.gather(1, y.view(-1,1))是(1,batch_size*default_box) 直接和log_sum_exp相减会进行广播到(batch_size*default_box, batch_size*default_box)
        # 所以需要进行squeeze()
        return log_sum_exp - x.gather(1, y.view(-1,1)).squeeze()

    def hard_negative_mining(self, conf_loss, pos):
        """
        conf_loss: tensor (batch_size*8732) 先用非背景类计算loss
        pos: tensor (batch_size, 8732) default box中和label box进行match得到的匹配的框，每个框的匹配程度iou N个图片，每个图片都有8732个default box
        return: neg tensor,(N, 8732) boolean矩阵，为1表示是选出来的negative box
        计算过全部default box的交叉熵之后在从负样本中选出3倍正样本的数目，然后选出来的这些pos neg的交叉熵再进行反向传播
        """
        batch_size, num_boxes = pos.size()
        conf_loss = conf_loss.view(batch_size, -1)
        conf_loss[pos] = 0

        _, idx = conf_loss.sort(dim=1, descending=True)
        _, rank = idx.sort(dim=1)

        num_pos = pos.long().sum(dim=1)
        num_neg = torch.clamp(3*num_pos, max=(num_boxes-int(num_pos.sum())-1))
        num_neg = num_neg.unsqueeze(1)
        neg = rank < num_neg.expand_as(rank)

        return neg

    def forward(self, pred_data, gt_data, priors):
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

        loc_preds, score_preds = pred_data
        batch_size, num_boxes, _ = loc_preds.size()
        num_priors = (priors.size(0))

        loc_targets = torch.Tensor(batch_size, num_priors, 4).to(self.device)
        score_targets = torch.LongTensor(batch_size, num_priors).to(self.device)
        for idx in range(batch_size):
            gt_loc = gt_data[idx][:,:-1]
            gt_cls = gt_data[idx][:,-1]
            if self.use_arm_barch:
                gt_cls = gt_cls > 0
            loc_targets[idx], score_targets[idx] = match(self.overlap_thresh, gt_loc, gt_cls, priors, self.variance)

        pos = score_targets > 0
        num_matched_boxes = pos.detach().sum()

        if num_matched_boxes == 0:
            return torch.zeros((1), requires_grad=True)

        pos_mask = pos.unsqueeze(2).expand_as(loc_preds)
        pos_loc_preds = loc_preds[pos_mask].view(-1, 4)
        pos_loc_targets = loc_targets[pos_mask].view(-1, 4)
        loc_loss = self.smooth_l1_loss(pos_loc_preds, pos_loc_targets)

        conf_loss = self.cross_entropy_loss(score_preds.view(-1, self.num_classes), score_targets.view(-1))
        # mining 出来的negtive box: (batch_size, 8732), 大于零的位置表示是mining出来的
        neg = self.hard_negative_mining(conf_loss, pos)
        neg_mask = neg.unsqueeze(2).expand_as(score_preds)
        pos_mask = pos.unsqueeze(2).expand_as(score_preds)

        mask = (neg_mask + pos_mask).gt(0)
        pos_and_neg = (pos + neg).gt(0)
        filtered_score_preds = score_preds[mask].view(-1, self.num_classes)
        filtered_score_targets = score_targets[pos_and_neg]
        conf_loss = F.cross_entropy(filtered_score_preds, filtered_score_targets, size_average=False)
        loc_loss /= num_matched_boxes.to(torch.float32)
        conf_loss /= num_matched_boxes.to(torch.float32)

        return loc_loss, conf_loss
