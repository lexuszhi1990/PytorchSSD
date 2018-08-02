import torch
from torch.autograd import Function

from src.utils.box_utils import decode, center_size
from src.nms import pytorch_nms

class Detector(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on confidence
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, top_k=200, conf_thresh=0.05, nms_thresh=0.45, variance=[0.1, 0.2], max_per_image=0, bg_label=0, nms_intersection_class=True, device=torch.device("cpu")):
        self.num_classes = num_classes
        self.bg_label = bg_label
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance
        self.max_per_image = max_per_image
        # TODO: apply nms for output
        self.nms_intersection_class = nms_intersection_class

    def forward(self, pred_data, prior, arm_data=(None, None)):
        """
        Args:
            pred_data:
                loc_data: (tensor) Loc preds from loc layers
                    Shape: [batch,num_priors*4]
                conf_data: (tensor) Shape: Conf preds from conf layers
                    Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        return
            output [num, class_num, top_k, 5]:
                shape: [score: float*4, score: float*1]
        """

        loc, conf = pred_data
        loc_data = loc.data
        conf_data = conf.data
        prior_data = prior.data
        num = loc_data.size(0)  # batch size
        scale = torch.Tensor([320, 320, 320, 320])
        assert num == 1, "num is %d" % num
        index = 0
        num_priors = prior_data.size(0)
        output = torch.zeros(self.num_classes, self.top_k, 6)

        arm_loc, arm_conf = arm_data
        if (arm_loc is not None) or (arm_conf is not None):
            arm_loc_data = arm_loc.data
            arm_conf_data = arm_conf.data
            arm_object_conf = arm_conf_data[:, 1:]
            no_object_index = arm_object_conf <= 0.01
            conf_data[no_object_index.expand_as(conf_data)] = 0

        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        if arm_loc is not None:
            default = decode(arm_loc_data[index], prior_data, self.variance)
            default = center_size(default)
        else:
            default = prior_data
        decoded_boxes = decode(loc_data[index], default, self.variance)
        conf_scores = conf_preds[index].clone()

        for cls_id in range(self.num_classes):
            cls_mask = conf_scores[cls_id].gt(self.conf_thresh)
            scores = conf_scores[cls_id][cls_mask]
            if scores.dim() == 0:
                continue
            decoded_cls_mask = cls_mask.unsqueeze(1).expand_as(decoded_boxes)
            boxes = decoded_boxes[decoded_cls_mask].view(-1, 4)
            # boxes = boxes * scale
            # boxes[boxes < 0] = 0
            # idx of highest scoring and non-overlapping boxes per class
            ids, count = pytorch_nms(boxes, scores, self.nms_thresh, self.top_k)
            cls_list = torch.Tensor([1 for _ in range(self.top_k)]).unsqueeze(1)
            output[cls_id, :count] = torch.cat((cls_list[:count], scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        # output = output.contiguous().view(-1, 6)
        # filter_mask = (output[:, 0] > 0).nonzero()
        # if len(filter_mask) > 0:
        #     return output[filter_mask.squeeze(1)]
        # else:
        #     return torch.zeros(self.top_k, 6)

        return output


class DetectorScratch(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on confidence
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, top_k=200, conf_thresh=0.05, nms_thresh=0.45, variance=[0.1, 0.2], max_per_image=0, bg_label=0, nms_intersection_class=True):
        self.num_classes = num_classes
        self.bg_label = bg_label
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance
        self.max_per_image = max_per_image
        # TODO: apply nms for output
        self.nms_intersection_class = nms_intersection_class

    def forward(self, pred_data, prior, arm_data=(None, None)):
        """
        Args:
            pred_data:
                loc_data: (tensor) Loc preds from loc layers
                    Shape: [batch,num_priors*4]
                conf_data: (tensor) Shape: Conf preds from conf layers
                    Shape: [batch*num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [1,num_priors,4]
        return
            output [num, class_num, top_k, 5]:
                shape: [score: float*4, score: float*1]
        """

        loc, conf = pred_data
        loc_data = loc.data
        conf_data = conf.data
        default_prior = prior.data
        arm_loc, arm_conf = arm_data
        num = loc_data.size(0)  # batch size
        assert num == 1, "num is %d" % num
        index = 0
        num_priors = default_prior.size(0)

        if (arm_loc is not None) or (arm_conf is not None):
            arm_loc_data = arm_loc.data
            arm_conf_data = arm_conf.data
            arm_object_conf = arm_conf_data[:, 1:]
            no_object_index = arm_object_conf <= 0.01
            conf_data[no_object_index.expand_as(conf_data)] = 0

        conf_preds = conf_data.unsqueeze(0)

        if arm_loc:
            default_prior = decode(arm_loc_data[index], prior_data, self.variance)
            default_prior = center_size(default_prior)
        decoded_boxes = decode(loc_data[index], default_prior, self.variance)
        conf_scores = conf_preds[index].clone()

        return decoded_boxes, conf_scores[0]
