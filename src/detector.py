import torch
from torch.autograd import Function

from .utils.box_utils import decode, center_size, pytorch_nms

class Detector(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on confidence
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, num_classes, bkg_label=0, top_k=100, conf_thresh=0.50, nms_thresh=0.5, variance=[0.1, 0.2], object_score=0, max_per_image=0, filter_results=True):
        self.num_classes = num_classes
        self.bkg_label = bkg_label
        self.object_score = object_score
        self.top_k = top_k
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.variance = variance
        self.max_per_image = max_per_image
        self.filter_results = filter_results

    def forward(self, pred_data, prior_data, arm_data=(None, None)):
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
        num = loc_data.size(0)  # batch size
        assert num == 1, "num is %d" % num
        num_priors = prior_data.size(0)
        output = torch.zeros(self.num_classes, self.top_k, 6)

        arm_loc, arm_conf = arm_data
        if (arm_loc is not None) or (arm_conf is not None):
            arm_loc_data = arm_loc.data
            arm_conf_data = arm_conf.data
            arm_object_conf = arm_conf_data[:, 1:]
            no_object_index = arm_object_conf <= self.object_score
            conf_data[no_object_index.expand_as(conf_data)] = 0

        conf_preds = conf_data.view(num, num_priors, self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.
        for i in range(num):
            if arm_loc is not None:
                default = decode(arm_loc_data[i], prior_data, self.variance)
                default = center_size(default)
            else:
                default = prior_data
            decoded_boxes = decode(loc_data[i], default, self.variance)
            conf_scores = conf_preds[i].clone()

            for cls_id in range(1, self.num_classes):
                cls_mask = conf_scores[cls_id].gt(self.conf_thresh)
                scores = conf_scores[cls_id][cls_mask]
                if scores.dim() == 0:
                    continue
                decoded_cls_mask = cls_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[decoded_cls_mask].view(-1, 4)
                boxes[boxes<0]=0
                # idx of highest scoring and non-overlapping boxes per class
                ids, count = pytorch_nms(boxes, scores, self.nms_thresh, self.top_k)
                cls_list = torch.Tensor([1 for _ in range(self.top_k)]).unsqueeze(1)
                output[cls_id, :count] = torch.cat((cls_list[:count], scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        output = output.contiguous().view(-1, 6)
        # TODO: apply nms for output
        filter_mask = (output[:, 0] > 0).nonzero()
        if len(filter_mask) > 0:
            return output[filter_mask.squeeze(1)]
        else:
            return torch.Tensor([])

        # return output

