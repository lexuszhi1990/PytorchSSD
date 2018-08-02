import numpy as np
import torch
from collections import OrderedDict

from data import COCOroot, VOC_300, VOC_512, COCO_300, COCO_320, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc
from models.RefineSSD_vgg import build_net
from layers.functions import Detect,PriorBox
from utils.nms_wrapper import nms

ckpt_path = '/mnt/ckpt/pytorchSSD/Refine_vgg_320/refinedet_vgg_0516/Refine_vgg_COCO_epoches_250.pth'
num_classes=81

image_id=31

cfg = COCO_320
priorbox = PriorBox(cfg)
detector = Detect(num_classes,0,cfg,object_score=0.01)
priors = Variable(priorbox.forward(), volatile=True)

testset = COCODetection(COCOroot, [('2017', 'val')], None)
img = testset.pull_image(image_id)

# >>> import cv2
# >>> cv2.imwrite('asdasd.png', img)

net = build_net(320, num_classes, use_refine=True)
print(net)

state_dict = torch.load(ckpt_path)

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    if k[:7] == 'module.':
        name = k[7:] # remove `module.`
    else:
        name = k
    new_state_dict[name] = v

net.load_state_dict(new_state_dict)

rgb_std = (1,1,1)
rgb_means = (104, 117, 123)
transform = BaseTransform(net.size, rgb_means,rgb_std, (2, 0, 1))
x = Variable(transform(img).unsqueeze(0), volatile=True)
x = x.cuda()

out = net(x=x, test=True)  # forward pass
arm_loc,arm_conf,odm_loc,odm_conf = out
boxes, scores = detector.forward((odm_loc,odm_conf), priors,(arm_loc,arm_conf))
boxes = boxes[0]
scores=scores[0]

boxes = boxes.cpu().numpy()
scores = scores.cpu().numpy()
# scale each detection back up to the image
scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).cpu().numpy()
boxes *= scale

j=1
thresh=0.55
inds = np.where(scores[:, j] > thresh)[0]
if len(inds) == 0:
    all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
    continue
c_bboxes = boxes[inds]
c_scores = scores[inds, j]
c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
    np.float32, copy=False)

num_images = len(testset)
all_boxes = [[[] for _ in range(num_images)] for _ in range(num_classes)]
for j in range(1, num_classes):
    inds = np.where(scores[:, j] > 0.1)[0]
    if len(inds) == 0:
        all_boxes[j][image_id] = np.empty([0, 5], dtype=np.float32)
        continue
    print("dets for class %d" % j)
    c_bboxes = boxes[inds]
    c_scores = scores[inds, j]
    c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
        np.float32, copy=False)
    keep = nms(c_dets, 0.45, force_cpu=True)
    keep = keep[:50]
    c_dets = c_dets[keep, :]
    all_boxes[j][image_id] = c_dets

testset.evaluate_detections(all_boxes, save_folder)

img_det=img.copy()
left, top, right, bottom, score = all_boxes[59][0][0]
cv2.rectangle(img_det, (left, top), (right, bottom), (255, 255, 0), 1)
cv2.putText(img_det, '%.3f'%score, (int(left), int(top)+30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1)
cv2.imwrite("./test_1.png", img_det)
print("save results at %s" % save_path)


def test_net(save_folder, net, detector, cuda, testset, transform, max_per_image=300, thresh=0.005):

    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    # dump predictions and assoc. ground truth to text file for now
    num_images = len(testset)
    num_classes = (21, 81)[args.dataset == 'COCO']
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]

    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(save_folder, 'detections.pkl')

    if args.retest:
        f = open(det_file,'rb')
        all_boxes = pickle.load(f)
        print('Evaluating detections')
        testset.evaluate_detections(all_boxes, save_folder)
        return


    for i in range(num_images):
        img = testset.pull_image(i)
        x = Variable(transform(img).unsqueeze(0),volatile=True)
        if cuda:
            x = x.cuda()

        _t['im_detect'].tic()
        out = net(x=x, test=True)  # forward pass
        arm_loc,arm_conf,odm_loc,odm_conf = out
        boxes, scores = detector.forward((odm_loc, odm_conf), priors,(arm_loc,arm_conf))
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores=scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        boxes *= scale

        _t['misc'].tic()

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > thresh)[0]
            if len(inds) == 0:
                all_boxes[j][i] = np.empty([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            if args.dataset == 'VOC':
                cpu = False
            else:
                cpu = False

            keep = nms(c_dets, 0.45, force_cpu=cpu)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j][i] = c_dets
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1] for j in range(1,num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]

        nms_time = _t['misc'].toc()

        if i % 20 == 0:
            print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'
                  .format(i + 1, num_images, detect_time, nms_time))
            _t['im_detect'].clear()
            _t['misc'].clear()

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    if args.dataset == 'VOC':
        APs,mAP = testset.evaluate_detections(all_boxes, save_folder)
        return APs,mAP
    else:
        testset.evaluate_detections(all_boxes, save_folder)

if __name__ == '__main__':
    train()
