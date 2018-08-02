### setup by docker image

on 177 server:
docker run --network host --ipc host -v /home/fulingzhi/workspace/PytorchSSD:/app -v /mnt/gf_mnt/datasets/cocoapi:/mnt/dataset/coco -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22-dev bash

on 172 server:
docker run --network host --ipc host -v /home/david/fashionAI/PytorchSSD:/app -v /data/david/cocoapi:/mnt/dataset/coco -v /data/david/open-image-v4:/mnt/dataset/open-image-v4 -v /data/david/models/pytorchSSD:/mnt/ckpt/pytorchSSD -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22-dev bash

local dev:
docker run --name py-ssd --network host --ipc host -v /Users/david/repo/detection/PytorchSSD:/app -v /Users/david/mnt/data/VOCdevkit:/mnt/dataset/voc2012 -it --rm floydhub/pytorch:0.3.1-py3.30 bash

on work-station:
docker run --name py-ssd --network host --ipc host -v /mnt/workspace/david/PytorchSSD:/app -v /mnt/datasets/pascal-voc/VOCdevkit:/mnt/dataset/VOCdevkit -it --rm floydhub/pytorch:0.3.1-py3.30 bash

docker run --name py-ssd --network host --ipc host -v /mnt/workspace/david/PytorchSSD:/app -v /mnt/datasets/station_car:/mnt/dataset/car -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22 bash

docker run --name py-ssd --network host --ipc host -v /mnt/workspace/david/PytorchSSD:/app -v /mnt/datasets/coco/cocoapi:/mnt/dataset/car -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22 bash

set nms:
```
cd src/utils
python build.py build_ext --inplace
```

### troubleshoot

#### pyinn

from pyinn.modules import Conv2dDepthwise

`pip install git+https://github.com/szagoruyko/pyinn.git@master`

#### torch dataloader

```
/usr/local/lib/python3.6/site-packages/torch/autograd/_functions/tensor.py:447: UserWarning: mask is not broadcastable to self, but they have the same number of elements.  Falling back to deprecated pointwise behavior.
  return tensor.masked_fill_(mask, value)
Epoch:1 || epochiter: 0/14785|| Totel iter 0 || L: 0.3921 C: 2.5372||Batch time: 6.4400 sec. ||LR: 0.00400000
Traceback (most recent call last):
  File "train_test.py", line 459, in <module>
    train()
  File "train_test.py", line 307, in train
    images, targets = next(batch_iterator)
  File "/usr/local/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 204, in __next__
    idx, batch = self.data_queue.get()
  File "/usr/local/lib/python3.6/multiprocessing/queues.py", line 344, in get
    return _ForkingPickler.loads(res)
  File "/usr/local/lib/python3.6/site-packages/torch/multiprocessing/reductions.py", line 70, in rebuild_storage_fd
    fd = df.detach()
  File "/usr/local/lib/python3.6/multiprocessing/resource_sharer.py", line 57, in detach
    with _resource_sharer.get_connection(self._id) as conn:
  File "/usr/local/lib/python3.6/multiprocessing/resource_sharer.py", line 87, in get_connection
    c = Client(address, authkey=process.current_process().authkey)
  File "/usr/local/lib/python3.6/multiprocessing/connection.py", line 487, in Client
    c = SocketClient(address)
  File "/usr/local/lib/python3.6/multiprocessing/connection.py", line 614, in SocketClient
    s.connect(address)
ConnectionRefusedError: [Errno 111] Connection refused
```

https://github.com/pytorch/pytorch#docker-image
http://noahsnail.com/2018/01/15/2018-01-15-PyTorch%20socket.error%20[Errno%20111]%20Connection%20refused/


#### data training

```
Traceback (most recent call last):
  File "train_test.py", line 456, in <module>
    train()
  File "train_test.py", line 307, in train
    images, targets = next(batch_iterator)
  File "/usr/local/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 196, in __next__
    return self._process_next_batch(batch)
  File "/usr/local/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 230, in _process_next_batch
    raise batch.exc_type(batch.exc_msg)
AttributeError: Traceback (most recent call last):
  File "/usr/local/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 42, in _worker_loop
    samples = collate_fn([dataset[i] for i in batch_indices])
  File "/usr/local/lib/python3.6/site-packages/torch/utils/data/dataloader.py", line 42, in <listcomp>
    samples = collate_fn([dataset[i] for i in batch_indices])
  File "/app/data/coco.py", line 167, in __getitem__
    height, width, _ = img.shape
AttributeError: 'NoneType' object has no attribute 'shape'
```

solution: `./make.sh`


### daily log

#### 2018.5.16

train RFB_SSD:
python train_test.py -d COCO -v RFB_vgg -s 300 --batch_size 14 --visdom True --send_images_to_visdom False --basenet weights/pretrained/vgg16_reducedfc.pth --gpu_ids 0 --save_folder /mnt/ckpt/pytorchSSD --date RFB_vgg_0516

train refinedet_SSD:
CUDA_VISIBLE_DEVICES=1 python refinedet_train_test.py -d COCO -v Refine_vgg -s 320 --batch_size 14 --visdom True --send_images_to_visdom False --basenet weights/pretrained/vgg16_reducedfc.pth --gpu_ids 0 --save_folder /mnt/ckpt/pytorchSSD --date refinedet_vgg_0516

CUDA_VISIBLE_DEVICES=1,2 python refinedet_train_test.py -d COCO -v Refine_vgg -s 320 --visdom True --send_images_to_visdom False --basenet weights/pretrained/vgg16_reducedfc.pth --save_folder /mnt/ckpt/pytorchSSD --date refinedet_vgg_0516 --gpu_ids 0,1 --batch_size 56

#### 2018.6.6

refactor the code
CUDA_VISIBLE_DEVICES=1,2 python refinedet_train_test.py --dataset COCO --gpu_ids 0,1 --cuda

#### 2018.6.7

CUDA_VISIBLE_DEVICES=2,3 python refinedet_train.py --dataset COCO --gpu_ids 0 1 --cuda --lr 0.05

CUDA_VISIBLE_DEVICES=1 python refinedet_train.py --dataset COCO --gpu_ids 0 --cuda --lr 0.05

#### 2018.6.8

python refinedet_val.py

Collecting person results (1/81) : 25.8
Collecting bicycle results (2/81) : 35.0
Collecting car results (3/81) : 15.7
Collecting motorcycle results (4/81) : 23.0

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.276
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.479
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.288
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.103
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.320
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.416
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.252
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.388
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.414
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.178
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.487
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.607

|Method|Data|Backbone|AP|AP/50|AP/75|AP/S|AP/M|AP/L|
|------|----|--------|--|-----|-----|----|----|----|
|RefineDet320-caffe|trainval35k|VGG-16|29.4|49.2|31.3|10.0|32.0|44.4|
|RefineDet320-pytorch|trainval35k|VGG-16|27.6|47.9|28.8|10.3|32.0|41.
6|


#### 2018.6.11

~~~~ Mean and per-category AP @ IoU=[0.50,0.95] ~~~~
32.8
32.8
~~~~ Summary metrics ~~~~
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.328
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.645
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.297
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.136
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.415
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.540
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.146
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.368
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.424
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.214
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.510
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.639
Wrote COCO eval results to: workspace/val-v2/detection_results.pkl


#### 2018.6.12

~~~~ Summary metrics ~~~~
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.368
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.677
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.364
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.156
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.466
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.590
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.159
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.402
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.455
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.547
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.680
Wrote COCO eval results to: workspace/val-v2/detection_results.pkl

train coco :
CUDA_VISIBLE_DEVICES=6,7 python refinedet_train.py --dataset COCO --gpu_ids 0 1 --cuda --batch_size 64 --workspace /mnt/ckpt/pytorchSSD/Refine_vgg_320/v1 --num_workers 8

#### 2018.6.13

CUDA_VISIBLE_DEVICES=5 python refinedet_val.py --dataset COCO --gpu_ids 0 --cuda

config.coco.train_sets = 'person_train2017'
config.coco.val_sets = 'person_val2017'
config.coco.num_classes = 2
resume_net_path = 'workspace/v2/refineDet-model-220.pth'

 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.372
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.685
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.369

#### 2018.6.14

CUDA_VISIBLE_DEVICES=7 python refinedet_val.py --dataset COCO --gpu_ids 0 --cuda

config.coco.train_sets = 'person_train2017'
config.coco.val_sets = 'person_val2017'
config.coco.num_classes = 2
resume_net_path = 'workspace/v2/refineDet-model-280.pth'

Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.376
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.687
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.376

train person from scratch:
CUDA_VISIBLE_DEVICES=6,7 python refinedet_train.py --dataset COCO --gpu_ids 0 1 --cuda --batch_size 64 --workspace /mnt/ckpt/pytorchSSD/Refine_vgg_320/scratch-v1 --num_workers 8 --basenet None --lr 0.005
不收敛。

#### 2018.6.15

CUDA_VISIBLE_DEVICES=6,7 python refinedet_train.py --dataset COCO --gpu_ids 0 1 --cuda --batch_size 64 --workspace /mnt/ckpt/pytorchSSD/Refine_vgg_320/scratch-v2 --num_workers 8 --basenet None --lr 0.0005

不收敛。

#### 2018.6.19

177 machine:
`CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 --cuda --batch_size 64 --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/scratch-v1 --num_workers 4 --basenet None --lr 0.0001`

`CUDA_VISIBLE_DEVICES=3 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 --cuda --batch_size 2 --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/scratch-test --num_workers 4 --basenet None --lr 0.001`

mac:
`python refinedet_mobile_train.py --dataset voc --workspace ./workspace/v1 --num_workers 2 --batch_size 8`


#### 2018.6.20

CUDA_VISIBLE_DEVICES=5 python refinedet_val.py --dataset COCO --gpu_ids 0 --cuda

CUDA_VISIBLE_DEVICES=5 python refinedet_mobile_val.py --dataset COCO --gpu_ids 0 --cuda --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/scratch-v1/refineDet-model-60.pth

on ws:
python refinedet_mobile_train.py --dataset voc --workspace ./workspace/v1 --num_workers 4 --batch_size 8 --lr 0.004

tag v2:
person 32%

#### 2018.6.22

CUDA_VISIBLE_DEVICES=0 python refinedet_mobile_val.py --dataset COCO --gpu_ids 0 --cuda --confidence_thresh 0.01 --top_k 200 --nms_thresh 0.45 --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v2/refinedet_model-40.pth

CUDA_VISIBLE_DEVICES=5 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 --cuda --batch_size 128 --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/scratch-v3 --num_workers 8 --basenet None --lr 0.004

scratch-v2: no arm branch, width 0.5
scratch-v3: arm branch, width 0.5
scratch-v4: arm branch, width 0.75

CUDA_VISIBLE_DEVICES=1 python refinedet_mobile_val.py --dataset COCO --gpu_ids 0 --cuda --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/scratch-v2/refinedet_model-70.pth

CUDA_VISIBLE_DEVICES=1 python refinedet_mobile_val.py --dataset COCO --gpu_ids 0 --cuda --confidence_thresh 0.01 --top_k 200 --nms_thresh 0.45 --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v2/refinedet_model-40.pth

CUDA_VISIBLE_DEVICES=5 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 --cuda --batch_size 96 --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v1 --num_workers 8 --confidence_thresh 0.05 --lr 0.04

CUDA_VISIBLE_DEVICES=6 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v2 --batch_size 64 --num_workers 8 --confidence_thresh 0.05 --lr 0.04

CUDA_VISIBLE_DEVICES=5 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v0-test --config_id v2

CUDA_VISIBLE_DEVICES=1 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v0.2 --config_id v2

CUDA_VISIBLE_DEVICES=1 python refinedet_mobile_val.py --dataset COCO --gpu_ids 0 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v0.2 --config_id v2 --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v0.2/refinedet_model-0.pth

CUDA_VISIBLE_DEVICES=0 python refinedet_mobile_val.py --dataset COCO --gpu_ids 0 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v0.2 --config_id v2 --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v0.2/refinedet_model-0.pth

train v1:
INFO:root:[56/120] || total_loss: 2.4257(mean_arm_loc_loss: 1.7216 mean_arm_cls_loss: 2.0315 mean_obm_loc_loss: 0.9019 mean_obm_cls_loss: 1.3022) || Batch time: 0.0119 sec. || LR: 0.000828

INFO:root:[288/1350] || total_loss: 1.6948(mean_arm_loc_loss: 0.7423 mean_arm_cls_loss: 1.4481 mean_obm_loc_loss: 0.6348 mean_obm_cls_loss: 0.8612) || Batch time: 0.0060 sec. || LR: 0.000116

#### 2018.6.24

CUDA_VISIBLE_DEVICES=4,5 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 1 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v3 --config_id v3

config v4:

```
CUDA_VISIBLE_DEVICES=6,7 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 1 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v4 --config_id v4

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --config_id v3 --gpu_ids 0 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v3/ss_predict --shape 320 --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v3/refinedet_model-190.pth --eval_img ./samples/demo/1045023827_4ec3e8ba5c_z.jpg

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --config_id v3 --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v3/ss_predict --shape 320 --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v3/refinedet_model-190.pth --eval_img ./samples/demo/1045023827_4ec3e8ba5c_z.jpg

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --config_id v4 --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v4/ss_predict --shape 320 --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v4/refinedet_model-190.pth --eval_img ./samples/demo/1045023827_4ec3e8ba5c_z.jpg

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --config_id v4 --gpu_ids 0 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v4/ss_predict --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v4/refinedet_model-190.pth --eval_img ./samples/demo/1045023827_4ec3e8ba5c_z.jpg

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --config_id v4 --gpu_ids 0 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v4/ss_predict --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v4/Final-refineDet-199.pth --eval_img ./samples/demo/1045023827_4ec3e8ba5c_z.jpg
```


#### 2018.6.25

config v5:

```
CUDA_VISIBLE_DEVICES=4,5 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 1 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v5 --config_id v5

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --config_id v5 --gpu_ids 0 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v5/ss_predict --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v5/Final-refineDet-200.pth --eval_img ./samples/demo/1045023827_4ec3e8ba5c_z.jpg

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v5/ss_predict --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v5/Final-refineDet-200.pth --eval_img ./samples/demo/1045023827_4ec3e8ba5c_z.jpg --config_id v5 --gpu_ids 0 --cuda

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v5/ss_predict --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v5/Final-refineDet-200.pth --eval_img ./samples/ebike-three.jpg --config_id v5 --gpu_ids 0 --cuda

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --workspace ./workspace/train-v5 --ckpt_path ./ckpt/Final-refineDet-200.pth --eval_img ./samples/ebike-three.jpg --config_id v5
```

#### 2018.6.27

```
CUDA_VISIBLE_DEVICES=2,3 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 1 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v6 --config_id v6
```

#### 2018.6.28

```
CUDA_VISIBLE_DEVICES=2,3 python refinedet_mobile_train.py --dataset COCO --gpu_ids 0 1 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v-r --config_id v-r
```

#### 2018.6.29

```
CUDA_VISIBLE_DEVICES=2 python refinedet_mobile_eval.py --workspace ./workspace/train-v6 --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v6/refinedet_model-190.pth --eval_img ./samples/ebike-three.jpg --config_id v6
```

```
CUDA_VISIBLE_DEVICES=1 python refinedet_mobile_val.py --dataset COCO --gpu_ids 0 --cuda --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v6/refinedet_model-190.pth  --config_id v6
```

#### 2018.7.3


test on previous models:

```
CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --workspace ./workspace/train-v3 --ckpt_path ./ckpt/train-v3-Final-refineDet-199.pth --eval_img ./samples/ebike-three.jpg --config_id v3

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --workspace ./workspace/train-v4 --ckpt_path ./ckpt/train-v4-Final-refineDet-199.pth --eval_img ./samples/ebike-three.jpg --config_id v4

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --workspace ./workspace/train-v5 --ckpt_path ./ckpt/train-v5-Final-refineDet-200.pth --eval_img ./samples/ebike-three.jpg --config_id v5

CUDA_VISIBLE_DEVICES=7 python refinedet_mobile_eval.py --workspace ./workspace/train-v6 --ckpt_path ckpt/train-v6-refinedet_model-190.pth --eval_img ./samples/ebike-three.jpg --config_id v6
```

CUDA_VISIBLE_DEVICES=4,5 python train.py --dataset COCO --gpu_ids 0 1 --cuda --workspace /mnt/ckpt/pytorchSSD/RefineDet_Renset/v1 --config_id r1 --resume 70

CUDA_VISIBLE_DEVICES=4,5 python train.py --dataset COCO --gpu_ids 0 1 --cuda --workspace /mnt/ckpt/pytorchSSD/RefineDet_Renset/v1 --config_id r1 --resume 70


### 2018.7.5

config v3:
```
CUDA_VISIBLE_DEVICES=0 python val.py --dataset COCO --gpu_ids 0 --cuda --workspace ./workspace/train-v3 --config_id v3 --ckpt_path ckpt/train-v3-Final-refineDet-199.pth

INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.268
INFO:root: Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.521
INFO:root: Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.245
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.069
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.312
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.529
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.136
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.317
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.366
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.135
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.433
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.639
```

config v4:

```
CUDA_VISIBLE_DEVICES=0 python val.py --dataset COCO --gpu_ids 0 --cuda --workspace ./workspace/train-v4 --config_id v4 --ckpt_path ckpt/train-v4-Final-refineDet-199.pth

INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.249
INFO:root: Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.479
INFO:root: Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.232
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.037
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.281
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.543
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.134
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.297
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.324
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.095
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.364
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.628
```

config v5:

```
CUDA_VISIBLE_DEVICES=0 python val.py --dataset COCO --gpu_ids 0 --cuda --workspace ./workspace/train-v5 --config_id v5 --ckpt_path ckpt/train-v5-Final-refineDet-200.pth

INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.223
INFO:root: Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.380
INFO:root: Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.230
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.007
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.237
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.558
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.132
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.248
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.249
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.001
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.259
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.624
```

config v6:

```
CUDA_VISIBLE_DEVICES=5 python val.py --dataset COCO --gpu_ids 0 --cuda --workspace ./workspace/train-v6 --config_id v6 --ckpt_path ./ckpt/train-v6-refinedet_model-190.pth

CUDA_VISIBLE_DEVICES=1 python refinedet_mobile_val.py --dataset COCO --gpu_ids 0 --cuda --ckpt_path ./ckpt/train-v6-refinedet_model-190.pth  --config_id v6
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.209
INFO:root: Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.368
INFO:root: Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.211
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.045
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.217
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.375
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.205
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.299
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.309
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.071
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.333
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.531

```

train with repulsion loss:

```
CUDA_VISIBLE_DEVICES=0 python train.py --dataset COCO --gpu_ids 0 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v5-t --config_id v5
```

### 2018.7.5

CUDA_VISIBLE_DEVICES=0 python demo.py --workspace ./workspace/train-r1 --ckpt_path ckpt/train-r1-Final-refineDet-100.pth --eval_img ./samples/ebike-three.jpg --config_id r1

CUDA_VISIBLE_DEVICES=6 python demo.py --workspace ./workspace/train-v4 --ckpt_path /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v4-new-2/Final-refineDet-100.pth --eval_img ./samples/ebike-three.jpg --config_id v4

### 2018.7.9

test r1:
```
CUDA_VISIBLE_DEVICES=6 python val.py --dataset COCO --gpu_ids 0 --cuda --workspace ./workspace/val-r1 --config_id r1 --ckpt_path ./ckpt/train-r1-Final-refineDet-100.pth

INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.341
INFO:root: Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.586
INFO:root: Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.344
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.091
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.406
INFO:root: Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.638
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.162
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.374
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.391
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.128
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.461
INFO:root: Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.711
```

train repulsion loss:
```
CUDA_VISIBLE_DEVICES=4 python train.py --dataset COCO --gpu_ids 0 --cuda --workspace /mnt/ckpt/pytorchSSD/Refine_mobilenet/train-v3-repulsion --config_id v3
```



### 2018.7.10

train for car:
```
pip3 install -i https://mirrors.aliyun.com/pypi/simple/ visdom graphviz easydict

CUDA_VISIBLE_DEVICES=0,1 python train.py --workspace ckpt/car-v1 --config_id c1 --cuda --gpu_ids 0 1

CUDA_VISIBLE_DEVICES=1 python train.py --workspace ckpt/car-v5 --config_id c1 --cuda --gpu_ids 0

CUDA_VISIBLE_DEVICES=1 python train.py --workspace ckpt/car-v5 --config_id c1 --cuda --gpu_ids 0

CUDA_VISIBLE_DEVICES=1 python val.py --workspace ckpt/car-v1 --config_id c1 --cuda --gpu_ids 0 --ckpt_path ckpt/car-v1/refinedet_model-200.pth
```

CUDA_VISIBLE_DEVICES=0 python demo.py --workspace ./workspace/car-val-v1 --ckpt_path ckpt/car-v1/Final-refineDet-200.pth --config_id r1 --gpu_ids 0 --cuda --eval_img samples/car_test/e8365e4f-a742-4952-b522-29e6bfcde62a.jpg

### training on coco dataset（80 classes）

CUDA_VISIBLE_DEVICES=1 python train.py --cuda --gpu_ids 0 --workspace ckpt/coco-v1 --config_id coco-v1

CUDA_VISIBLE_DEVICES=0 python train.py --cuda --gpu_ids 0 --workspace ckpt/coco-v2 --config_id coco-v2