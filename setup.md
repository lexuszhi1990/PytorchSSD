### start by docker image

on 177 server:
docker run --network host --ipc host -v /home/fulingzhi/workspace/PytorchSSD:/app -v /mnt/gf_mnt/datasets/cocoapi:/mnt/dataset/coco -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22-dev bash

on 172 server:
docker run --network host --ipc host -v /home/david/fashionAI/PytorchSSD:/app -v /data/david/cocoapi:/mnt/dataset/coco -v /data/david/open-image-v4:/mnt/dataset/open-image-v4 -v /data/david/models/pytorchSSD:/mnt/ckpt/pytorchSSD -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22-dev bash

local dev:
docker run --name py-ssd --network host --ipc host -v /Users/david/repo/detection/PytorchSSD:/app -v /Users/david/mnt/data/VOCdevkit:/mnt/dataset/VOCdevkit -it --rm floydhub/pytorch:0.4.0-py3.31 bash

on work-station:
docker run --name py-ssd --network host --ipc host -v /mnt/workspace/david/PytorchSSD:/app -v /mnt/datasets/pascal-voc/VOCdevkit:/mnt/dataset/VOCdevkit -it --rm floydhub/pytorch:0.3.1-py3.30 bash

docker run --name py-ssd --network host --ipc host -v /mnt/workspace/david/PytorchSSD:/app -v /mnt/datasets/station_car:/mnt/dataset/car -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22 bash

docker run --name py-ssd --network host --ipc host -v /mnt/workspace/david/PytorchSSD:/app -v /mnt/datasets/coco/cocoapi:/mnt/dataset/car -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22 bash


### setup project


### install dependencies

`pip3 install -i https://mirrors.aliyun.com/pypi/simple/ visdom graphviz easydict`

#### build extension

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


### daily log

### training on coco dataset（80 classes）

CUDA_VISIBLE_DEVICES=0 python train.py --cuda --gpu_ids 0 --config coco_v1


### training on voc dataset

`python train.py --config voc_v1`
