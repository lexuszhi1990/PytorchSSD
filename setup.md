### setup by docker image

on 177 server:
docker run --network host --ipc host -v /home/fulingzhi/workspace/PytorchSSD:/app -v /mnt/gf_mnt/datasets/cocoapi:/mnt/dataset/coco -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22-dev bash

on 172 server:
docker run --network host --ipc host -v /home/david/fashionAI/PytorchSSD:/app -v /data/david/cocoapi:/mnt/dataset/coco -v /data/david/models/pytorchSSD:/mnt/ckpt/pytorchSSD -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22-dev bash

local dev:
docker run --name py-ssd --network host --ipc host -v /Users/david/repo/detection/PytorchSSD:/app -v /Users/david/mnt/data/VOCdevkit:/mnt/dataset/voc2012 -it --rm floydhub/pytorch:0.3.1-py3.26 bash

### daily log

#### 2018.5.16

train RFB_SSD:
python train_test.py -d COCO -v RFB_vgg -s 300 --batch_size 14 --visdom True --send_images_to_visdom False --basenet weights/pretrained/vgg16_reducedfc.pth --gpu_ids 0 --save_folder /mnt/ckpt/pytorchSSD --date RFB_vgg_0516

train refinedet_SSD:
CUDA_VISIBLE_DEVICES=1 python refinedet_train_test.py -d COCO -v Refine_vgg -s 320 --batch_size 14 --visdom True --send_images_to_visdom False --basenet weights/pretrained/vgg16_reducedfc.pth --gpu_ids 0 --save_folder /mnt/ckpt/pytorchSSD --date refinedet_vgg_0516

CUDA_VISIBLE_DEVICES=1,2 python refinedet_train_test.py -d COCO -v Refine_vgg -s 320 --visdom True --send_images_to_visdom False --basenet weights/pretrained/vgg16_reducedfc.pth --save_folder /mnt/ckpt/pytorchSSD --date refinedet_vgg_0516 --gpu_ids 0,1 --batch_size 56

#### 2018.5.30

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
