
### setup by docker image
docker run --network host --ipc host -v /home/fulingzhi/workspace/PytorchSSD:/app -v /mnt/gf_mnt/datasets/cocoapi:/mnt/dataset/coco -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22 bash

### training

pip install git+https://github.com/szagoruyko/pyinn.git@master

python train_test.py -d COCO -v RFB_mobile -s 300 --ngpu 1 --basenet weights/pretrained/mobilenet_feature.pth

python train_test.py -d COCO -v RFB_vgg -s 300 --ngpu 1 --basenet weights/pretrained/vgg16_reducedfc.pth



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
