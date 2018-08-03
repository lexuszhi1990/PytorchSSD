# one-stage detection series(ssd and its variants)

### Contents
1. [Results](#results)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Todos](#todos)
5. [References](#references)

### Results

coco-person:

framework|model|precision|recall|speed
---------|----|---------|-------|------
ssd|resneXt50|0.341|0.711|10fps

### Installation

install packages with pip:

`pip install -i https://mirrors.aliyun.com/pypi/simple/ torch visdom graphviz easydict`

run with docker:

```shell
docker run --ipc host -v /your-dir:/app  -it --rm floydhub/pytorch:0.4.0-py3.31 bash
```

### Usage

#### train

```
CUDA_VISIBLE_DEVICES=0 python train.py --cuda --gpu_ids 0 --config coco_v1
CUDA_VISIBLE_DEVICES=1 python val.py --cuda --gpu_ids 0 --ckpt_path workspace/coco-v1/coco-resnet-v1-0.pth --config coco_v1
```

#### val

```
CUDA_VISIBLE_DEVICES=1 python val.py --cuda --gpu_ids 0 --config coco_v2 --ckpt_path workspace/coco-mobilenet-v1/coco-resnet-10.pth
```

#### inference

```
python eval.py --config coco_v1 --ckpt_path workspace/coco-mobilenet-v1/coco-resnet-10.pth --img_path samples/demo/ebike-three.jpg

python eval.py --config coco_v1 --img_path samples/demo/ebike-three.jpg --ckpt_path workspace/coco-v1/coco-resnet-v1-0.pth

python eval.py --config coco_v2 --ckpt_path workspace/coco-mobilenet-v1/coco-resnet-10.pth --img_path samples/demo/ebike-three.jpg
```

### Todos
- [x] upgrade framework to pytorch 0.4
- [ ] support atribary shape as inputs
- [ ] implement refinedDet

### References

- SSD [SSD: Single Shot Multibox  Detector](https://arxiv.org/abs/1512.02325)
- RefineDet[Single-Shot Refinement Neural Network for Object Detection](https://arxiv.org/pdf/1711.06897.pdf)
- FSSD [FSSD: Feature Fusion Single Shot Multibox Detector](https://arxiv.org/abs/1712.00960)
- RFB-SSD[Receptive Field Block Net for Accurate and Fast Object Detection](https://arxiv.org/abs/1711.07767)
