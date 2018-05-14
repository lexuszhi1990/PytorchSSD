
### setup by docker image
docker run --network host -v /home/fulingzhi/workspace/PytorchSSD:/app -v /mnt/gf_mnt/datasets/cocoapi:/mnt/dataset/coco -it --rm floydhub/pytorch:0.3.0-gpu.cuda9cudnn7-py3.22 bash

### training

pip install git+https://github.com/szagoruyko/pyinn.git@master

python train_test.py -d COCO -v RFB_mobile -s 300 --ngpu 1 --basenet weights/pretrained/mobilenet_feature.pth

python train_test.py -d COCO -v RFB_vgg -s 300 --ngpu 1 --basenet vgg16_reducedfc.pth

