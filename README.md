# GRCNN
This is an implementation of the paper "Convolutional Neural Networks with Gated Recurrent Connections".

Build
-----

This GRCNN implementation is built upon the PyTorch. The requirements are:

1. PyTorch 1.7.0
2. CUDA 10.1

Training on Cifar
-----------------
To simply train on cifar-10, please run with the following command:
 
  **GRCNN-56:** <br />
  
    python train_cifar.py --gpu-id 0,1 -a grcnn56 
    
  **GRCNN-110:** <br />
  
    python train_cifar.py --gpu-id 0,1 -a grcnn110 
    
  **SK-GRCNN-110:** <br />
  
    python train_cifar.py --gpu-id 0,1 -a grcnn110_sk

If you want to use the weight sharing setting, you can set "--weight-sharing" to "True".

To train on the cifar-100, you can add "--dataset cifar100" to above commands.


Training on ImageNet
-----------------
To train GRCNN or SK-GRCNN on ImageNet, please run with the following command::

  **GRCNN-55:** <br />
  
```
python imagenet_train.py \
  --epochs 100 \
  --dist-url 'tcp://localhost:10010' --multiprocessing-distributed --world-size 1 --rank 0 \
  --cos False \
  --arch grcnn55 \
```

  **GRCNN-109:** <br />

```
python imagenet_train.py \
  --epochs 100 \
  --dist-url 'tcp://localhost:10010' --multiprocessing-distributed --world-size 1 --rank 0 \
  --cos False \
  --arch grcnn109 \
 ```
 
 **SK-GRCNN-55:** <br />

```
python imagenet_train.py \
  --epochs 120 \
  --dist-url 'tcp://localhost:10010' --multiprocessing-distributed --world-size 1 --rank 0 \
  --cos True \
  --arch skgrcnn55 \
```

  **GRCNN-109:** <br />

```
python imagenet_train.py \
  --epochs 120 \
  --dist-url 'tcp://localhost:10010' --multiprocessing-distributed --world-size 1 --rank 0 \
  --cos True \
  --arch skgrcnn109 \
```  

The pretrained will be released soon .

Citation
-----------------

```
@Article{jianfeng2021grcnn,
  author  = {Jianfeng Wang and Xiaolin Hu},
  title   = {Convolutional Neural Networks with Gated Recurrent Connections},
  journal = {TPAMI},
  year    = {2021},
}
```
