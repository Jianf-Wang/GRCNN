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

if you want to use the weight sharing setting, you can set "--weight-sharing" to "True".

To train on the cifar-100, you can add "--dataset cifar100" to above commands.


Training on ImageNet
-----------------
The code on imagenet will be released soon.
