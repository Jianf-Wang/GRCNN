# GRCNN

A Pytroch implementation of T-PAMI 2021 paper "Convolutional Neural Networks with Gated Recurrent Connections",  which is an extended journal version of the previous work "Gated Recurrent Convolution Neural Network for
OCR" (https://github.com/Jianf-Wang/GRCNN-for-OCR) presented in NeurIPS 2017. Extensive experiments are presented in this journal version. 

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

For other network architectures, please set "-a".

If you want to use the weight sharing setting, you can set "--weight-sharing" to "True".

To train on the cifar-100, you can add "--dataset cifar100" to above commands.

Training on ImageNet
-----------------
To train GRCNN or SK-GRCNN on ImageNet, please run with the following command:

  **GRCNN-55:** <br />
  
```
python imagenet_train.py \
  --epochs 100 \
  --dist-url 'tcp://localhost:10010' --multiprocessing-distributed --world-size 1 --rank 0 \
  --arch grcnn55 \
```
 
 **SK-GRCNN-55:** <br />

```
python imagenet_train.py \
  --epochs 120 \
  --dist-url 'tcp://localhost:10010' --multiprocessing-distributed --world-size 1 --rank 0 \
  --arch skgrcnn55 \
  --cos \
```
As for GRCNN-109 and SK-GRCNN-109, please set "--arch".

Pretrained Model
-----------------
The pretrained model are released:

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">name</th>
<th valign="bottom">top-1 acc.</th>
<th valign="bottom">model (Google Drive)</th>
<!-- TABLE BODY -->
<tr>
<td align="center">GRCNN-55</td>
<td align="center">77.02</td>
<td align="center"><a href="https://drive.google.com/file/d/1xm5ILJbuyp6p5XsUvrH2alvBZHnWo6EE/view?usp=sharing">download</a></td>
</tr>
<tr>
<td align="center">SK-GRCNN-55</td>
<td align="center">79.38</td>
<td align="center"><a href="https://drive.google.com/file/d/1AlDQCo1hLe5P7OiDI3Pvlq_qHVdvGGxx/view?usp=sharing">download</a></td>
</tr>
</tr>
<tr>
<td align="center">GRCNN-109</td>
<td align="center">Comming Soon...</td>
<td align="center">Comming Soon...</td>
</tr>
</tr>
<tr>
<td align="center">SK-GRCNN-109</td>
<td align="center">Comming Soon...</td>
<td align="center">Comming Soon...</td>
</tr>
</tbody></table>

To simply use the pretrained models, run following commands:

```
import torch
import models.imagenet.GRCNN as grcnn
model = grcnn.grcnn_55()
model.load_state_dict(torch.load('checkpoint_params_grcnn55.pt'))

```


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
