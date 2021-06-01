import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functools import reduce
import math
from ..builder import BACKBONES
from mmcv.cnn import build_conv_layer, build_norm_layer, constant_init, kaiming_init
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import BaseModule

class SKConv(BaseModule):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32, groups=32, init_cfg=None, norm_cfg=None):

        super(SKConv,self).__init__(init_cfg)
        d=max(in_channels//r,L)   
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList() 
        for i in range(M):
            conv1 = build_conv_layer(None, in_channels, out_channels, 3, stride=stride, padding=1+i, dilation=1+i, bias=False, groups=groups)
            self.conv.append(nn.Sequential(conv1,
                                           build_norm_layer(norm_cfg, out_channels)[1],
                                           nn.ReLU(inplace=True)))
        self.global_pool=nn.AdaptiveAvgPool2d(1) 
        conv_fc = build_conv_layer(None, out_channels, d, 1, stride=1, padding=0, bias=False)
        self.fc1=nn.Sequential(conv_fc,
                               build_norm_layer(norm_cfg, d)[1],
                               nn.ReLU(inplace=True))   
        self.fc2 = build_conv_layer(None, d, out_channels*M, 1, stride=1, padding=0, bias=False) 
        self.softmax=nn.Softmax(dim=1)

    def forward(self, input):
        batch_size=input.size(0)
        output=[]
        for i,conv in enumerate(self.conv):
            output.append(conv(input))
        U=reduce(lambda x,y:x+y,output) 
        s=self.global_pool(U)
        z=self.fc1(s)  
        a_b=self.fc2(z) 
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,-1)
        a_b=self.softmax(a_b) 
        a_b=list(a_b.chunk(self.M,dim=1))
        a_b=list(map(lambda x:x.reshape(batch_size,self.out_channels,1,1),a_b)) 
        V=list(map(lambda x,y:x*y,output,a_b)) 
        V=reduce(lambda x,y:x+y,V) 
        return V


class GRCL(BaseModule):
  def __init__(self, inplanes, planes, downsample=True, iter = 3, SKconv=True, expansion=2, weight_sharing=False, norm_cfg=dict(type='BN', requires_grad=True), init_cfg=None):
    super(GRCL, self).__init__(init_cfg)

    self.iter = iter
    self.expansion = expansion
    # feed-forward part

    self.add_module('bn_f', build_norm_layer(norm_cfg, inplanes)[1])
    self.add_module('relu_f', nn.ReLU(inplace=True))
    conv_f = build_conv_layer(None, inplanes, int(planes*self.expansion), 3, stride=1, padding=1, bias=False, groups=32)
    self.add_module('conv_f', conv_f)
    
    self.add_module('bn_g_f', build_norm_layer(norm_cfg, inplanes)[1])
    self.add_module('relu_g_f', nn.ReLU(inplace=True))
    conv_g_f = build_conv_layer(None, inplanes, int(planes*self.expansion), 1, stride=1, padding=0, bias=True, groups=32)

    self.add_module('conv_g_f', conv_g_f)
    conv_g_r = build_conv_layer(None, int(planes* self.expansion), int(planes*self.expansion), 1, stride=1, padding=0, bias=False, groups=32)
    self.add_module('conv_g_r', conv_g_r)
    self.add_module('sig', nn.Sigmoid())

    # recurrent part
    if not weight_sharing:
     for i in range(0, self.iter):
      layers = []
      layers_g_bn = []
    
      layers.append(build_norm_layer(norm_cfg, planes*self.expansion)[1])
      layers.append(nn.ReLU(inplace=True))
      conv_1 = build_conv_layer(None, int(planes*self.expansion), planes, 1, stride=1, padding=0, bias=False)
      layers.append(conv_1)

      layers.append(build_norm_layer(norm_cfg, planes)[1])
      layers.append(nn.ReLU(inplace=True))

      if SKconv:
       layers.append(SKConv(planes, planes, norm_cfg = norm_cfg ))
      else:
       layers.append(build_conv_layer(None, planes, planes, 3, stride=1, padding=1, bias=False))
       layers.append(build_norm_layer(norm_cfg, planes)[1])
       layers.append(nn.ReLU(inplace=True))
   
      conv_2 = build_conv_layer(None, planes, int(planes*self.expansion), 1, stride=1, padding=0, bias=False)
      layers.append(conv_2)
      layers_g_bn.append(build_norm_layer(norm_cfg, int(planes*self.expansion))[1])

      layers_g_bn.append(nn.ReLU(inplace=True)) 

      self.add_module('iter_'+str(i+1), nn.Sequential(*layers))
      self.add_module('iter_g_'+str(i+1), nn.Sequential(*layers_g_bn))
    else:
      conv_1 = build_conv_layer(None, int(planes*self.expansion), planes, 1, stride=1, padding=0, bias=False)
      if SKconv:
       sk_conv = SKConv(planes, planes, groups=16, r=32)
      else:
       conv_ = build_conv_layer(None, planes, planes, 3, stride=1, padding=1, bias=False)
      conv_2 = build_conv_layer(None, planes, int(planes*self.expansion), 1, stride=1, padding=0, bias=False)

      for i in range(0, self.iter):
       layers = []
       layers_g_bn = []

       layers.append(build_norm_layer(norm_cfg, int(planes*self.expansion))[1])
       layers.append(nn.ReLU(inplace=True))
       layers.append(conv_1)

       layers.append(build_norm_layer(norm_cfg, planes)[1])
       layers.append(nn.ReLU(inplace=True))

       if SKconv:
        layers.append(sk_conv)  
       else:
        layers.append(conv_)

        layers.append(build_norm_layer(norm_cfg, planes)[1])
        layers.append(nn.ReLU(inplace=True))

       layers.append(conv_2)

       self.add_module('iter_'+str(i+1), nn.Sequential(*layers))
       layers_g_bn.append(build_norm_layer(norm_cfg, int(planes*self.expansion))[1])
  
       layers_g_bn.append(nn.ReLU(inplace=True)) 
       self.add_module('iter_g_'+str(i+1), nn.Sequential(*layers_g_bn))

    self.downsample = downsample
    if self.downsample:
       self.add_module('d_bn', build_norm_layer(norm_cfg, int(planes*self.expansion))[1])
       self.add_module('d_relu', nn.ReLU(inplace=True))
       d_conv = build_conv_layer(None, int(planes * self.expansion), int(planes*self.expansion), 1, stride=1, padding=0, bias=False)
       self.add_module('d_conv', d_conv)
       self.add_module('d_ave', nn.AvgPool2d((2, 2), stride=2))
  
       self.add_module('d_bn_1', build_norm_layer(norm_cfg, int(planes*self.expansion))[1])
       self.add_module('d_relu_1', nn.ReLU(inplace=True))
       d_conv_1 = build_conv_layer(None, int(planes*self.expansion), planes, 1, stride=1, padding=0, bias=False)
       self.add_module('d_conv_1', d_conv_1)

       self.add_module('d_bn_3', build_norm_layer(norm_cfg, planes)[1])
       self.add_module('d_relu_3', nn.ReLU(inplace=True))
       
       if SKconv:
         d_conv_3 = SKConv(planes, planes, stride=2, norm_cfg=norm_cfg)
         self.add_module('d_conv_3', d_conv_3)
       else:
         d_conv_3 = build_conv_layer(None, planes, planes, 3, stride=2, padding=1, bias=False)
         self.add_module('d_conv_3', d_conv_3)

       d_conv_1e = build_conv_layer(None, planes, int(planes*self.expansion), 1, stride=1, padding=0, bias=False)
       self.add_module('d_conv_1e', d_conv_1e)

  def forward(self, x):

    # feed-forward
    x_bn = self.bn_f(x)
    x_act = self.relu_f(x_bn)
    x_s = self.conv_f(x_act)
    
    x_g_bn = self.bn_g_f(x)
    x_g_act = self.relu_g_f(x_g_bn)
    x_g_s = self.conv_g_f(x_g_act)

    # recurrent 
    for i in range(0, self.iter):
       x_g_r = self.conv_g_r(self.__dict__['_modules']["iter_g_%s" % str(i+1)](x_s))
       x_s = self.__dict__['_modules']["iter_%s" % str(i+1)](x_s) * torch.sigmoid(x_g_r + x_g_s) + x_s
    
    if self.downsample:
      x_s_1 = self.d_conv(self.d_ave(self.d_relu(self.d_bn(x_s))))
      x_s_2 = self.d_conv_1e(self.d_conv_3(self.d_relu_3(self.d_bn_3(self.d_conv_1(self.d_relu_1(self.d_bn_1(x_s)))))))
      x_s_d = x_s_1 + x_s_2
      return x_s, x_s_d
    else:
      return x_s



@BACKBONES.register_module()
class GRCNN(BaseModule):
 
  def __init__(self, frozen_stages=-1, norm_cfg=dict(type='BN', requires_grad=True), norm_eval='True', name='GRCNN55', pretrained = None, init_cfg=None):
    """ Args:
      iters:iterations.
      num_classes: number of classes
    """
    super(GRCNN, self).__init__(init_cfg)

    assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be setting at the same time'
    block_init_cfg = None
    if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is a deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)

    weight_share = False
    if '55' in name:
       self.iters = [3, 3, 4, 3]
       if 'SK' in name:
          SKconv = True
          self.maps = [128, 256, 512, 1024]
          self.expansion = 2
       elif 'SHARE' in name:
          SKconv=False
          weight_share=True
          self.maps = [64, 128, 256, 512]
          self.expansion = 4
       else:
          SKconv = False
          self.maps = [64, 128, 256, 512]
          self.expansion = 4
    elif '109' in name:
       self.iters = [3, 3, 22, 3]
       if 'SK' in name:
          SKconv = True
          self.maps = [128, 256, 512, 1024]
          self.expansion = 2
       elif 'SHARE' in name:
          SKconv=False
          weight_share=True
          self.maps = [64, 128, 256, 512]
          self.expansion = 4
       else:
          SKconv = False
          self.maps = [64, 128, 256, 512]
          self.expansion = 4
    else:
       raise TypeError('Please select the backbone from [GRCNN55, GRCNN55_SHARE, SKGRCNN55, GRCNN109, GRCNN109_SHARE, SKGRCNN109]')

    self.norm_eval=norm_eval
    self.frozen_stages=frozen_stages

    conv1 = build_conv_layer(None, 3,  64, 7, stride=2, padding=3, bias=False)
    self.add_module('conv1', conv1)
    self.add_module('bn1', build_norm_layer(norm_cfg, 64)[1])
    self.add_module('relu', nn.ReLU(inplace=True))
    self.add_module('maxpool', nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
    
    conv2 = build_conv_layer(None, 64, 64, 3, stride=1, padding=1, bias=False)
    self.add_module('conv2', conv2)


    self.add_module('layer1', GRCL(64, self.maps[0], True, self.iters[0], SKconv, self.expansion, weight_share, norm_cfg))
    self.add_module('layer2', GRCL(self.maps[0] * self.expansion, self.maps[1], True, self.iters[1], SKconv, self.expansion, weight_share, norm_cfg))
    self.add_module('layer3', GRCL(self.maps[1] * self.expansion, self.maps[2], True, self.iters[2], SKconv, self.expansion, weight_share, norm_cfg))
    self.add_module('layer4', GRCL(self.maps[2] * self.expansion, self.maps[3], False, self.iters[3], SKconv, self.expansion, weight_share,  norm_cfg))

    self._freeze_stages()
  
  def _freeze_stages(self):     
    if self.frozen_stages >=0:
       self.conv1.eval()
       for param in self.conv1.parameters():
           param.requires_grad = False
       self.bn1.eval()
       for param in self.bn1.parameters():
           param.requires_grad = False
       self.conv2.eval()
       for param in self.conv2.parameters():
           param.requires_grad = False

       for i in range(1, self.frozen_stages + 1):
            m = getattr(self, f'layer{i}')
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
  
  def features(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.maxpool(x)
    x = self.conv2(x)
    
    out_feat = []

    x_out, x = self.layer1(x)
    out_feat.append(x_out)

    x_out, x = self.layer2(x)
    out_feat.append(x_out)

    x_out, x = self.layer3(x)
    out_feat.append(x_out)

    x = self.layer4(x)
    out_feat.append(x)
 
    return tuple(out_feat)

  def forward(self, x):
        x = self.features(x)
        return x

  def train(self, mode=True):
        """Convert the model into training mode whill keeping the normalization
        layer freezed."""
        super(GRCNN, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                # trick: eval have effect on BatchNorm only
                if isinstance(m, _BatchNorm):
                    m.eval()
  
