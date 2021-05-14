import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from functools import reduce
import math

__all__ = [
    'grcnn56', 'grcnn110_sk', 'grcnn110',
]



class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1,M=2,r=16,L=32, groups=32):

        super(SKConv,self).__init__()
        d=max(in_channels//r,L)   
        self.M=M
        self.out_channels=out_channels
        self.conv=nn.ModuleList()  

        for i in range(M):
            if i == 0:
               conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                          groups=groups, bias=False)
            else:
               conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0,
                         groups=groups, bias=False)

            init.kaiming_normal_(conv1.weight)

            self.conv.append(nn.Sequential(conv1, 
                                           nn.BatchNorm2d(out_channels, momentum=0.05),
                                           nn.ReLU(inplace=True)))

        self.global_pool=nn.AdaptiveAvgPool2d(1)
        conv_fc = nn.Conv2d(out_channels,d,1,bias=False)
        init.normal_(conv_fc.weight, std=0.01)

        self.fc1=nn.Sequential(conv_fc,
                               nn.BatchNorm2d(d, momentum=0.05),
                               nn.ReLU(inplace=True))

        self.fc2=nn.Conv2d(d,out_channels*M,1,1,bias=False)   
        init.normal_(self.fc2.weight, std=0.01)
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



class GRCL(nn.Module):
  def __init__(self, inplanes, planes, downsample=True, iter = 3,  SKconv=False, expansion=4, weight_sharing=False):
    super(GRCL, self).__init__()

    self.iter = iter
    self.expansion = expansion
   
    # feed-forward part
    self.add_module('bn_f', nn.BatchNorm2d(inplanes, momentum=0.05))
    self.add_module('relu_f', nn.ReLU(inplace=True))
    conv_f = nn.Conv2d(inplanes, int(planes* self.expansion), kernel_size=3, stride=1, padding=1, bias=False, groups=16)
    init.kaiming_normal_(conv_f.weight)
    self.add_module('conv_f', conv_f)
    
    self.add_module('bn_g_f', nn.BatchNorm2d(inplanes, momentum=0.05))
    self.add_module('relu_g_f', nn.ReLU(inplace=True))
    conv_g_f = nn.Conv2d(inplanes, int(planes* self.expansion), kernel_size=1, stride=1, padding=0, bias=True, groups=16)
    init.normal_(conv_g_f.weight, std=0.01)
    self.add_module('conv_g_f', conv_g_f)

    conv_g_r = nn.Conv2d(int(planes* self.expansion), int(planes* self.expansion), kernel_size=1, stride=1, padding=0, groups=16, bias=False)
    init.normal_(conv_g_r.weight, std=0.01)
    self.add_module('conv_g_r', conv_g_r)


    # recurrent part

    if not weight_sharing:
     for i in range(0, self.iter):
      layers = []
      layers_g_bn = []
    
      layers.append(nn.BatchNorm2d(int(planes*self.expansion), momentum=0.05))
      layers.append(nn.ReLU(inplace=True))
      conv_1 = nn.Conv2d(int(planes*self.expansion), planes, kernel_size=1, stride=1, padding=0, bias=False)
      init.kaiming_normal_(conv_1.weight)
      layers.append(conv_1)

      layers.append(nn.BatchNorm2d(planes, momentum=0.05))
      layers.append(nn.ReLU(inplace=True))

      if SKconv:
       layers.append(SKConv(planes, planes, groups=16, r=32))  
      else:
       layers.append(nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False))
       layers.append(nn.BatchNorm2d(planes, momentum=0.05))
       layers.append(nn.ReLU(inplace=True))

      conv_2 = nn.Conv2d(planes, int(planes*self.expansion), kernel_size=1, stride=1, padding=0, bias=False)   
      init.kaiming_normal_(conv_2.weight)
      layers.append(conv_2)

      self.add_module('iter_'+str(i+1), nn.Sequential(*layers))
      layers_g_bn.append(nn.BatchNorm2d(int(planes*self.expansion), momentum=0.05))
      layers_g_bn.append(nn.ReLU(inplace=True)) 

      self.add_module('iter_g_'+str(i+1), nn.Sequential(*layers_g_bn))
    else:
     conv_1 = nn.Conv2d(int(planes*self.expansion), planes, kernel_size=1, stride=1, padding=0, bias=False)
     init.kaiming_normal_(conv_1.weight)
     if SKconv:
       sk_conv = SKConv(planes, planes, groups=16, r=32)
     else:
       conv_ = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
       init.kaiming_normal_(conv_.weight)

     conv_2 = nn.Conv2d(planes, int(planes*self.expansion), kernel_size=1, stride=1, padding=0, bias=False)
     init.kaiming_normal_(conv_2.weight)

     for i in range(0, self.iter):
      layers = []
      layers_g_bn = []
    
      layers.append(nn.BatchNorm2d(int(planes*self.expansion), momentum=0.05))
      layers.append(nn.ReLU(inplace=True))
      layers.append(conv_1)

      layers.append(nn.BatchNorm2d(planes, momentum=0.05))
      layers.append(nn.ReLU(inplace=True))

      if SKconv:
       layers.append(sk_conv)  
      else:
       layers.append(conv_)
       layers.append(nn.BatchNorm2d(planes, momentum=0.05))
       layers.append(nn.ReLU(inplace=True))

      layers.append(conv_2)

      self.add_module('iter_'+str(i+1), nn.Sequential(*layers))
      layers_g_bn.append(nn.BatchNorm2d(int(planes*self.expansion), momentum=0.05))
      layers_g_bn.append(nn.ReLU(inplace=True)) 

      self.add_module('iter_g_'+str(i+1), nn.Sequential(*layers_g_bn))  

    self.downsample = downsample
    if self.downsample:
       self.add_module('d_bn', nn.BatchNorm2d(int(planes * self.expansion), momentum=0.05))
       self.add_module('d_relu', nn.ReLU(inplace=True))
       d_conv = nn.Conv2d(int(planes* self.expansion), int(planes* self.expansion), kernel_size=1, stride=1, padding=0, bias=False)
       init.kaiming_normal_(d_conv.weight)
       self.add_module('d_conv', d_conv)
       self.add_module('d_ave', nn.AvgPool2d((2, 2)))


       self.add_module('d_bn_1', nn.BatchNorm2d(int(planes * self.expansion), momentum=0.05))
       self.add_module('d_relu_1', nn.ReLU(inplace=True))
       d_conv_1 = nn.Conv2d(int(planes* self.expansion), planes, kernel_size=1, stride=1, padding=0,
       bias=False)
       init.kaiming_normal_(d_conv_1.weight)
       self.add_module('d_conv_1', d_conv_1)

       self.add_module('d_bn_3', nn.BatchNorm2d(planes, momentum=0.05))
       self.add_module('d_relu_3', nn.ReLU(inplace=True))

       if SKconv:
         d_conv_3 = SKConv(planes, planes, stride=2, r=32, groups=16)
         self.add_module('d_conv_3', d_conv_3)
       else:
         d_conv_3 = nn.Conv2d(planes, planes, kernel_size=3, stride=2, padding=1, bias=False)
         init.kaiming_normal_(d_conv_3.weight)
         self.add_module('d_conv_3', d_conv_3)
         self.add_module('d_bn_1e', nn.BatchNorm2d(planes, momentum=0.05))
         self.add_module('d_relu_1e', nn.ReLU(inplace=True))

       d_conv_1e = nn.Conv2d(planes, int(planes * self.expansion), kernel_size=1, stride=1, padding=0, bias=False)
       init.kaiming_normal_(d_conv_1e.weight)
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
         x_s = x_s_1 + x_s_2

    return x_s

class CifarGRCNN(nn.Module):
 
  def __init__(self, iters, maps, num_classes, SKconv=False, expansion=4, weight_sharing=False):
   
    super(CifarGRCNN, self).__init__()
    self.iters = iters
    self.maps = maps
    self.num_classes = num_classes

    self.conv_3x3 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

    init.kaiming_normal_(self.conv_3x3.weight)
    
    self.expansion = expansion
    self.SK = SKconv
 
    self.stage_1 = GRCL(64, self.maps[0], True, self.iters[0], self.SK, self.expansion, weight_sharing)
    self.stage_2 = GRCL(int(self.maps[0] * self.expansion), self.maps[1], True, self.iters[1],  self.SK, self.expansion, weight_sharing)
    self.stage_3 = GRCL(int(self.maps[1] * self.expansion), self.maps[2], False, self.iters[2], self.SK, self.expansion, weight_sharing)
    self.lastact = nn.Sequential(nn.BatchNorm2d(int(self.maps[2] * self.expansion), momentum=0.05), nn.ReLU(inplace=True))
    self.avgpool = nn.AvgPool2d(8)
    self.classifier = nn.Linear(int(self.maps[2] * self.expansion), num_classes)

    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        if m.bias is not None:
           m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()


  def forward(self, x):
    x = self.conv_3x3(x)
    x = self.stage_1(x)
    x = self.stage_2(x)
    x = self.stage_3(x)
    x = self.lastact(x)
    x = self.avgpool(x)
    x = x.view(x.size(0), -1)
    return self.classifier(x)


def grcnn110(num_classes=10, weight_sharing=False):
  """Args:
    num_classes (uint): number of classes
  """

  model = CifarGRCNN([6, 9, 18], [128, 160, 192], num_classes, expansion=4, weight_sharing=weight_sharing)
  return model


def grcnn110_sk(num_classes=10, weight_sharing=False):
  """Args:
    num_classes (uint): number of classes
  """

  model = CifarGRCNN([6, 9, 18], [192, 320, 448], num_classes, SKconv=True, expansion=2, weight_sharing=weight_sharing)

  return model

def grcnn56(num_classes=10, weight_sharing=False):
  """Args:
    num_classes (uint): number of classes
  """

  model = CifarGRCNN([3, 5, 7], [128, 160, 192], num_classes, expansion=4, weight_sharing=weight_sharing)
  return model


