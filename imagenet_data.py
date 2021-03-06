import os


from PIL import Image
import numpy as np
import torch
import torch.utils.data
import random
import pickle as pickle


class Imagenet_D(torch.utils.data.Dataset):

    def __init__(self, imagenet_root, img_list_pkl, transform, label_list, mode='train', cutout=True):

        self.img_list = img_list_pkl
        if mode == 'train':
          self.dir =  os.path.join(imagenet_root, 'train') 
        else:
          self.dir = os.path.join(imagenet_root, 'val')
        self.cutout = cutout
        self.transform = transform

        self.img_label_list = [] 
        self.label_list = label_list

        img_list_all  = pickle.load(open(self.img_list, 'rb'))
        
        i = 0
        for ele in img_list_all:
            label = ele[0].split('/')[-2]
            self.label_list[label] = i
            i+=1

        for ele in img_list_all:
            label = ele[0].split('/')[-2]
            for ele_1 in ele:
                name = ele_1.split('/')[-1]
                path_ = os.path.join(label, name)
                path_total = os.path.join(self.dir, path_)
                self.img_label_list.append([path_total, self.label_list[label]])


    def __len__(self):

        return len(self.img_label_list)


    def __getitem__(self, idx):
        img = self.img_label_list[idx][0]
        label = self.img_label_list[idx][1]

        image = Image.open(img).convert('RGB')
        
        
        image = self.transform(image)
       

        return image, label
