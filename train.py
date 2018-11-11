#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 12:27:04 2018

@author: wangxiaokai
"""
from __future__ import division
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import torch.nn.functional as F
import time
import argparse
import torchvision.datasets.folder as fold
import numpy as np
import os
import pandas as pd
from model import UNet 
from CrossEntropy2d import CrossEntropy2d
from PIL import Image
import matplotlib.pyplot as plt
# set parameters
#test_batch_size = 100
#total_train = 100
#total_test = 100


parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=20, help='number of epochs of training')
parser.add_argument('--num_classes', type = int, default = 2, help = 'number of classes')
parser.add_argument('--network_depth', type = int, default = 5, help = 'number of network depth')
parser.add_argument('--dataset_name', type=str, default="2d_music_data", help='name of the dataset')
parser.add_argument('--dataset_path', type = str, default = '/Users/shenguangyu1/Desktop/purdue/CS501/proSE/Music_Dataset/2d_music_data/',help = 'path of the dataset')
parser.add_argument('--train_batch_size', type=int, default=5, help='size of the batches')
parser.add_argument('--eta', type=float, default=0.001, help='adam: learning rate')
parser.add_argument('--img_height', type=int, default=100, help='size of image height')
parser.add_argument('--img_width', type=int, default=90, help='size of image width')
parser.add_argument('--input_channels', type=int, default=1, help='number of image channels')
parser.add_argument('--path_image_new', type = str, default = '/Users/shenguangyu1/Desktop/purdue/BME595/proj/data/image_new/', help = 'path of images')
parser.add_argument('--path_label_new', type = str, default = '/Users/shenguangyu1/Desktop/purdue/BME595/proj/data/label_new/', help = 'path of labels')
parser.add_argument('--val_batch',type = int, default = 10, help = 'num of batch for val')

opt = parser.parse_args()

path_image_new = '/Users/shenguangyu1/Desktop/purdue/BME595/proj/data/image_new/'
path_label_new = '/Users/shenguangyu1/Desktop/purdue/BME595/proj/data/label_new/'

# define the dataset
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, label_dir, train, transform_image, transform_label,loader=pd.read_csv):
        """
        Args:
            image_dir (string) : Path to the csv file storing images.
            label_dir (string) : Path to the csv file storing labels.
            transform_image : Transform used on the image.
            transform_label : Transform used on the label.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.transform_image = transform_image
        self.transform_label = transform_label
        self.loader = loader
        self.classes_image, self.classes_idx_image = fold.find_classes(self.image_dir)
        self.classes_label, self.classes_idx_label = fold.find_classes(self.label_dir)
        self.images = fold.make_dataset(self.image_dir, self.classes_idx_image,'.csv')
        self.labels = fold.make_dataset(self.label_dir, self.classes_idx_label,'.csv')
        if len(self.images) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + self.image_dir + "\n"
                               "Supported extensions are: " + ",".join(["CSV"])))
        if len(self.labels) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + self.label_dir + "\n"
                               "Supported extensions are: " + ",".join(["CSV"])))
        if len(self.images) != len(self.labels):
            raise(RuntimeError("The images and labels are not paired."))
        
        
    def __getitem__(self, index):
        path_image, index_image = self.images[index]
        path_label, index_label = self.labels[index]
        image = torch.tensor(self.loader(path_image,header=None).values,dtype = torch.float)
        label = torch.tensor(self.loader(path_label,header=None).values,dtype = torch.long)
        if self.transform_image:
            image = self.transform_image(image)
        if self.transform_label:
            label = self.transform_label(label)
        return image,label
    
    def __len__(self):
        return len(self.images)

# set parameters
#train_batch_size = 5
#test_batch_size = 100
#total_train = 100
#total_test = 100
'''
epochs = 20
eta = 0.001
num_classes = 2
input_channels = 1
network_depth = 5
'''

# load the dataset
train_dataset = SegmentationDataset(opt.path_image_new, opt.path_label_new, train=True, transform_image=None,transform_label=None) # Supply proper root_dir
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=opt.train_batch_size, shuffle=True)

# define the UNet
network = UNet(opt.num_classes, in_channels=opt.input_channels, depth=opt.network_depth)

# train
for epoch in list(range(opt.epochs)):
    # set the start time point
    since = time.time()
    # initialization
    optimizer = optim.Adam(network.parameters(),lr = opt.eta)
    loss_function = CrossEntropy2d()
    loss_total = 0
    dice_total = 0

    for batch_idx, (image,label) in enumerate(train_loader):
        # N * H * W * C
        image_train = image.unsqueeze(-1)
        #label_train = label.unsqueeze(-1)
        real_image = image_train
        #real_label = label_train
        '''
        image1 = image[1,:,:]
        Label1 = label[1,:,:]

        label_np = Label1.numpy()
        image_np = image1.numpy()
        l1 = Image.fromarray(label_np.astype('uint8'))
        i1 = Image.fromarray(image_np.astype('uint8'))
        l1 = l1.convert('L')
        i1 = i1.convert('L')
        plt.subplot(121)
        plt.imshow(l1)
        plt.subplot(122)
        plt.imshow(i1)
        plt.show()
        '''

        #real_label = torch.zeros(label_train.size()[0],label_train.size()[1],label_train.size()[2],2,dtype = torch.int64)
        #real_label[:][:][:][0] = label_train[:][:][:][0]
        #real_label[:][:][:][1] = 1 - label_train[:][:][:][0]   
        # N * C * H * W
        real_image = real_image.permute(0,3,1,2)
        #real_label = real_label.permute(0,3,1,2)

        est_label = network(real_image)


        # H * W * N * C
        '''
        est_label = est_label.permute(2, 3, 0, 1).contiguous().view(-1,opt.num_classes)
        real_label = real_label.permute(2, 3, 0, 1).contiguous().view(-1, 1)
        real_label = torch.squeeze(real_label)
        print(real_label.size())
        print(est_label.size())
        '''
        #print(real_label[0,:])
        weight = torch.FloatTensor(opt.num_classes)
        weight[0] = 1
        weight[1] = 1000
        loss = loss_function(est_label,label,weight = weight)
        


        # dice score
        _,prediction = torch.max(est_label,1)
        annotation = label
        a,_ = torch.max(annotation,0)
        b,_ = torch.max(prediction,0)
        c = (torch.sum(annotation * prediction)).type(torch.FloatTensor)
        #print(c)
        d = (torch.sum(annotation) + torch.sum(prediction)).type(torch.FloatTensor)
        dice = (c*2.0)/(d+0.00001)
        if (torch.sum(annotation)).item() == 0 & (torch.sum(prediction)).item() == 0:
            dice = 1

        print('[epoch %d/%d] [batch %d/%d] [loss: %.5f] [dice: %.5f]'%(epoch, opt.epochs,batch_idx,len(train_loader),loss,dice))
        loss.backward()
        optimizer.step()
        loss_total = loss_total + loss
        dice_total = dice_total + dice


    loss_total = loss_total / (batch_idx + 1)
    dice_total = dice_total / (batch_idx + 1)
    print('[epoch %d] [Total_loss: %.5f] [Total_dice: %.5f]'%(epoch,loss_total,dice_total))
    # set the end time point 
    time_elapsed = time.time() - since
    
