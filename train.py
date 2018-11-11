#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 12:27:04 2018

@author: wangxiaokai
"""
from __future__ import print_function
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
import scipy.misc
from model import UNet 
from CrossEntropy2d import CrossEntropy2d

os.chdir('/home/wang4001/dl_project')
path_image_new = '/home/wang4001/dl_project/image_new/'
path_label_new = '/home/wang4001/dl_project/label_new/'
path_save_images = '/home/wang4001/dl_project/results'
if not os.path.exists(path_save_images):
	os.makedirs(path_save_images)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# define the dataset
class SegmentationDataset(torch.utils.data.Dataset):
	def __init__(self, image_dir, label_dir, train, transform_image, transform_label,loader=pd.read_csv):
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
		label = torch.tensor(self.loader(path_label,header=None).values)
		image_cuda = image.to(device)
		label_cuda = label.to(device)
		if self.transform_image:
			image_cuda = self.transform_image(image_cuda)
		if self.transform_label:
			label_cuda = self.transform_label(label_cuda)
		return image_cuda,label_cuda
	def __len__(self):
		return len(self.images)
# set parameters
train_batch_size = 26
#test_batch_size = 100
total_train = 14430
#total_test = 100
epochs = 5
eta = 0.001
num_classes = 2
input_channels = 1
network_depth = 5
# load the dataset
train_dataset = SegmentationDataset(path_image_new, path_label_new, train=True, transform_image=None,transform_label=None) # Supply proper root_dir
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)
# define the UNet
network = UNet(num_classes, in_channels=input_channels, depth=network_depth)
network_cuda = network.to(device)
print(network_cuda)
# train
os.chdir(path_save_images)
for epoch in list(range(epochs)):
	# set the start time point
	since = time.time()
	# initialization
	optimizer = optim.Adam(network.parameters(),lr = eta)
	# loss function is nn.CrossEntropy by default
	# loss_function = nn.CrossEntropyLoss()
	# loss function is CrossEntropy2d()
	loss_function = CrossEntropy2d()
	loss_total = 0
	for batch_idx, (image,label) in enumerate(train_loader):
		label = label/255
		since_temp = time.time()
		# N * H * W * C
		real_image = image.unsqueeze(-1)
		real_label = label.unsqueeze(-1)
		# N * C * H * W
		real_image = real_image.permute(0,3,1,2)
		real_label = real_label.permute(0,3,1,2)
		# forward to yield estimation
		est_label = network_cuda(real_image)
		# save results
		if epoch == epochs - 1:
			for index_in_batch in list(range(train_batch_size)):
				imagename = 'image_epoch'+str(epoch+1)+'_batch'+str(batch_idx)+'_id'+str(index_in_batch)+'.jpg'
				labelname = 'label_epoch'+str(epoch+1)+'_batch'+str(batch_idx)+'_id'+str(index_in_batch)+'.jpg'
				est_labelname = 'est_label_epoch'+str(epoch+1)+'_batch'+str(batch_idx)+'_id'+str(index_in_batch)+'.jpg'
				scipy.misc.imsave(imagename,real_image[index_in_batch][:][:][:].squeeze().cpu().numpy())
				scipy.misc.imsave(labelname,real_label[index_in_batch][:][:][:].squeeze().cpu().numpy())
				scipy.misc.imsave(est_labelname,est_label[index_in_batch][0][:][:].squeeze().cpu().detach().numpy())
		# loss function is nn.CrossEntropy by default
		# H * W * N * C
		# est_label = est_label.permute(2, 3, 0, 1).contiguous().view(-1,num_classes)
		# real_label = real_label.permute(2, 3, 0, 1).contiguous().view(-1,1)
		# loss = loss_function(est_label, torch.max(real_label,1)[1])
		# loss_ave = torch.mean(loss)
		# loss function is CrossEntropy2d()
		# est_label: N * C * H * W; label: N * H * W
		weight = torch.FloatTensor(opt.num_classes)
		sum_weight = torch.sum(label)
		size = label.size()
		all_weight = size[0]*size[1]*size[2]
		weight[0] = sum_weight
		weight[1] = all_weight-sum_weight
		weight_cuda = weight.to(device)
		loss = loss_function.forward(est_label, label, weight_cuda)
		print('Epoch [{:.0f}/{:.0f}] Batch [{:.0f}/{:.0f}]'.format(epoch+1,epochs,batch_idx+1,total_train/train_batch_size))
		print('CrossEntropy Loss {:.8f}'.format(loss))
		time_partial = time.time() - since_temp
		print('Time {:.2f}s\n'.format(time_partial))
		loss.backward()
		optimizer.step()
		loss_total = loss_total + loss
	loss_total = loss_total / (batch_idx + 1)
	print('Done for the Epoch [{:.0f}/{:.0f}].\n'.format(epoch+1,epochs))
	print('Averaged CrossEntropy Loss {:.2f}'.format(loss_total))
	# set the end time point
time_elapsed = time.time() - since
print('Training is Done.\n')
print('Total time {:.2f}s'.format(time_elapsed))
