#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:58:05 2018

@author: wangxiaokai
"""
from __future__ import print_function
import os
os.chdir('/home/wang4001/dl_project')
path_image = '/home/wang4001/dl_project/sorted_images_2d/'
path_label = '/home/wang4001/dl_project/sorted_labels_2d/'
path_image_train = '/home/wang4001/dl_project/image_train_2d/'
path_label_train = '/home/wang4001/dl_project/label_train_2d/'
path_image_val = '/home/wang4001/dl_project/image_val_2d/'
path_label_val = '/home/wang4001/dl_project/label_val_2d/'

if not os.path.exists(path_image_train):
    os.makedirs(path_image_train)
if not os.path.exists(path_label_train):
    os.makedirs(path_label_train)
if not os.path.exists(path_image_val):
    os.makedirs(path_image_val)
if not os.path.exists(path_label_val):
    os.makedirs(path_label_val)
# prepare the data
val_sub_id = set('02','13','28') 
os.chdir(path_image)
num_image_train = 0
num_image_val = 0
for name in os.listdir(path_image):
    if os.path.isfile(name):
        file_name,file_ext = os.path.splitext(name)
        if file_ext == '.csv':
		file_sub_id = file_name[0:2] 
		if file_sub_id in val_sub_id:
            		num_image_val = num_image_val + 1
            		# Create folder if not present, and move image into proper folder 
			val_sub_path = os.path.join(path_image_val,file_name)   
			if not os.path.exists(val_sub_path): 
                		os.makedirs(val_sub_path)  
			os.rename(os.path.join(path_image,name),os.path.join(val_sub_path,name))       
		else:					
			num_image_train	= num_image_train + 1		
			# Create folder if not present, and move image into proper folder 
			train_sub_path = os.path.join(path_image_train,file_name)
            		if not os.path.exists(train_sub_path): 
                		os.makedirs(train_sub_path)
            		os.rename(os.path.join(path_image,name),os.path.join(train_sub_path,name))
os.chdir(path_label)
num_label_train = 0
num_label_val = 0
for name in os.listdir(path_label):
    if os.path.isfile(name):
        file_name,file_ext = os.path.splitext(name)
        if file_ext == '.csv':
		file_sub_id = file_name[0:2] 
		if file_sub_id in val_sub_id:
			num_label_val = num_label_val + 1
            		# Create folder if not present, and move image into proper folder 
            		val_sub_path = os.path.join(path_label_val,file_name)
            		if not os.path.exists(val_sub_path): 
                		os.makedirs(val_sub_path)
            		os.rename(os.path.join(path_label,name),os.path.join(val_sub_path,name))
		else:
            		num_label_train = num_label_train + 1
            		# Create folder if not present, and move image into proper folder 
            		train_sub_path = os.path.join(path_label_train,file_name)
            		if not os.path.exists(train_sub_path): 
                		os.makedirs(train_sub_path)
            		os.rename(os.path.join(path_label,name),os.path.join(train_sub_path,name))
