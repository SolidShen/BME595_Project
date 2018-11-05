#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 19:58:05 2018

@author: wangxiaokai
"""
from __future__ import print_function
import os
os.chdir('/home/wang4001/dl_project')
path_image = '/home/wang4001/dl_project/sorted_images/'
path_label = '/home/wang4001/dl_project/sorted_labels/'
path_image_new = '/home/wang4001/dl_project/image_new/'
path_label_new = '/home/wang4001/dl_project/label_new/'
if not os.path.exists(path_image_new):
    os.makedirs(path_image_new)
if not os.path.exists(path_label_new):
    os.makedirs(path_label_new)
# prepare the data
os.chdir(path_image)
num_image = 0
for name in os.listdir(path_image):
    if os.path.isfile(name):
        file_name,file_ext = os.path.splitext(name)
        if file_ext == '.csv':
            num_image = num_image + 1
            # Create folder if not present, and move image into proper folder 
            new_sub_path = os.path.join(path_image_new,file_name)
            if not os.path.exists(new_sub_path): 
                os.makedirs(new_sub_path)
            os.rename(os.path.join(path_image,name),os.path.join(new_sub_path,name))
os.chdir(path_label)
num_label = 0
for name in os.listdir(path_label):
    if os.path.isfile(name):
        file_name,file_ext = os.path.splitext(name)
        if file_ext == '.csv':
            num_label = num_label + 1
            # Create folder if not present, and move image into proper folder 
            new_sub_path = os.path.join(path_label_new,file_name)
            if not os.path.exists(new_sub_path): 
                os.makedirs(new_sub_path)
            os.rename(os.path.join(path_label,name),os.path.join(new_sub_path,name))
