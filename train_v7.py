#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  16 16:33:24 2018

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

########################################### arguements####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("-e","--epochs",default=50,help="total number of epochs")
parser.add_argument("-se","--save_epoch_index",default=50,help="the epoch index of the results that are going to be saved")
args = parser.parse_args()
##########################################################################################################################


########################################### path #########################################################################
os.chdir('/home/wang4001/dl_project')
path_image_train = '/home/wang4001/dl_project/image_train_2d/'
path_label_train = '/home/wang4001/dl_project/label_train_2d/'
path_save_results = '/home/wang4001/dl_project/results/results_v7'
if not os.path.exists(path_save_results):
	os.makedirs(path_save_results)
##########################################################################################################################


########################################### device #######################################################################
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
##########################################################################################################################


########################################### functions ####################################################################
# define the dice score
def func_dice_score(func_est_label,func_label):
	# est_label: N,C,H,W
	# label: N,H,W
	_,prediction = torch.max(func_est_label,1)
	annotation = func_label
	a,_ = torch.max(annotation,0)
	b,_ = torch.max(prediction,0)
	c = (torch.sum(annotation * prediction)).type(torch.cuda.FloatTensor)
	d = (torch.sum(annotation) + torch.sum(prediction)).type(torch.cuda.FloatTensor)
	dice = (c*2.0)/(d+0.00001)
	if ((torch.sum(annotation)).item() == 0) & ((torch.sum(prediction)).item() == 0):
		dice = torch.ones(1,dtype=torch.float32,device=device)
	return dice
# define the precision score
def func_precision_score(func_est_label,func_label):
	# est_label: N,C,H,W
	# label: N,H,W
	_,prediction = torch.max(func_est_label,1)
	annotation = func_label
	TP = (torch.sum(annotation * prediction)).type(torch.cuda.FloatTensor)
	# precision = TP / (TP + FP)
	precision = TP / (torch.sum(prediction)+0.00001).type(torch.cuda.FloatTensor)
	if ((torch.sum(annotation)).item() == 0) & ((torch.sum(prediction)).item() == 0):
		precision = torch.ones(1,dtype=torch.float32,device=device)
	return precision
# define the sensitivity score
def func_sensitivity_score(func_est_label,func_label):
	# est_label: N,C,H,W
	# label: N,H,W
	_,prediction = torch.max(func_est_label,1)
	annotation = func_label
	TP = (torch.sum(annotation * prediction)).type(torch.cuda.FloatTensor)
	# sensitivity = TP / (TP + FN)
	sensitivity = TP / (torch.sum(annotation)+0.00001).type(torch.cuda.FloatTensor)
	if ((torch.sum(annotation)).item() == 0) & ((torch.sum(prediction)).item() == 0):
		sensitivity = torch.ones(1,dtype=torch.float32,device=device)
	return sensitivity
# define the specificity score
def func_specificity_score(func_est_label,func_label):
	# est_label: N,C,H,W
	# label: N,H,W
	_,prediction = torch.max(func_est_label,1)
	annotation = func_label
	TP = (torch.sum(annotation * prediction)).type(torch.cuda.FloatTensor)
	sum_whole = torch.tensor(annotation.size()[0]*annotation.size()[1]*annotation.size()[2],dtype=torch.float32,device=device)
	sum_prediction = torch.tensor(torch.sum(prediction),dtype=torch.float32,device=device)
	sum_annotation = torch.tensor(torch.sum(annotation),dtype=torch.float32,device=device)
	TN = sum_whole - sum_prediction - sum_annotation + TP
	TN_sum_FP = sum_whole - sum_annotation
	# specificity = TN / (TN + FP)
	specificity = TN / (TN_sum_FP+0.00001)
	if (TN.item() == 0) & (TN_sum_FP.item() == 0):
		specificity = torch.ones(1,dtype=torch.float32,device=device)
	return specificity
# define the dataset
class func_SegmentationDataset(torch.utils.data.Dataset):
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

# define function to change learning rate adaptively
def func_adjust_learning_rate(func_optimizer,func_epoch,func_eta):
	current_eta = func_eta * (0.1 ** (func_epoch // 10))
	for param_group in func_optimizer.param_groups:
		param_group['lr'] = current_eta
	return current_eta
##########################################################################################################################


########################################### parameters ###################################################################
# set train parameters
param_train_batch_size = 9
param_train_total_size = 12960
param_epochs = int(args.epochs)
param_eta = 0.00001 # initial eta
param_num_classes = 2
param_input_channels = 1
param_network_depth = 5
param_weight_cuda = torch.tensor([0.5,0.5],device=device)
# initialize evaluation parameters
eval_train_loss = torch.zeros(param_epochs,device=device)
eval_train_dice = torch.zeros(param_epochs,device=device)
eval_train_precision = torch.zeros(param_epochs,device=device)
eval_train_sensitivity = torch.zeros(param_epochs,device=device)
eval_train_specificity = torch.zeros(param_epochs,device=device)
##########################################################################################################################



########################################## datasets ######################################################################
# load the dataset
train_dataset = func_SegmentationDataset(path_image_train, path_label_train, train=True, transform_image=None,transform_label=None) # Supply proper root_dir
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=param_train_batch_size, shuffle=True)
##########################################################################################################################


########################################## Initialization ################################################################
# define the UNet
network = UNet(param_num_classes, in_channels=param_input_channels, depth=param_network_depth)
network_cuda = network.to(device)
print(network_cuda)
# initialize optimizer and loss function
def_optimizer = optim.Adam(network_cuda.parameters(),lr = param_eta)
def_loss_function = CrossEntropy2d()
def_softmax = torch.nn.Softmax2d()
##########################################################################################################################


########################################## train #########################################################################
for epoch in list(range(param_epochs)):
	# updata optimizer at specific epochs
	param_eta_update = func_adjust_learning_rate(def_optimizer,epoch,param_eta)
	# set the start time point
	since = time.time()
	for batch_idx, (image,label) in enumerate(train_loader):
		since_temp = time.time()
		# N * H * W * C
		real_image = image.unsqueeze(-1)
		real_label = label.unsqueeze(-1)
		# N * C * H * W
		real_image = real_image.permute(0,3,1,2)
		real_label = real_label.permute(0,3,1,2)
		# forward to yield estimation
		est_label = network_cuda(real_image)
		# Calculate dice, precision, sensitivity, and specificity: est_label - N * C * H * W; label - N * H * W
		train_dice = func_dice_score(def_softmax(est_label),real_label.squeeze())
		train_precision = func_precision_score(def_softmax(est_label),real_label.squeeze())
		train_sensitivity = func_sensitivity_score(def_softmax(est_label),real_label.squeeze())
		train_specificity = func_specificity_score(def_softmax(est_label),real_label.squeeze())
		# save image results
		_,est_label_save = torch.max(est_label,1)
		if epoch == int(args.save_epoch_index) - 1:
			for index_in_batch in list(range(param_train_batch_size)):
				if index_in_batch == 5:
					imagename = 'train_image_epoch'+str(epoch+1)+'_batch'+str(batch_idx+1)+'_id'+str(index_in_batch+1)+'.jpg'
					labelname = 'train_label_epoch'+str(epoch+1)+'_batch'+str(batch_idx+1)+'_id'+str(index_in_batch+1)+'.jpg'
					est_labelname = 'train_est_label_epoch'+str(epoch+1)+'_batch'+str(batch_idx+1)+'_id'+str(index_in_batch+1)+'.jpg'
					scipy.misc.imsave(os.path.join(path_save_results,imagename),real_image[index_in_batch][:][:][:].squeeze().cpu().numpy())
					scipy.misc.imsave(os.path.join(path_save_results,labelname),real_label[index_in_batch][:][:][:].squeeze().cpu().numpy())
					scipy.misc.imsave(os.path.join(path_save_results,est_labelname),est_label_save[index_in_batch][:][:].squeeze().cpu().detach().numpy())
		# loss function is CrossEntropy2d()
		# est_label: N * C * H * W; label: N * H * W
		train_loss = def_loss_function.forward(est_label, real_label.squeeze(), param_weight_cuda)
		# display results
		print('Training Stage')
		print('Epoch [{:.0f}/{:.0f}] Batch [{:.0f}/{:.0f}]'.format(epoch+1,param_epochs,batch_idx+1,param_train_total_size/param_train_batch_size))
		print('CrossEntropy Loss {:.5f}'.format(train_loss))
		print('DICE {:.5f}'.format(train_dice))
		print('Precision {:.5f}'.format(train_precision))
		print('Sensitivity {:.5f}'.format(train_sensitivity))
		print('Specificity {:.5f}'.format(train_specificity))
		time_partial = time.time() - since_temp
		print('Time {:.2f}s\n'.format(time_partial))
		train_loss.backward()
		def_optimizer.step()
		eval_train_loss[epoch] = eval_train_loss[epoch] + train_loss
		eval_train_dice[epoch] = eval_train_dice[epoch] + train_dice
		eval_train_precision[epoch] = eval_train_precision[epoch] + train_precision
		eval_train_sensitivity[epoch] = eval_train_sensitivity[epoch] + train_sensitivity
		eval_train_specificity[epoch] = eval_train_specificity[epoch] + train_specificity
	# average for the current epoch
	eval_train_loss[epoch] = eval_train_loss[epoch] / (batch_idx + 1)
	eval_train_dice[epoch] = eval_train_dice[epoch] / (batch_idx + 1)
	eval_train_precision[epoch] = eval_train_precision[epoch] / (batch_idx + 1)
	eval_train_sensitivity[epoch] = eval_train_sensitivity[epoch] / (batch_idx + 1)
	eval_train_specificity[epoch] = eval_train_specificity[epoch] / (batch_idx + 1)
	# save trained parameters
	parametername = 'parameters_epoch'+str(epoch+1)
	torch.save(network_cuda.state_dict(),os.path.join(path_save_results,parametername))
	print('Done for the Epoch [{:.0f}/{:.0f}]'.format(epoch+1,param_epochs))
	print('Adam Optimizer Learning Rate {:.11f}'.format(param_eta_update))
	print('Averaged CrossEntropy Loss: {:.5f}'.format(eval_train_loss[epoch]))
	print('Averaged DICE: {:.5f}'.format(eval_train_dice[epoch]))
	print('Averaged Precision: {:.5f}'.format(eval_train_precision[epoch]))
	print('Averaged Sensitivity: {:.5f}'.format(eval_train_sensitivity[epoch]))
	print('Averaged Specificity: {:.5f}'.format(eval_train_specificity[epoch]))
# set the end time point
time_elapsed = time.time() - since
print('Training is Done.')
print('Total time {:.2f}s'.format(time_elapsed))
########################################## results #######################################################################
modelname = 'trained_unet.pkl'
torch.save(network_cuda,os.path.join(path_save_results,modelname))
# save the evaluation parameters
path_train_loss = 'train_loss'
path_train_dice = 'train_dice'
path_train_precision = 'train_precision'
path_train_sensitivity = 'train_sensitivity'
path_train_specificity = 'train_specificity'
np.save(os.path.join(path_save_results,path_train_loss),eval_train_loss.cpu().detach().numpy())
np.save(os.path.join(path_save_results,path_train_dice),eval_train_dice.cpu().numpy())
np.save(os.path.join(path_save_results,path_train_precision),eval_train_precision.cpu().numpy())
np.save(os.path.join(path_save_results,path_train_sensitivity),eval_train_sensitivity.cpu().numpy())
np.save(os.path.join(path_save_results,path_train_specificity),eval_train_specificity.cpu().numpy())
##########################################################################################################################


########################################## figures #######################################################################
# plot figures
epoch_list = list(range(1,param_epochs+1))
# loss vs epochs
fig1 = plt.figure()
ax1 = fig1.add_subplot(1,1,1)
ax1.plot(epoch_list,eval_train_loss.cpu().detach().numpy())
ax1.set_title('CE loss vs Epochs')
plt.show()
# dice vs epochs
fig2 = plt.figure()
ax2 = fig2.add_subplot(1,1,1)
ax2.plot(epoch_list,eval_train_dice.cpu().numpy())
ax2.set_title('DICE vs Epochs')
plt.show()
# precision vs epochs
fig3 = plt.figure()
ax3 = fig3.add_subplot(1,1,1)
ax3.plot(epoch_list,eval_train_precision.cpu().numpy())
ax3.set_title('Precision vs Epochs')
plt.show()
# sensitivity vs epochs
fig4 = plt.figure()
ax4 = fig4.add_subplot(1,1,1)
ax4.plot(epoch_list,eval_train_sensitivity.cpu().numpy())
ax4.set_title('Sensitivity vs Epochs')
plt.show()
# specificity vs epochs
fig5 = plt.figure()
ax5 = fig5.add_subplot(1,1,1)
ax5.plot(epoch_list,eval_train_specificity.cpu().numpy())
ax5.set_title('Specificity vs Epochs')
plt.show()
##########################################################################################################################
