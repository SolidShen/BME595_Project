import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from losses_BDCLSTM import DICELossMultiClass
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.transforms as tr
from data import SegmentationDatasetBDCLSTM
from CLSTM import BDCLSTM
from model import *
from torchvision.utils import save_image
import torch
import torch.nn as nn
from torch.autograd import Variable
import os



# %% Training settings
parser = argparse.ArgumentParser(description='UNet+BDCLSTM for BraTS Dataset')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--model_params', type=str, default='/Users/shenguangyu1/Desktop/purdue/BME595/proj/result/results/results_v8/parameters_epoch50',help='UNET model params path')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--train', action='store_true', default=True,
                    help='Argument to train model (default: False)')
parser.add_argument('--epochs', type=int, default=1, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--mom', type=float, default=0.99, metavar='MOM',
                    help='SGD momentum (default=0.99)')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='enables CUDA training (default: False)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='batches to wait before logging training status')
parser.add_argument('--size', type=int, default=256, metavar='N',
                    help='imsize')
parser.add_argument('--drop', action='store_true', default=False,
                    help='enables drop')
parser.add_argument('--num_of_classes', type=int, default=2, help='num of classes for UNET')
parser.add_argument('--network_depth',type=int, default=5, help='num of network_depth')
parser.add_argument('--input_channels',type=int, default=1, help='num of input channels for UNET')
parser.add_argument('--path_image_train',type=str, default='/Users/shenguangyu1/Desktop/purdue/BME595/proj/data/image_train_2d/',help='train image path')
parser.add_argument('--path_label_train',type=str, default='/Users/shenguangyu1/Desktop/purdue/BME595/proj/data/label_train_2d/',help='train label path')
parser.add_argument('--path_image_val',type=str, default='/Users/shenguangyu1/Desktop/purdue/BME595/proj/data/image_val_2d/',help='test image path')
parser.add_argument('--path_label_val',type=str, default='/Users/shenguangyu1/Desktop/purdue/BME595/proj/data/label_val_2d/',help='test label path')
parser.add_argument('--path_save_results', type=str,default = '/Users/shenguangyu1/Desktop/purdue/BME595/proj/results/',help='result path')

args = parser.parse_args()
print(args)

if os.path.exists(args.path_save_results) == False:
    os.makedirs('%s/sample_img_train' % args.path_save_results)
    os.makedirs('%s/sample_img_val' % args.path_save_results)

#################################################################################
args.cuda = args.cuda and torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if args.cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if args.cuda else torch.LongTensor
if args.cuda:
    print("We are on the GPU!")
#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

################################ function #######################################

def_softmax = torch.nn.Softmax2d()
def dice_loss(pred, target):
    """This definition generalize to real valued pred and target vector.
This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    
    import torch.nn.functional as F
    cuda = True if torch.cuda.is_available() else False
    
    smooth = 1.
    
    target = target.type(LongTensor)
    batch_size,_,H,W = target.size()
    target_one_hot = torch.zeros(batch_size,2,H,W).type(LongTensor).scatter_(1,target,1)
    target_one_hot = target_one_hot.type(FloatTensor)
    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target_one_hot.contiguous().view(-1)
    intersection = (iflat * tflat).sum()

    A_sum = torch.sum(iflat * iflat)
    B_sum = torch.sum(tflat * tflat)
    
    return 1 - ((2. * intersection + smooth) / (A_sum + B_sum + smooth) )


# define the dice score
def func_dice_score(func_est_label,func_label):
	# est_label: N,C,H,W
	# label: N,H,W
	_,prediction = torch.max(func_est_label,1)
	annotation = func_label
	a,_ = torch.max(annotation,0)
	b,_ = torch.max(prediction,0)
	c = (torch.sum(annotation * prediction)).type(FloatTensor)
	d = (torch.sum(annotation) + torch.sum(prediction)).type(FloatTensor)
	dice = (c*2.0)/(d+0.00001)
	if ((torch.sum(annotation)).item() == 0) & ((torch.sum(prediction)).item() == 0):
		dice = torch.ones(1).type(FloatTensor)
	return dice
# define the precision score
def func_precision_score(func_est_label,func_label):
	# est_label: N,C,H,W
	# label: N,H,W
	_,prediction = torch.max(func_est_label,1)
	annotation = func_label
	TP = (torch.sum(annotation * prediction)).type(FloatTensor)
	# precision = TP / (TP + FP)
	precision = TP / (torch.sum(prediction)+0.00001).type(FloatTensor)
	if ((torch.sum(annotation)).item() == 0) & ((torch.sum(prediction)).item() == 0):
		precision = torch.ones(1).type(FloatTensor)
	return precision
# define the sensitivity score
def func_sensitivity_score(func_est_label,func_label):
	# est_label: N,C,H,W
	# label: N,H,W
	_,prediction = torch.max(func_est_label,1)
	annotation = func_label
	TP = (torch.sum(annotation * prediction)).type(FloatTensor)
	# sensitivity = TP / (TP + FN)
	sensitivity = TP / (torch.sum(annotation)+0.00001).type(FloatTensor)
	if ((torch.sum(annotation)).item() == 0) & ((torch.sum(prediction)).item() == 0):
		sensitivity = torch.ones(1).type(FloatTensor)
	return sensitivity
# define the specificity score
def func_specificity_score(func_est_label,func_label):
	# est_label: N,C,H,W
	# label: N,H,W
	_,prediction = torch.max(func_est_label,1)
	annotation = func_label
	TP = (torch.sum(annotation * prediction)).type(FloatTensor)
	sum_whole = torch.tensor(annotation.size()[0]*annotation.size()[1]*annotation.size()[2]).type(FloatTensor)
	sum_prediction = torch.tensor(torch.sum(prediction)).type(FloatTensor)
	sum_annotation = torch.tensor(torch.sum(annotation)).type(FloatTensor)
	TN = sum_whole - sum_prediction - sum_annotation + TP
	TN_sum_FP = sum_whole - sum_annotation
	# specificity = TN / (TN + FP)
	specificity = TN / (TN_sum_FP+0.00001)
	if (TN.item() == 0) & (TN_sum_FP.item() == 0):
		specificity = torch.ones(1).type(FloatTensor)
	return specificity
###################################################################################################################

###################################Evaluation_Variables###############################################
eval_train_loss = torch.zeros(args.epochs).type(FloatTensor)
eval_train_dice = torch.zeros(args.epochs).type(FloatTensor)
eval_train_precision = torch.zeros(args.epochs).type(FloatTensor)
eval_train_sensitivity = torch.zeros(args.epochs).type(FloatTensor)
eval_train_specificity = torch.zeros(args.epochs).type(FloatTensor)


eval_val_loss = torch.zeros(args.epochs).type(FloatTensor)
eval_val_dice = torch.zeros(args.epochs).type(FloatTensor)
eval_val_precision = torch.zeros(args.epochs).type(FloatTensor)
eval_val_sensitivity = torch.zeros(args.epochs).type(FloatTensor)
eval_val_specificity = torch.zeros(args.epochs).type(FloatTensor)

###################################################################################################################

# %% Loading in the Dataset
train_dataset = SegmentationDatasetBDCLSTM(args.path_image_train, args.path_label_train, train=True, transform_image=None,transform_label=None) # Supply proper root_dir
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

test_dataset = SegmentationDatasetBDCLSTM(args.path_image_val, args.path_label_val, train=False, transform_image=None,transform_label=None) # Supply proper root_dir
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.test_batch_size, shuffle=False)


# %% Loading in the models
unet = UNet(args.num_of_classes, in_channels=args.input_channels, depth=args.network_depth)
unet.load_state_dict(torch.load(args.model_params))
model = BDCLSTM(input_channels=64, hidden_channels=[64])

if args.cuda:
    unet.cuda()
    model.cuda()

# Setting Optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.mom)
#criterion = DICELossMultiClass()

# Define Training Loop


def train(epoch):
    model.train()
    for batch_idx, (image1, image2, image3, mask) in enumerate(train_loader):
        if args.cuda:
            image1, image2, image3, mask = image1.cuda(), \
                image2.cuda(), \
                image3.cuda(), \
                mask.cuda()

        image1, image2, image3, mask = Variable(image1), \
            Variable(image2), \
            Variable(image3), \
            Variable(mask)
        image1 = (image1.unsqueeze(-1)).permute(0,3,1,2)
        image2 = (image2.unsqueeze(-1)).permute(0,3,1,2)
        image3 = (image3.unsqueeze(-1)).permute(0,3,1,2)
        mask = (mask.unsqueeze(-1)).permute(0,3,1,2)
        mask = mask.type(FloatTensor)
        optimizer.zero_grad()
        map1 = unet(image1,return_features=True)
        map2 = unet(image2,return_features=True)
        map3 = unet(image3,return_features=True)
        output = model(map1, map2, map3)
        loss = dice_loss(output, mask)
        mask = mask.type(LongTensor)

        train_dice = func_dice_score(def_softmax(output),mask.squeeze())
        train_precision = func_precision_score(def_softmax(output),mask.squeeze())
        train_sensitivity = func_sensitivity_score(def_softmax(output),mask.squeeze())
        train_specificity = func_specificity_score(def_softmax(output),mask.squeeze())

        eval_train_loss[epoch] = eval_train_loss[epoch] + loss
        eval_train_dice[epoch] = eval_train_dice[epoch] + train_dice
        eval_train_precision[epoch] = eval_train_precision[epoch] + train_precision
        eval_train_sensitivity[epoch] = eval_train_sensitivity[epoch] + train_sensitivity
        eval_train_specificity[epoch] = eval_train_specificity[epoch] + train_specificity
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\tLoss: {:.6f} Dice: {:.6f}  Precision: {:.6f}  Sensitivity: {:.6f}  Specificity: {:.6f}'.format(
                epoch, batch_idx * len(image1), len(train_loader.dataset),
                loss.item(),train_dice.item(),train_precision.item(),train_sensitivity.item(), train_specificity.item()))
            
            _,output_img = torch.max(output,1)
            '''
            _,img2 = torch.max(img2,1)
            img2 = img2.unsqueeze(1).type(torch.cuda.LongTensor)
            '''
            output_img = output_img.unsqueeze(1).type(LongTensor)
            img_sample = torch.cat((mask.data,output_img.data),0)
            save_image(img_sample,'%s/sample_img_train/train_%s_epoch_%s.png' % (args.path_save_results,i,batch_idx),nrow=args.batch_size,normalize=False)

    eval_train_loss[epoch] = eval_train_loss[epoch] / (batch_idx + 1)
    eval_train_dice[epoch] = eval_train_dice[epoch] / (batch_idx + 1)
    eval_train_precision[epoch] = eval_train_precision[epoch] / (batch_idx + 1)
    eval_train_sensitivity[epoch] = eval_train_sensitivity[epoch] / (batch_idx + 1)
    eval_train_specificity[epoch] = eval_train_specificity[epoch] / (batch_idx + 1)

    print('Train Epoch: {} done\tAvg Loss: {:.6f} Avg Dice: {:.6f} Avg Precision: {:.6f} Avg Sensitivity: {:.6f} Avg Specificity: {:.6f}'.format(
                epoch,
                eval_train_loss[epoch].item(),eval_train_dice[epoch].item(),eval_train_precision[epoch].item(),eval_train_sensitivity[epoch].item(), eval_train_specificity[epoch].item()))


## Define Testing Loop
def test(epoch,train_accuracy=False):
    test_loss = 0

    if train_accuracy == True:
        loader = train_loader
    else:
        loader = test_loader

    for batch_idx,(image1, image2, image3, mask) in enumerate(loader):
        if args.cuda:
            image1, image2, image3, mask = image1.cuda(), \
                image2.cuda(), \
                image3.cuda(), \
                mask.cuda()

        image1, image2, image3, mask = Variable(image1, volatile=True), \
            Variable(image2, volatile=True), \
            Variable(image3, volatile=True), \
            Variable(mask, volatile=True)

        image1 = (image1.unsqueeze(-1)).permute(0,3,1,2)
        image2 = (image2.unsqueeze(-1)).permute(0,3,1,2)
        image3 = (image3.unsqueeze(-1)).permute(0,3,1,2)
        mask = (mask.unsqueeze(-1)).permute(0,3,1,2)
        mask = mask.type(FloatTensor)
        map1 = unet(image1, return_features=True)
        map2 = unet(image2, return_features=True)
        map3 = unet(image3, return_features=True)


        output = model(map1, map2, map3)
        test_loss += dice_loss(output, mask)

        mask = mask.type(LongTensor)
        val_dice = func_dice_score(def_softmax(output),mask.squeeze(1))
        val_precision = func_precision_score(def_softmax(output),mask.squeeze(1))
        val_sensitivity = func_sensitivity_score(def_softmax(output),mask.squeeze(1))
        val_specificity = func_specificity_score(def_softmax(output),mask.squeeze(1))

        eval_val_loss[epoch] = eval_val_loss[epoch] + test_loss
        eval_val_dice[epoch] = eval_val_dice[epoch] + val_dice
        eval_val_precision[epoch] = eval_val_precision[epoch] + val_precision
        eval_val_sensitivity[epoch] = eval_val_sensitivity[epoch] + val_sensitivity
        eval_val_specificity[epoch] = eval_val_specificity[epoch] + val_specificity
        if batch_idx % args.log_interval == 0:
            _,output_img = torch.max(output,1)
            #_,img2 = torch.max(img2,1)
            #img2 = img2.unsqueeze(1).type(torch.cuda.LongTensor)
            output_img = output_img.unsqueeze(1).type(LongTensor)
            img_sample = torch.cat((mask.data,output_img.data),0)
            save_image(img_sample,'%s/sample_img_val/val_%s_epoch_%s.png' % (args.path_save_results,epoch,batch_idx),nrow=args.test_batch_size,normalize=False)

    eval_val_loss[epoch] = eval_val_loss[epoch] / (batch_idx + 1)
    eval_val_dice[epoch] = eval_val_dice[epoch] / (batch_idx + 1)
    eval_val_precision[epoch] = eval_val_precision[epoch] / (batch_idx + 1)
    eval_val_sensitivity[epoch] = eval_val_sensitivity[epoch] / (batch_idx + 1)
    eval_val_specificity[epoch] = eval_val_specificity[epoch] / (batch_idx + 1)

    if train_accuracy:
        print('Train Epoch: {} done\tAvg Loss: {:.6f} Avg Dice: {:.6f} Avg Precision: {:.6f} Avg Sensitivity: {:.6f} Avg Specificity: {:.6f}'.format(
                epoch,
                eval_val_loss[epoch].item(),eval_val_dice[epoch].item(),eval_val_precision[epoch].item(),eval_val_sensitivity[epoch].item(), eval_val_specificity[epoch].item()))
    else:
        print('Test Epoch: {} done\tAvg Loss: {:.6f} Avg Dice: {:.6f} Avg Precision: {:.6f} Avg Sensitivity: {:.6f} Avg Specificity: {:.6f}'.format(
                epoch,
                eval_val_loss[epoch].item(),eval_val_dice[epoch].item(),eval_val_precision[epoch].item(),eval_val_sensitivity[epoch].item(), eval_val_specificity[epoch].item()))

##Training
if args.train:
    for i in range(args.epochs):
        train(i)
        torch.save(model.state_dict(),'bdclstm-{}'.format(i))
        #test(i)
    
    torch.save(model.state_dict(),
               'bdclstm-{}'.format(args.epochs))
    path_train_loss = 'train_loss'
    path_train_dice = 'train_dice'
    path_train_precision = 'train_precision'
    path_train_sensitivity = 'train_sensitivity'
    path_train_specificity = 'train_specificity'

    path_val_loss = 'val_loss'
    path_val_dice = 'val_dice'
    path_val_precision = 'val_precision'
    path_val_sensitivity = 'val_sensitivity'
    path_val_specificity = 'val_specificity'
    np.save(os.path.join(args.path_save_results,path_train_loss),eval_train_loss.cpu().detach().numpy())
    np.save(os.path.join(args.path_save_results,path_train_dice),eval_train_dice.cpu().numpy())
    np.save(os.path.join(args.path_save_results,path_train_precision),eval_train_precision.cpu().numpy())
    np.save(os.path.join(args.path_save_results,path_train_sensitivity),eval_train_sensitivity.cpu().numpy())
    np.save(os.path.join(args.path_save_results,path_train_specificity),eval_train_specificity.cpu().numpy())

    np.save(os.path.join(args.path_save_results,path_val_loss),eval_val_loss.cpu().detach().numpy())
    np.save(os.path.join(args.path_save_results,path_val_dice),eval_val_dice.cpu().numpy())
    np.save(os.path.join(args.path_save_results,path_val_precision),eval_val_precision.cpu().numpy())
    np.save(os.path.join(args.path_save_results,path_val_sensitivity),eval_val_sensitivity.cpu().numpy())
    np.save(os.path.join(args.path_save_results,path_val_specificity),eval_val_specificity.cpu().numpy())
else:
    model.load_state_dict(torch.load('bdclstm-{}'.format(args.epochs)))

    test(args.epochs,train_accuracy=True)
