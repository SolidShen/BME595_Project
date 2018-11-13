import torch
import numpy as np
import torch.nn.functional as F
def dice_loss(input,target,num_of_classes):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    '''
    -------------
    target resize
    -------------
    '''
    
    '''
    smooth = 1.
def dice_loss(y_pred, y_true):
    product = nd.multiply(y_pred, y_true)
    intersection = nd.sum(product)
    coefficient = (2.*intersection +smooth) / (nd.sum(y_pred)+nd.sum(y_true) +smooth)
    loss = 1. - coefficient
    # or "-coefficient"
    return(loss)
    '''
    
    batch_size,H,W = target.size()
    target_one_hot = torch.zeros(batch_size,num_of_classes,H,W)
    for i in range(batch_size):
        for j in range(H):
            for k in range(W):
                if target[i][j][k] == 0:
                    target_one_hot[i][0][j][k] = 1
                elif target[i][j][k] == 1:
                    target_one_hot[i][1][j][k] = 1
    
        
    
    assert input.size() == target_one_hot.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    uniques=np.unique(target_one_hot.numpy())
    assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"

    probs=F.softmax(input)
    num=probs*target_one_hot#b,c,h,w--p*g
    num=torch.sum(num,dim=3)#b,c,h
    num=torch.sum(num,dim=2)
    

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=3)#b,c,h
    den1=torch.sum(den1,dim=2)
    

    den2=target_one_hot*target_one_hot#--g^2
    den2=torch.sum(den2,dim=3)#b,c,h
    den2=torch.sum(den2,dim=2)#b,c
    

    dice=2*(num/(den1+den2))
    dice_eso=dice[:,1:]#we ignore bg dice val, and take the fg

    dice_total=-1*torch.sum(dice_eso)/dice_eso.size(0)#divide by batch_sz

    return dice_total

