import torch
def dice_score(est_label,label):
        #est_label: N,C,H,W
        #label: N,H,W
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

        return dice
    