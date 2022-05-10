
import torch
import torch.nn as nn
import torch.nn.functional as F

def minmax_ang(logit, logit_adv, label):
    

    
    cos_n = torch.gather(logit, 1, torch.unsqueeze(label, 1))
    
    cos_a = torch.gather(logit_adv, 1, torch.unsqueeze(label, 1))

    #cos_a = torch.max(cos_a, dim=1)[0]

    #cos_n = torch.max(cos_n, dim=1)[0]
    
    

    
    label_adv = torch.acos(cos_a)

    label_adv = label_adv.pow(2).mean()

    

    sine_nat = torch.sqrt(1.00001 - torch.pow(cos_n, 2))

    sine_adv = torch.sqrt(1.00001 - torch.pow(cos_a, 2))


    ang_diff = cos_a * cos_n  + sine_adv * sine_nat

    
    

    ang_ = ang_diff.clamp(-0.99999, 0.99999).mean()
    

    loss1 =  ang_



    loss2 = label_adv
     
    
        
    return loss1, loss2


