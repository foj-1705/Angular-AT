import torch
import torch.nn.functional as F
#import mxnet as mx

#https://github.com/wznpub/MMA_Regularization/blob/main/MMA.py
def get_mma_loss(weight):
    

    
    if weight.dim() > 2:
        weight = weight.view(weight.size(0), -1)

    # computing cosine similarity: dot product of normalized weight vectors
    weight_ = F.normalize(weight, p=2, dim=1)
    cosine = torch.matmul(weight_, weight_.t())

    # make sure that the diagnonal elements cannot be selected
    cosine = cosine - 2. * torch.diag(torch.diag(cosine))

    # maxmize the minimum angle
    loss = -torch.acos(cosine.max(dim=1)[0].clamp(-0.99999, 0.99999)).mean()

    #loss = cosine.max(dim=1)[0].clamp(-0.99999, 0.99999).mean()
    return loss



