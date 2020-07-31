import torch

def triplet_loss(anchor, positive, negative, alpha = 0.4, device='cuda:0'):
    """
    Implementation of the triplet loss function
    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    embedding -- python list containing three objects:
            anchor -- the encodings for the anchor data
            positive -- the encodings for the positive data (similar to anchor)
            negative -- the encodings for the negative data (different from anchor)
    Returns:
    loss -- real number, value of the loss
    """
#     print('embedding.shape = ',embedding)
    
#     print('total_lenght=',  total_lenght)
#     total_lenght =12
    
#     anchor = embedding[:,0,:]
#     positive = embedding[:,1,:]
#     negative = embedding[:,2,:]

    # distance between the anchor and the positive
    pos_dist = torch.sum((anchor-positive).pow(2),axis=1)

    # distance between the anchor and the negative
    neg_dist = torch.sum((anchor-negative).pow(2),axis=1)

    # compute loss
    basic_loss = pos_dist-neg_dist+alpha
    loss = torch.max(basic_loss, torch.tensor([0], device=device).float())
 
    return loss