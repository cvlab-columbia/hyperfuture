import numpy as np
import torch


def poincare_distance(pred, gt):
    '''
    Calculate pair-wise poincare distance between each row in two input tensors
    
    See equation (1) in this paper for mathematical expression:
    https://arxiv.org/abs/1705.08039
    '''
    (N_pred, D) = pred.shape
    (N_gt, D) = gt.shape
    a = (1 - square_norm(pred)).view(N_pred, 1)
    b = (1 - square_norm(gt)).view(1, N_gt)
    return torch.acosh(1 + 2 * pairwise_distances(pred, gt) / torch.matmul(a, b))


def square_norm(x):
    """
    Helper function returning square of the euclidean norm.
    Also here we clamp it since it really likes to die to zero.
    """
    norm = torch.norm(x, dim=-1, p=2) ** 2
    return torch.clamp(norm, min=1e-5)


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)8
    return torch.clamp(dist, 1e-7, np.inf)
