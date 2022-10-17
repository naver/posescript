import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def BBC(scores):
    # build the ground truth label tensor: the diagonal corresponds to the
    # correct classification
    batch_size = scores.shape[0]
    GT_labels = torch.arange(batch_size).long()
    GT_labels = torch.autograd.Variable(GT_labels)
    if torch.cuda.is_available():
        GT_labels = GT_labels.cuda()
    
    loss = F.cross_entropy(scores, GT_labels) # mean reduction

    return loss


def laplacian_nll(x_tilde, x, log_sigma):
    """ Negative log likelihood of an isotropic Laplacian density """
    log_norm = - (np.log(2) + log_sigma)
    log_energy = - (torch.abs(x_tilde - x)) / torch.exp(log_sigma)
    return - (log_norm + log_energy)


def gaussian_nll(x_tilde, x, log_sigma):
    """ Negative log-likelihood of an isotropic Gaussian density """
    log_norm = - 0.5 * (np.log(2 * np.pi) + log_sigma)
    log_energy = - 0.5 * F.mse_loss(x_tilde, x, reduction='none') / torch.exp(log_sigma)
    return - (log_norm + log_energy)
        
        