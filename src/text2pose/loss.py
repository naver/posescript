import torch
import torch.nn.functional as F
import numpy as np


def BBC(scores):
    # build the ground truth label tensor: the diagonal corresponds to the
    # correct classification
    GT_labels = torch.arange(scores.shape[0], device=scores.device).long()
    loss = F.cross_entropy(scores, GT_labels) # mean reduction
    return loss


def symBBC(scores):
    x2y_loss = BBC(scores)
    y2x_loss = BBC(scores.t())
    return (x2y_loss + y2x_loss) / 2.0


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