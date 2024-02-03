import torch
from torch.nn import functional as F
import numpy as np

def reconstruction_loss(recon_x, x):
    recons_loss = F.mse_loss(recon_x, x, reduction="sum")
    return recons_loss

def KL_loss(mu, logvar):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return KLD

def softclip(tensor, min):
    """ Clips the tensor values at the minimum value min in a softway. Taken from Handful of Trials """
    result_tensor = min + F.softplus(tensor - min)

    return result_tensor

def gaussian_nll(mu, log_sigma, x):
    return 0.5 * torch.pow((x - mu) / log_sigma.exp(), 2) + log_sigma + 0.5 * np.log(2 * np.pi)

def sigma_reconstruction_loss(x_hat, x):
    """ Computes the likelihood of the data given the latent variable """

    log_sigma = ((x - x_hat) ** 2).mean([0,1,2,3,4], keepdim=True).sqrt()#.log()

    # Learning the variance can become unstable in some cases. Softly limiting log_sigma to a minimum of -6
    # ensures stable training.
    log_sigma = softclip(log_sigma, -6)
    
    rec = gaussian_nll(x_hat, log_sigma, x).sum() 

    return rec, log_sigma

def mlp_loss_function(y, out_mlp, weights=False):
    if type(weights) != bool:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum', pos_weight=weights)
    else:
        criterion = torch.nn.BCEWithLogitsLoss(reduction='sum')
    batch_size = out_mlp.shape[0]
    mean_loss = criterion( out_mlp.type(torch.float),  y.type(torch.float) ) / batch_size
    
    return mean_loss

