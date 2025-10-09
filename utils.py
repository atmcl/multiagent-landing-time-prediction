import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class NegativeLogLikelihood(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.register_buffer('log_2pi', torch.log(torch.tensor(2 * torch.pi)))  

    def forward(self, mu, sigma, target):
        if mu.dim() == 3: mu = mu.squeeze(-1) if mu.shape[2] == 1 else mu.squeeze(1)
        if sigma.dim() == 3: sigma = sigma.squeeze(-1) if sigma.shape[2] == 1 else sigma.squeeze(1)
        if target.dim() == 3: target = target.squeeze(-1) if target.shape[2] == 1 else target.squeeze(1)

        sigma = sigma + self.eps  

        log_term = 2 * torch.log(sigma) + self.log_2pi
        sq_term = ((target - mu) ** 2) / (2 * sigma ** 2)

        nll = 0.5 * log_term + sq_term

        return nll.mean()
