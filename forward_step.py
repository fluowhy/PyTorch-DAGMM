import torch
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import pdb


def recon_loss(x, x_hat, eps=1e-10):
    seq_len = (x[:, :, 0] != 0).sum(-1).float() + 1
    mask = (x[:, :, 0] != 0).type(torch.float)  # .to(self.device)
    mse = ((((x_hat - x[:, :, 1]) / (x[:, :, 2] + eps)).pow(2) * mask).sum(- 1) / seq_len).mean()
    return mse


class ComputeLoss:
    def __init__(self, model, lambda_gmm, device, n_gmm):
        self.model = model
        self.lambda_gmm = lambda_gmm
        self.device = device
        self.n_gmm = n_gmm
        self.cross_entropy = torch.nn.CrossEntropyLoss()
    
    def forward(self, x, y, x_hat, z, gamma):
        """Computing the loss function for DAGMM."""
        reconst_loss = recon_loss(x, x_hat)
        sample_energy, _ = self.compute_energy(z, gamma)
        cross_entropy = self.cross_entropy(gamma, y)
        loss = reconst_loss + self.lambda_gmm * sample_energy + cross_entropy
        return Variable(loss, requires_grad=True), reconst_loss, sample_energy, cross_entropy
    
    def compute_energy(self, z, gamma, phi=None, mu=None, cov=None, sample_mean=True):
        """Computing the sample energy function"""
        if (phi is None) or (mu is None) or (cov is None):
            phi, mu, cov = self.compute_params(z, gamma)

        z_mu = z.unsqueeze(1) - mu.unsqueeze(0)

        eps = 1e-12
        _, l, _ = cov.shape
        cte = l * np.log(2 * 3.1416)
        cov_inverse = torch.inverse(cov)
        det_cov = (0.5 * (cte + torch.logdet(cov))).exp()
        cov_diag = 0

        E_z = - 0.5 * torch.sum(torch.sum(z_mu.unsqueeze(-1) * cov_inverse.unsqueeze(0), dim=-2) * z_mu, dim=-1)
        E_z = torch.exp(E_z)
        E_z = torch.sum(phi.unsqueeze(0) * E_z / (torch.sqrt(det_cov)).unsqueeze(0), dim=1)
        E_z = - torch.log(E_z + eps)
        if sample_mean:
            E_z = torch.mean(E_z)
        return E_z, cov_diag

    def compute_params(self, z, gamma):
        """Computing the parameters phi, mu and gamma for sample energy function """ 
        # K: number of Gaussian mixture components
        # N: Number of samples
        # D: Latent dimension
        # z = NxD
        # gamma = NxK

        #phi = D
        phi = gamma.mean(dim=0)
        gamma_sum = gamma.sum(dim=0, keepdim=True)

        #mu = KxD
        mu = torch.sum(z.unsqueeze(-1) * gamma.unsqueeze(1), dim=0) / gamma_sum

        z_mu = z.unsqueeze(-1) - mu.unsqueeze(0)
        z_mu_z_mu_t = torch.matmul(z_mu.transpose(0, 2), z_mu.transpose(0, 2).transpose(1, 2)).squeeze(0)
        
        #cov = K x D x D
        cov = torch.sum(gamma.unsqueeze(-1).unsqueeze(-1) * z_mu_z_mu_t, dim=0)
        cov = cov / gamma_sum.squeeze().unsqueeze(-1).unsqueeze(-1)

        return phi, mu.transpose(0, 1), cov
        

class Cholesky(torch.autograd.Function):
    def forward(ctx, a):
        l = torch.cholesky(a, False)
        ctx.save_for_backward(l)
        return l
    def backward(ctx, grad_output):
        l, = ctx.saved_variables
        linv = l.inverse()
        inner = torch.tril(torch.mm(l.t(), grad_output)) * torch.tril(
            1.0 - Variable(l.data.new(l.size(1)).fill_(0.5).diag()))
        s = torch.mm(linv.t(), torch.mm(inner, linv))
        return s
