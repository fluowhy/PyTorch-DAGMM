import torch
from torch import optim
import torch.nn.functional as F

import numpy as np
from barbar import Bar
import pdb

from model import DAGMMTS
from forward_step import ComputeLoss
from utils.utils import weights_init_normal
from test import evaluate

class TrainerDAGMM:
    """Trainer class for DAGMM."""
    def __init__(self, args, data, device):
        self.args = args
        self.train_loader, self.test_loader = data
        self.device = device


    def train(self):
        """Training the DAGMM model"""
        self.model = DAGMMTS(
            self.args.nin,
            self.args.nh,
            self.args.nout,
            self.args.nlayers,
            self.args.do,
            self.args.n_gmm,
            self.args.latent_dim
            ).to(self.device)
        self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.compute = ComputeLoss(self.model, self.args.lambda_energy, self.args.lambda_cov, 
                                   self.device, self.args.n_gmm)
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            for x, _, m, s in Bar(self.train_loader):
                x = x.float().to(self.device)
                m = m.float().to(self.device)
                s = s.float().to(self.device)
                optimizer.zero_grad()
                
                _, x_hat, z, gamma = self.model(x, m, s)

                loss = self.compute.forward(x, x_hat, z, gamma)
                loss.backward(retain_graph=True)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()

                total_loss += loss.item()
            print('Training DAGMMTS... Epoch: {}, Loss: {:.3f}'.format(
                   epoch, total_loss / len(self.train_loader)))
                

        

