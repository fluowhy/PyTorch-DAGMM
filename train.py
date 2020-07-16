import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import numpy as np
from barbar import Bar
import pdb
import os
import shutil

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
        log_path = "logs/dagmmts"
        if os.path.exists(log_path) and os.path.isdir(log_path):
            shutil.rmtree(log_path)
        self.writer = SummaryWriter(log_path)

    def train(self):
        """Training the DAGMM model"""
        self.model = DAGMMTS(
            self.args.nin,
            self.args.nh,
            self.args.nout,
            self.args.nlayers,
            self.args.do,
            self.args.n_gmm,
            self.args.latent_dim,
            self.args.folded
            ).to(self.device)
        self.model.apply(weights_init_normal)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)

        self.compute = ComputeLoss(self.model, self.args.lambda_gmm,
            self.device, self.args.n_gmm)
        self.model.train()
        for epoch in range(self.args.num_epochs):
            total_loss = 0
            recon_loss = 0
            gmm_loss = 0
            ce_loss = 0       
            for batch in Bar(self.train_loader):
                optimizer.zero_grad()
                x = batch[0].float().to(self.device)
                y = batch[1].long().to(self.device)
                m = batch[2].float().to(self.device)
                s = batch[3].float().to(self.device)
                if self.args.folded:
                    p = batch[4].float().to(self.device)
                    _, x_hat, z, gamma = self.model(x, m, s, p)
                else:
                    _, x_hat, z, gamma = self.model(x, m, s)

                loss, reconst_loss, sample_energy, cross_entropy = self.compute.forward(x, y, x_hat, z, gamma)
                loss.backward(retain_graph=True)
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)
                optimizer.step()
                total_loss += loss.item()
                recon_loss += reconst_loss.item()
                gmm_loss += sample_energy.item()
                ce_loss += cross_entropy.item()

            total_loss = total_loss / len(self.train_loader)
            recon_loss = recon_loss / len(self.train_loader)
            gmm_loss = gmm_loss / len(self.train_loader)
            ce_loss = ce_loss / len(self.train_loader)

            total_loss_val = 0
            recon_loss_val = 0
            gmm_loss_val = 0
            ce_loss_val = 0
            
            print('Training DAGMMTS... Epoch: {}, Loss: {:.3f}'.format(epoch, total_loss))
            self.writer.add_scalars("total", {"train": total_loss, "val": total_loss_val}, global_step=epoch)
            self.writer.add_scalars("recon", {"train": recon_loss, "val": recon_loss_val}, global_step=epoch)
            self.writer.add_scalars("gmm", {"train": gmm_loss, "val": gmm_loss_val}, global_step=epoch)
            self.writer.add_scalars("cross_entropy", {"train": ce_loss, "val": ce_loss_val}, global_step=epoch)
        return
