import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb


class DAGMM(nn.Module):
    def __init__(self, n_gmm=2, z_dim=1):
        """Network for DAGMM (KDDCup99)"""
        super(DAGMM, self).__init__()
        #Encoder network
        self.fc1 = nn.Linear(118, 60)
        self.fc2 = nn.Linear(60, 30)
        self.fc3 = nn.Linear(30, 10)
        self.fc4 = nn.Linear(10, z_dim)

        #Decoder network
        self.fc5 = nn.Linear(z_dim, 10)
        self.fc6 = nn.Linear(10, 30)
        self.fc7 = nn.Linear(30, 60)
        self.fc8 = nn.Linear(60, 118)

        #Estimation network
        self.fc9 = nn.Linear(z_dim+2, 10)
        self.fc10 = nn.Linear(10, n_gmm)

    def encode(self, x):
        h = torch.tanh(self.fc1(x))
        h = torch.tanh(self.fc2(h))
        h = torch.tanh(self.fc3(h))
        return self.fc4(h)

    def decode(self, x):
        h = torch.tanh(self.fc5(x))
        h = torch.tanh(self.fc6(h))
        h = torch.tanh(self.fc7(h))
        return self.fc8(h)
    
    def estimate(self, z):
        h = F.dropout(torch.tanh(self.fc9(z)), 0.5)
        return F.softmax(self.fc10(h), dim=1)
    
    def compute_reconstruction(self, x, x_hat):
        relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x, x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity
    
    def forward(self, x):
        z_c = self.encode(x)
        x_hat = self.decode(z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat)
        z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1)], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma


class DAGMMTS(nn.Module):
    def __init__(self, nin, nh, nout, nlayers, do, n_gmm=2, z_dim=1, folded=False):
        """Network for DAGMMTS (KDDCup99)"""
        super(DAGMMTS, self).__init__()
        self.nh = nh
        self.z_dim = z_dim
        if nlayers >= 2:
            self.enc = torch.nn.LSTM(input_size=nin, hidden_size=nh, num_layers=nlayers, dropout=do, batch_first=True)
            self.dec = torch.nn.LSTM(input_size=z_dim + 1, hidden_size=nh, num_layers=nlayers, dropout=do, batch_first=True)
        else:
            self.enc = torch.nn.LSTM(input_size=nin, hidden_size=nh, num_layers=nlayers, batch_first=True)
            self.dec = torch.nn.LSTM(input_size=z_dim + 1, hidden_size=nh, num_layers=nlayers, batch_first=True)
        self.fcd = torch.nn.Linear(nh, nout)
        self.fce = torch.nn.Linear(nh, z_dim)
        self.do = torch.nn.Dropout(p=do)

        #Estimation network
        self.fc9 = nn.Linear(z_dim + 5, 16) if folded else nn.Linear(z_dim + 4, 16)
        self.fc10 = nn.Linear(16, n_gmm)

    def encode(self, x):
        # input (batch, seq_len, input_size)
        # output  (batch, seq_len, num_directions * hidden_size)
        x, (_, _) = self.enc(x)
        x = self.fce(x)
        return x

    def decode(self, dt, z):
        n, l = dt.shape
        z = self.do(z)
        x_lat = torch.zeros((n, l, self.z_dim + 1)).to(dt.device)
        new_z = z.view(-1, self.z_dim, 1).expand(-1, -1, l).transpose(1, 2)
        x_lat[:, :, :-1] = new_z
        x_lat[:, :, -1] = dt
        output, (_, _) = self.dec(x_lat)  # input shape (seq_len, batch, features)
        output = self.fcd(output).squeeze()
        return output.squeeze()
    
    def estimate(self, z):
        h = F.dropout(torch.tanh(self.fc9(z)), 0.5)
        return F.softmax(self.fc10(h), dim=1)
    
    def compute_reconstruction(self, x, x_hat, seq_len, eps=1e-10):
        # mask = (x[:, :, 2] != 0).type(torch.float)  # .to(self.device)
        # relative_euclidean_distance = ((((x_hat - x[:, :, 1]) / (x[:, :, 2] + eps)).pow(2) * mask).sum(- 1) / seq_len)
        mask = (x[:, :, 0] != 0).type(torch.float)  # .to(self.device)
        relative_euclidean_distance = ((x_hat - x[:, :, 1]).pow(2) * mask).sum(- 1) / x[:, :, 1].norm(2, dim=1)
        # relative_euclidean_distance = (x-x_hat).norm(2, dim=1) / x.norm(2, dim=1)
        cosine_similarity = F.cosine_similarity(x[:, :, 1], x_hat, dim=1)
        return relative_euclidean_distance, cosine_similarity
    
    def forward(self, x, m, s, p=None):
        seq_len = (x[:, :, 0] != 0).sum(-1)
        n, _, _ = x.shape
        z_c = self.encode(x)
        z_c = z_c[torch.arange(n), (seq_len - 1).type(dtype=torch.long)]
        x_hat = self.decode(x[:, :, 0], z_c)
        rec_1, rec_2 = self.compute_reconstruction(x, x_hat, seq_len)
        if p is None:
            z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1), m, s], dim=1)
        else:
            z = torch.cat([z_c, rec_1.unsqueeze(-1), rec_2.unsqueeze(-1), m, s, p], dim=1)
        gamma = self.estimate(z)
        return z_c, x_hat, z, gamma
