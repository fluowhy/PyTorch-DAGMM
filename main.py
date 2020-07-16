# code based on https://github.com/danieltan07

import numpy as np
import argparse 
import torch

from train import TrainerDAGMM
from test import evaluate
from preprocess import get_KDDCup99, get_synthetic_time_series, get_asas_sn
from utils.utils import seed_everything


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=200,
                        help="number of epochs")
    parser.add_argument("--patience", type=int, default=50, 
                        help="Patience for Early Stopping")
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='learning rate')
    parser.add_argument('--lr_milestones', type=list, default=[50],
                        help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument("--batch_size", type=int, default=1024, 
                        help="Batch size")
    parser.add_argument('--latent_dim', type=int, default=1,
                        help='Dimension of the latent variable z')
    parser.add_argument('--n_gmm', type=int, default=4,
                        help='Number of Gaussian components ')
    parser.add_argument('--lambda_energy', type=float, default=0.1,
                        help='Parameter labda1 for the relative importance of sampling energy.')
    parser.add_argument('--lambda_cov', type=int, default=0.005,
                        help='Parameter lambda2 for penalizing small values on'
                             'the diagonal of the covariance matrix')
    parser.add_argument('--nin', type=int, default=2,
                        help='Input dimension.')
    parser.add_argument('--nout', type=int, default=1,
                        help='Output dimension.')
    parser.add_argument('--nh', type=int, default=1,
                        help='Hidden size.')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='Number of hidden layers.')
    parser.add_argument('--do', type=float, default=0.5,
                        help='Dropout.')

    #parsing arguments.
    args = parser.parse_args()

    seed_everything()

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get train and test dataloaders.
    data = get_asas_sn(args, folded=True)

    DAGMM = TrainerDAGMM(args, data, device)
    DAGMM.train()
    # DAGMM.eval(DAGMM.model, data[1], device) # data[1]: test dataloader

    labels, scores = evaluate(DAGMM.model, data, device, args.n_gmm)

    import matplotlib.pyplot as plt
    import pandas as pd 

    scores_in = scores[np.where(labels==0)[0]]
    scores_out = scores[np.where(labels==1)[0]]


    in_ = pd.DataFrame(scores_in, columns=['Inlier'])
    out_ = pd.DataFrame(scores_out, columns=['Outlier'])

    fig, ax = plt.subplots()
    in_.plot.kde(ax=ax, legend=True, title='Outliers vs Inliers (Deep SVDD)')
    out_.plot.kde(ax=ax, legend=True)
    ax.grid(axis='x')
    ax.grid(axis='y')
    plt.savefig("scores.png", dpi=200)