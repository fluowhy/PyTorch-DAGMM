# code based on https://github.com/danieltan07

import numpy as np
import argparse 
import torch
import pdb

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
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='learning rate')
    parser.add_argument('--lr_milestones', type=list, default=[50],
                        help='Milestones at which the scheduler multiply the lr by 0.1')
    parser.add_argument("--batch_size", type=int, default=2048, 
                        help="Batch size")
    parser.add_argument('--latent_dim', type=int, default=16,
                        help='Dimension of the latent variable z')
    parser.add_argument('--n_gmm', type=int, default=8,
                        help='Number of Gaussian components.')
    parser.add_argument('--lambda_gmm', type=float, default=1e-3,
                        help='Parameter lambda gmm in loss function.')
    parser.add_argument('--nin', type=int, default=3,
                        help='Input dimension.')
    parser.add_argument('--nout', type=int, default=1,
                        help='Output dimension.')
    parser.add_argument('--nh', type=int, default=96,
                        help='Hidden size.')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='Number of hidden layers.')
    parser.add_argument('--do', type=float, default=0.5,
                        help='Dropout.')
    parser.add_argument('--folded', action="store_true", help="Folded time series.")

    #parsing arguments.
    args = parser.parse_args()
    print(args)

    seed_everything()

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Get train and test dataloaders.
    data = get_asas_sn(args, folded=args.folded)

    DAGMM = TrainerDAGMM(args, data, device)
    DAGMM.train()
    # DAGMM.eval(DAGMM.model, data[1], device) # data[1]: test dataloader

    labels, scores_test, scores_train = evaluate(DAGMM.model, data, device, args)
