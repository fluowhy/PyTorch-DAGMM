import torch
import numpy as np
import pdb

from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn import metrics
import matplotlib.pyplot as plt
import pandas as pd

from forward_step import ComputeLoss


def metrics_summary(scores_test, scores_train, y_true, percentile):
    thr = np.percentile(scores_train, percentile)
    y_pred = np.ones(len(scores_test))
    y_pred[scores_test < thr] = 0
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    return precision, recall, f1


def evaluate(model, dataloaders, device, args):
    """Testing the DAGMM model"""
    dataloader_train, dataloader_test = dataloaders
    model.eval()
    print('Testing...')
    compute = ComputeLoss(model, None, device, args.n_gmm)
    with torch.no_grad():
        N_samples = 0
        gamma_sum = 0
        mu_sum = 0
        cov_sum = 0
        # Obtaining the parameters gamma, mu and cov using the train (clean) data.
        for batch in dataloader_train:
            x = batch[0].float().to(device)
            y = batch[1].int().to(device)
            m = batch[2].float().to(device)
            s = batch[3].float().to(device)
            if args.folded:
                p = batch[4].float().to(device)
                _, _, z, gamma = model(x, m, s, p)
            else:
                _, _, z, gamma = model(x, m, s)

            
            phi_batch, mu_batch, cov_batch = compute.compute_params(z, gamma)

            batch_gamma_sum = torch.sum(gamma, dim=0)
            gamma_sum += batch_gamma_sum
            mu_sum += mu_batch * batch_gamma_sum.unsqueeze(-1)
            cov_sum += cov_batch * batch_gamma_sum.unsqueeze(-1).unsqueeze(-1)
            
            N_samples += x.size(0)
            
        train_phi = gamma_sum / N_samples
        train_mu = mu_sum / gamma_sum.unsqueeze(-1)
        train_cov = cov_sum / gamma_sum.unsqueeze(-1).unsqueeze(-1)

        # Obtaining Labels and energy scores for train data
        energy_train = []
        labels_train = []
        for batch in dataloader_train:
            x = batch[0].float().to(device)
            y = batch[1].int().to(device)
            m = batch[2].float().to(device)
            s = batch[3].float().to(device)
            if args.folded:
                p = batch[4].float().to(device)
                _, _, z, gamma = model(x, m, s, p)
            else:
                _, _, z, gamma = model(x, m, s)

            sample_energy, cov_diag  = compute.compute_energy(z, gamma, phi=train_phi,
                                                              mu=train_mu, cov=train_cov, 
                                                              sample_mean=False)

            energy_train.append(sample_energy.detach().cpu())
            labels_train.append(y)
        energy_train = torch.cat(energy_train).cpu().numpy()
        labels_train = torch.cat(labels_train).cpu().numpy()

        # Obtaining Labels and energy scores for test data
        energy_test = []
        labels_test = []
        for batch in dataloader_test:
            x = batch[0].float().to(device)
            y = batch[1].int().to(device)
            m = batch[2].float().to(device)
            s = batch[3].float().to(device)
            if args.folded:
                p = batch[4].float().to(device)
                _, _, z, gamma = model(x, m, s, p)
            else:
                _, _, z, gamma = model(x, m, s)

            sample_energy, cov_diag  = compute.compute_energy(z, gamma, train_phi,
                                                              train_mu, train_cov,
                                                              sample_mean=False)
            
            energy_test.append(sample_energy.detach().cpu())
            labels_test.append(y)
        energy_test = torch.cat(energy_test).cpu().numpy()
        labels_test = torch.cat(labels_test).cpu().numpy()
    
        scores_total = np.concatenate((energy_train, energy_test), axis=0)
        labels_total = np.concatenate((labels_train, labels_test), axis=0)

    labels = np.zeros(len(labels_test))
    labels[labels_test == 8] = 1  # class 8 is outlier

    scores_in = energy_test[labels == 0]
    scores_out = energy_test[labels == 1]
    average_precision = (labels == 1).sum() / len(labels)

    score_min = energy_test.min()
    score_max = energy_test.max()
    n_bins = 100
    bins = np.linspace(score_min, score_max, n_bins)

    pr95, re95, f195 = metrics_summary(energy_test, energy_train, labels, 95)
    pr80, re80, f180 = metrics_summary(energy_test, energy_train, labels, 80)

    precision, recall, _ = precision_recall_curve(labels, energy_test, pos_label=1)
    fpr, tpr, _ = metrics.roc_curve(labels, energy_test, pos_label=1)

    aucroc = metrics.auc(fpr, tpr)
    aucpr = metrics.auc(recall, precision)

    df = pd.DataFrame()
    df["percentile"] = [95, 80]
    df["precision"] = [pr95, pr80]
    df["recall"] = [re95, re80]
    df["f1"] = [f195, f180]
    df["aucroc"] = [aucroc, aucroc]
    df["aucpr"] = [aucpr, aucpr]
    print(df)
    df.to_csv("summary.csv", index=False)

    plt.clf()
    plt.hist(scores_in, bins=bins, color="navy", alpha=0.5, label="inlier")
    plt.hist(scores_out, bins=bins, color="red", alpha=0.5, label="outlier")
    plt.grid()
    plt.yscale("log")
    plt.legend()
    plt.xlabel("energy")
    plt.ylabel("counts")
    plt.tight_layout()
    plt.savefig("scores.png", dpi=200)

    plt.clf()
    plt.title("AUCPR: {:.4f}".format(aucpr))
    plt.plot(recall, precision, color="red")
    plt.axhline(average_precision, color="black", linestyle="--")
    plt.grid()
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig("precision_recall_curve.png", dpi=200)

    plt.clf()
    plt.title("AUCROC: {:.4f}".format(aucroc))
    plt.plot(fpr, tpr, color="red")
    aux = np.linspace(0, 1, 10)
    plt.plot(aux, aux, color="black", linestyle="--")
    plt.grid()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig("roc_curve.png", dpi=200)
    return labels, energy_test, energy_train
