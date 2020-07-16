import torch
import numpy as np
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import pickle as pl
import pdb


class KDDCupData:
    def __init__(self, data_dir, mode):
        """Loading the data for train and test."""
        data = np.load(data_dir, allow_pickle=True)

        labels = data["kdd"][:,-1]
        features = data["kdd"][:,:-1]
        #In this case, "atack" has been treated as normal data as is mentioned in the paper
        normal_data = features[labels==0] 
        normal_labels = labels[labels==0]

        n_train = int(normal_data.shape[0]*0.5)
        ixs = np.arange(normal_data.shape[0])
        np.random.shuffle(ixs)
        normal_data_test = normal_data[ixs[n_train:]]
        normal_labels_test = normal_labels[ixs[n_train:]]

        if mode == 'train':
            self.x = normal_data[ixs[:n_train]]
            self.y = normal_labels[ixs[:n_train]]
        elif mode == 'test':
            anomalous_data = features[labels==1]
            anomalous_labels = labels[labels==1]
            self.x = np.concatenate((anomalous_data, normal_data_test), axis=0)
            self.y = np.concatenate((anomalous_labels, normal_labels_test), axis=0)

    def __len__(self):
        """Number of images in the object dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        return np.float32(self.x[index]), np.float32(self.y[index])


class ToyDataset:
    def __init__(self, data_dir, mode, eps=1e-10):
        """Loading the data for train and test."""
        self.x = np.load("{}/x_{}_nsamples_32_nclasses_5.npy".format(data_dir, mode))
        y = np.load("{}/y_{}_nsamples_32_nclasses_5.npy".format(data_dir, mode))

        # standard normalization
        mean = self.x[:, :, 1].mean(axis=1)[:, np.newaxis]
        std = self.x[:, :, 1].std(axis=1)[:, np.newaxis]
        self.x[:, :, 1] = (self.x[:, :, 1] - mean) / (std + eps)

        # remove outlier data from training
        # inlier labels: 0, 1, 2
        # outlier labels: 3, 4
        if mode == "train":
            inlier_mask = (y == 0) | (y == 1) | (y == 2)
            self.x = self.x[inlier_mask]
            y = y[inlier_mask]
        # rename inlier/outlier labels
        self.y = np.zeros(len(y))            
        if mode == "test":
            outlier_mask = (y == 3) | (y == 4)
            self.y[outlier_mask] = 1
        self.m = mean.astype(np.float32)
        self.s = std.astype(np.float32)

    def __len__(self):
        """Number of images in the object dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        return self.x[index], self.y[index], self.m[index], self.s[index]


class ASASSN:
    def __init__(self, data_dir, mode, eps=1e-10):
        """Loading the data for train and test."""
        self.x = np.load("{}/x_{}.npy".format(data_dir, mode))
        self.y = np.load("{}/y_{}.npy".format(data_dir, mode))

        # standard normalization
        mean = self.x[:, :, 1].mean(axis=1)[:, np.newaxis]
        std = self.x[:, :, 1].std(axis=1)[:, np.newaxis]
        self.x[:, :, 1] = (self.x[:, :, 1] - mean) / (std + eps)
        self.x[:, :, 2] = self.x[:, :, 2] / (std + eps)

        # time normalization
        self.x[:, :, 0] = self.x[:, :, 0] - self.x[:, 0, 0][:, np.newaxis]
        self.x[:, 1:, 0] = self.x[:, 1:, 0] - self.x[:, :-1, 0]

        self.m = mean.astype(np.float32)
        self.s = std.astype(np.float32)

    def __len__(self):
        """Number of images in the object dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        return self.x[index], self.y[index], self.m[index], self.s[index]


class FoldedASASSN:
    def __init__(self, data_dir, mode, eps=1e-10):
        """Loading the data for train and test."""
        self.x = np.load("{}/pf_{}.npy".format(data_dir, mode))
        self.y = np.load("{}/y_{}.npy".format(data_dir, mode))
        phase = np.load("{}/p_{}.npy".format(data_dir, mode))

        # standard normalization
        mean = self.x[:, :, 1].mean(axis=1)[:, np.newaxis]
        std = self.x[:, :, 1].std(axis=1)[:, np.newaxis]
        self.x[:, :, 1] = (self.x[:, :, 1] - mean) / (std + eps)
        self.x[:, :, 2] = self.x[:, :, 2] / (std + eps)

        # time normalization
        self.x[:, :, 0] = self.x[:, :, 0] - self.x[:, 0, 0][:, np.newaxis]
        self.x[:, 1:, 0] = self.x[:, 1:, 0] - self.x[:, :-1, 0]

        self.m = mean.astype(np.float32)
        self.s = std.astype(np.float32)

        # phase processing
        self.p = np.log10(phase)[:, np.newaxis]

    def __len__(self):
        """Number of images in the object dataset."""
        return self.x.shape[0]

    def __getitem__(self, index):
        """Return a sample from the dataset."""
        return self.x[index], self.y[index], self.m[index], self.s[index], self.p[index]



def get_KDDCup99(args, data_dir='./data/kdd_cup.npz'):
    """Returning train and test dataloaders."""
    train = KDDCupData(data_dir, 'train')
    dataloader_train = DataLoader(train, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    
    test = KDDCupData(data_dir, 'test')
    dataloader_test = DataLoader(test, batch_size=args.batch_size, 
                              shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test


def get_synthetic_time_series(args, data_dir="../datasets/toy"):
    train = ToyDataset(data_dir, "train")
    dataloader_train = DataLoader(train, batch_size=args.batch_size, 
                              shuffle=True, num_workers=0)
    
    test = ToyDataset(data_dir, "test")
    dataloader_test = DataLoader(test, batch_size=args.batch_size, 
                              shuffle=False, num_workers=0)
    return dataloader_train, dataloader_test


def get_asas_sn(args, data_dir="../datasets/asas_sn", folded=False):
    if folded:
        train = FoldedASASSN(data_dir, "train")
        test = FoldedASASSN(data_dir, "test")
    else:
        train = ASASSN(data_dir, "train")
        test = ASASSN(data_dir, "test")

    dataloader_train = DataLoader(
        train,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
        )

    dataloader_test = DataLoader(
        test,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
        )

    return dataloader_train, dataloader_test


if __name__ == "__main__":
    class Args:
        batch_size = 32

    args = Args()
    get_asas_sn(args, folded=True)
    