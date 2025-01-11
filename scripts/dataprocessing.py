import numpy as np
import random
import pandas as pd

from sklearn.model_selection import StratifiedKFold
import os


import torch
from torch.utils.data import Dataset


def make_imbalance(data, rate, taskclass):
    idx = []
    categories = []

    j = 0
    rk = 0
    for i in taskclass:
        index_list = [a for a, b in enumerate(data[:, -1]) if b == i]
        lens = int(len(index_list) * rate[rk])

        idx = idx + random.sample(index_list, lens)

        categories = categories + [j] * lens
        rk += 1
        j = j + 1

    return data[idx,:-1], categories

def make_dir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

def Default_processing(data_name):
    # import and normalize data
    datapath = f'./datasets/{data_name}/{data_name}.xlsx'
    if os.path.exists(datapath):
        data = pd.read_excel(datapath).values.astype(float)
    else:
        datapath = f'./datasets/{data_name}/{data_name}.csv'
        data = pd.read_csv(datapath).values.astype(float)

    savepath = f'./datasets/{data_name}'
    x = data[:,:-1]
    y = data[:,-1]


    # make path
    dir_num = ['DGOT', 'CLASSIC', 'CTGAN', 'SOS', 'TEST']
    path_dir = []
    for n, dirs in enumerate(dir_num):
        path_dir.append(os.path.join(savepath, dirs))
        make_dir(path_dir[n])


    # 5-fold cross validation data partitioning
    skfolds = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)

    dmax = np.max(x, axis=0)
    dmin = np.min(x, axis=0)
    x_norm = ((x - dmin) / (dmax - dmin + 1e-8)) * 2 - 1

    k = 0
    for train_idx, test_idx in skfolds.split(x, y):
        path_dir_exp = []
        for n, dirs in enumerate(path_dir):
            path_dir_exp.append(os.path.join(dirs, f'exp{k}'))
            make_dir(path_dir_exp[n])

        X_train_folds = x[train_idx]
        y_train_folds = y[train_idx]
        X_test_fold = x[test_idx]
        y_test_fold = y[test_idx]

        # normalization by the maximum and minimum value of the training set
        X_norm_train_folds = x_norm[train_idx]
        X_norm_test_folds = x_norm[test_idx]

        # DGOT
        np.save(os.path.join(path_dir_exp[0], 'xtrain.npy'), X_norm_train_folds[:,None,:])
        np.save(os.path.join(path_dir_exp[0], 'ytrain.npy'), y_train_folds)

        # CLASSIC
        np.save(os.path.join(path_dir_exp[1], 'xtrain.npy'), X_train_folds)
        np.save(os.path.join(path_dir_exp[1], 'ytrain.npy'), y_train_folds)

        # CTGAN
        X_train_folds_df = pd.DataFrame(data[train_idx])
        X_train_folds_df.to_csv(os.path.join(path_dir_exp[2], 'train_df.csv'), index = False)

        # SOS
        np.savez(os.path.join(path_dir_exp[3], f'{data_name}.npz'), train=data[train_idx], test=data[test_idx])

        # TEST
        np.save(os.path.join(path_dir_exp[4], 'xtest.npy'), X_norm_test_folds[:, :])
        np.save(os.path.join(path_dir_exp[4], 'ytest.npy'), y_test_fold)
        k=k+1


class datasets(Dataset):
    def __init__(self, path):
        # import training data
        x = np.load(os.path.join(path,'xtrain.npy'))
        y = np.load(os.path.join(path,'ytrain.npy'))

        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.int32)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        data = self.x[index, :, :]
        label = self.y[index]

        return data, label

