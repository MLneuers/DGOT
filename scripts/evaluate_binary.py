import yaml
import argparse
from tqdm.auto import tqdm
from collections import Counter

import numpy as np
import pandas as pd
import os

import torch
from models.GaussionDiffusion import sample_posterior

from sklearn.metrics import accuracy_score
import sklearn.metrics as metrics



def exists(x):
    return x is not None

def configs_read(configs_file):
    with open(configs_file) as f:
        configs_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    configs = argparse.Namespace(**configs_dict)

    return configs

def indicator_cls(labels, predicted):
    accuracy = accuracy_score(labels, predicted)
    macro_f1 = metrics.f1_score(labels, predicted, average='macro')
    mcc = metrics.matthews_corrcoef(labels, predicted)

    return {"accuracy": accuracy,
            "macro_f1": macro_f1,
            'mcc':mcc,}


def sample_from_model(coefficients, generator, n_time, x_init, nz, nclass):
    x = x_init
    with torch.no_grad():
        for i in reversed(range(n_time)):
            t = torch.full((x.size(0),), i, dtype=torch.int64).to(x.device)

            t_time = t
            latent_z = torch.randn(x.size(0), nz, device=x.device)
            latent_zc = torch.cat([latent_z, nclass], 1)
            latent_zc = latent_zc.to(torch.float)

            x_0 = generator(x, t_time, latent_zc)
            x_new = sample_posterior(coefficients, x_0, x, t)
            x = x_new.detach()

    return x

def x_init_sample(pos_coeff,sample_batch,args):
    Xdata = np.load(f'./datasets/{args.dataset}/DGOT/{args.exp}/xtrain.npy')
    Ydata = np.load(f'./datasets/{args.dataset}/DGOT/{args.exp}/ytrain.npy')
    index = np.where(Ydata == 1)
    minor = Xdata[index, :, :][0]
    idx = np.random.randint(0, minor.shape[0], sample_batch)
    minor_torch = torch.tensor(minor[idx], device='cuda:0',dtype=torch.float32)
    X_init = pos_coeff.sqrt_alphas_cumprod[-1] * minor_torch + pos_coeff.sqrt_one_minus_alphas_cumprod[-1] * torch.rand_like(minor_torch)

    return X_init

#%%
def sample_evaluate(init_num, final_num, xtrain, ytrain, xtest, ytest, classifiers, args_train, netG, pos_coeff, device):
    attrvalues = np.eye(args_train.class_num)
    for i, j in enumerate(init_num):
        sample_batch = int(final_num - j)
        classidx = [i] * sample_batch
        ytrain = np.hstack([ytrain, np.array(classidx)])

        classnum = torch.tensor(attrvalues[np.array(classidx)]).to(device)
        x_t_1 = torch.randn([sample_batch, 1, args_train.feature_len], device=device).float()
        # x_t_1 = x_init_sample(pos_coeff,sample_batch,args_train)
        fake_sample = sample_from_model(pos_coeff, netG, args_train.num_timesteps,
                                        x_t_1, args_train.nz, classnum)
        fake_data = np.array(fake_sample.to('cpu'))
        fake_data = fake_data.reshape(sample_batch, args_train.feature_len)
        xtrain = np.vstack([xtrain, fake_data])

    data = np.hstack([xtrain, ytrain[:, None]])
    np.random.shuffle(data)

    # evaluation
    model = classifiers
    model.fit(data[:, :-1], data[:, -1])
    pred = model.predict(xtest)
    result = indicator_cls(ytest, pred)

    return result


def DGOT(filepath, testpath, classifiers, oversample_rate, repetitions=20, devices='cuda'):
    from models.GaussionDiffusion import Posterior_Coefficients
    from models.Generator import Unet

    # initial configuration
    configs_file = os.path.join(filepath, 'configs.yaml')
    args_train = configs_read(configs_file)
    device = torch.device(devices)
    pos_coeff = Posterior_Coefficients(args_train, device)
    attrvalues = np.eye(args_train.class_num)  # Conditional information

    datapath = f'./datasets/{args_train.dataset}/DGOT/{args_train.exp}'

    # load test data
    xtrain = np.load(os.path.join(datapath, 'xtrain.npy')).squeeze()
    ytrain = np.load(os.path.join(datapath, 'ytrain.npy'))
    xtest = np.load(os.path.join(testpath, 'xtest.npy'))
    ytest = np.load(os.path.join(testpath, 'ytest.npy'))


    init_num = [j for _, j in sorted(Counter(ytrain).items())]  # The number of each category
    final_num = int(max(init_num)*oversample_rate) # The number of major category

    # load model
    netG = Unet(
        in_ch=1,
        out_ch=1,
        init_dim=args_train.feature_len,
        nz = args_train.nz+attrvalues.shape[0],
        init_ch = args_train.init_ch,
        ch_mult=args_train.ch_mult,
        resnet_block_groups=args_train.rbg,
    ).to(device)

    model_path = os.path.join(filepath, f'netG.pth')
    netG.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    performance = []
    with tqdm(initial=0,total=repetitions) as pbar:
        for k in range(repetitions):
            results = sample_evaluate(init_num, final_num, xtrain, ytrain, xtest, ytest, classifiers, args_train, netG,
                                      pos_coeff, device)
            performance.append(results)

            pbar.update(1)

    temp = pd.DataFrame(performance)
    means = temp.mean(axis=0)
    std = temp.std(axis=0)
    temp = temp._append(means, ignore_index=True)
    results = temp._append(std, ignore_index=True)

    return results
