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
from imblearn.metrics import geometric_mean_score as gm


import imblearn.over_sampling as ios
import imblearn.combine as ico
import smote_variants as sv


def exists(x):
    return x is not None

def configs_read(configs_file):
    with open(configs_file) as f:
        configs_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    configs = argparse.Namespace(**configs_dict)

    return configs


def indicator_cls(labels, predicted, predicted_prob):
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
    Xdata = np.load(f'./datasets_prep/{args.dataset}/BTDG/{args.exp}/xtrain.npy')
    Ydata = np.load(f'./datasets_prep/{args.dataset}/BTDG/{args.exp}/ytrain.npy')
    index = np.where(Ydata == 1)
    minor = Xdata[index, :, :][0]
    idx = np.random.randint(0, minor.shape[0], sample_batch)
    minor_torch = torch.tensor(minor[idx], device='cuda:0',dtype=torch.float32)
    X_init = pos_coeff.alphas_cumprod[-1] * minor_torch + (1-pos_coeff.alphas_cumprod[-1]) * torch.rand_like(minor_torch)

    return X_init

#%%
def BTDG_sample_evaluate(init_num, final_num, xtrain, ytrain, xtest, ytest, classifiers, args_train, netG, pos_coeff, device):
    attrvalues = np.eye(args_train.class_num)
    for i, j in enumerate(init_num):
        sample_batch = int(final_num - j)
        classidx = [i] * sample_batch
        ytrain = np.hstack([ytrain, np.array(classidx)])

        classnum = torch.tensor(attrvalues[np.array(classidx)]).to(device)
        # x_t_1 = torch.randn([sample_batch, 1, args_train.feature_len], device=device).float()
        x_t_1 = x_init_sample(pos_coeff,sample_batch,args_train)
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
    pred_prob = model.predict_proba(xtest)
    result = indicator_cls(ytest, pred, pred_prob[:, 1])

    return result



def BTDG(filepath, testpath, classifiers, oversample_rate, repetitions=20, slog=None, devices='cuda'):
    from models.GaussionDiffusion import Posterior_Coefficients
    from models.Generator import Unet

    # initial configuration
    configs_file = os.path.join(filepath, 'configs.yaml')
    args_train = configs_read(configs_file)
    device = torch.device(devices)
    pos_coeff = Posterior_Coefficients(args_train, device)
    attrvalues = np.eye(args_train.class_num)  # Conditional information

    datapath = f'./datasets_prep/{args_train.dataset}/BTDG/{args_train.exp}'

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
    # exp = args_train.exp
    # parent_dir = "./saved_log/BTDG/pure_D/{}".format(args_train.dataset)
    exp_path = filepath

    # Whether to automatically find the best results
    if exists(slog):
        model_path = os.path.join(exp_path, f'netG_{slog}.pth')
        netG.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    else:
        best_macro_f1 = 0
        slog_list = [i for i in range(0, args_train.num_epoch, args_train.save_ckpt_every)]

        with tqdm(initial=10, total=len(slog_list)) as pbar:
            for logs in slog_list[10:]:
                model_path = os.path.join(exp_path, f'netG_{logs}.pth')
                netG.load_state_dict(torch.load(model_path, map_location=device), strict=False)
                performance = []
                for k in range(1):
                    results = BTDG_sample_evaluate(init_num, final_num, xtrain, ytrain, xtest, ytest, classifiers, args_train, netG, pos_coeff, device)
                    performance.append(results)
                temp = pd.DataFrame(performance)
                means = temp.mean(axis=0)

                # best_macro_f1
                if best_macro_f1 < means['macro_f1']:
                    best_macro_f1 = means['macro_f1']
                    slog = logs

                pbar.set_description(f'Searching the optimal model: slog {slog} macro_f1 {best_macro_f1}')
                pbar.update(1)

    # over_sample and evaluation
    model_path = os.path.join(exp_path, f'netG_{slog}.pth')
    netG.load_state_dict(torch.load(model_path, map_location=device), strict=False)
    performance = []
    with tqdm(initial=0,total=repetitions) as pbar:
        for k in range(repetitions):
            results = BTDG_sample_evaluate(init_num, final_num, xtrain, ytrain, xtest, ytest, classifiers, args_train, netG, pos_coeff, device)
            performance.append(results)

            pbar.set_description(f'the testing procedure of BTDG_{slog} over-sampling approach')
            pbar.update(1)

    temp = pd.DataFrame(performance)
    means = temp.mean(axis=0)
    std = temp.std(axis=0)
    temp = temp.append(means, ignore_index=True)
    results = temp.append(std, ignore_index=True)

    return results, slog


def classic_baseline(filepath, testpath, classifiers, strategies='Indentity', repetitions=10):

    # initial configuration
    performance = []


    oversampling = lambda x, y:[x, y]  #Identity

    if strategies == 'ADASYN':
        oversampling = ios.ADASYN().fit_resample
    elif strategies == 'SMOTE':
        oversampling = ios.SMOTE().fit_resample
    elif  strategies == 'BSMOTE':
        oversampling = ios.BorderlineSMOTE().fit_resample
    elif strategies == 'SMOTETL':
        oversampling = ico.SMOTETomek().fit_resample
    elif strategies == 'SMOTEENN':
        oversampling = ico.SMOTEENN().fit_resample
    elif strategies == 'MDO':
        oversampling = sv.MDO().sample


    # load test data
    xtrain = np.load(os.path.join(filepath, 'xtrain.npy'))
    ytrain = np.load(os.path.join(filepath, 'ytrain.npy'))
    xtest = np.load(os.path.join(testpath, 'xtest.npy'))
    ytest = np.load(os.path.join(testpath, 'ytest.npy'))


    # over_sample and evaluation

    with tqdm(initial=0, total=repetitions) as pbar:
        for i in range(repetitions):
            X_res, y_res = oversampling(xtrain, ytrain)

            dmax = np.max(xtrain, axis=0)
            dmin = np.min(xtrain, axis=0)
            X_res = ((X_res - dmin) / (dmax - dmin+1e-8)) * 2 - 1

            model = classifiers
            model.fit(X_res, y_res)
            pred = model.predict(xtest)
            pred_prob = model.predict_proba(xtest)
            results = indicator_cls(ytest, pred, pred_prob[:,1])
            performance.append(results)

            pbar.set_description('the testing procedure of {} over-sampling approach'.format(strategies))
            pbar.update(1)

    temp = pd.DataFrame(performance)
    means = temp.mean(axis=0)
    std = temp.std(axis=0)
    temp = temp.append(means, ignore_index=True)
    results = temp.append(std, ignore_index=True)

    return results


def CTGAN_TVAE_sample(dataframe, model , classnums, classes, renums):
    syn_data = model.sample(200000)
    temp = syn_data.values
    if sum(temp[:,-1]) < 20:
        print("失败")
        return dataframe
    for i in classes:
        compensation = renums - classnums[i]
        if compensation > 0:
            taskdata = syn_data[syn_data[syn_data.columns[-1]] == i]
            if taskdata.shape[0] < compensation:
                dataframe = pd.concat([dataframe, taskdata], ignore_index=True)
                classnums[i] = classnums[i] + taskdata.shape[0]
            else:
                dataframe = pd.concat([dataframe, taskdata.head(compensation)], ignore_index=True)
                classnums[i] = classnums[i] + compensation

    if min(classnums) < renums:
        dataframe = CTGAN_TVAE_sample(dataframe, model, classnums, classes, renums)
        return dataframe
    else:
        return dataframe




def CTGAN_TVAE_evaluation(datapath, model, testpath, classifiers, discrete_columns,
                          epoch, oversample_rate, repetitions=10, strategy = 'CTGAN'):
    # load data
    data = pd.read_csv(os.path.join(datapath, 'train_df.csv'))

    xtest = np.load(os.path.join(testpath, 'xtest.npy'))
    ytest = np.load(os.path.join(testpath, 'ytest.npy'))



    # balancing initialization
    init_labels = data[data.columns[-1]].values
    init_class = [int(j) for j, _ in sorted(Counter(init_labels).items())]
    init_num = [j for _, j in sorted(Counter(init_labels).items())]
    final_num = int(max(init_num) * oversample_rate)
    performance = []



    with tqdm(initial=0, total=repetitions) as pbar:
        for i in range(repetitions):
            init_num = [j for _, j in sorted(Counter(init_labels).items())]
            syn_data = CTGAN_TVAE_sample(data, model, init_num, init_class, final_num)

            # data normalization
            init_xtrain = data.values[:,:-1].astype(float)
            syn_data_np = syn_data.values.astype(float)
            np.random.shuffle(syn_data_np)
            xtrain = syn_data_np[:,:-1]
            ytrain = syn_data_np[:,-1]

            # mu = np.mean(init_xtrain, axis=0)
            # std = np.std(init_xtrain, axis=0)

            # xtrain = (xtrain - mu) / std
            dmax = np.max(init_xtrain, axis=0)
            dmin = np.min(init_xtrain, axis=0)

            xtrain = ((xtrain - dmin) / (dmax - dmin+1e-8)) * 2 - 1


            # evaluation
            val_model = classifiers
            val_model.fit(xtrain, ytrain)
            pred = val_model.predict(xtest)
            pred_prob = val_model.predict_proba(xtest)
            results = indicator_cls(ytest, pred, pred_prob[:, 1])
            performance.append(results)

            pbar.set_description(f'the testing procedure of {strategy} over-sampling approach')
            pbar.update(1)

        temp = pd.DataFrame(performance)
        means = temp.mean(axis=0)
        std = temp.std(axis=0)
        temp = temp.append(means, ignore_index=True)
        results = temp.append(std, ignore_index=True)

        return results


