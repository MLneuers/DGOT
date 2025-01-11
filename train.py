import yaml
import argparse
from tqdm.auto import tqdm
import pandas as pd
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from collections import Counter
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings



#%%
_MODELS = {
	'binary_classification': [

		{
			'class': XGBClassifier,
			'kwargs': {
				'max_depth': 3,
				'n_estimators': 100
			}
		},

		{
			'class': DecisionTreeClassifier,
			'kwargs': {
				'max_depth': 30
			}
		},

		{
			'class': LogisticRegression,
			'kwargs': {
				'penalty': 'l2',
				'max_iter': 500
			}
		},

		{
			'class': RandomForestClassifier,
			'kwargs': {
				'n_estimators': 100
			}
		},

		{
			'class': KNeighborsClassifier,
			'kwargs': {
				'n_neighbors': 5
			}
		},
	]
}
from sklearn.exceptions import ConvergenceWarning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=ConvergenceWarning)

def configs_save(args, filepath):
    configs_dict = vars(args)
    with open(os.path.join(filepath, 'configs.yaml'), 'w') as f:
        f.write(yaml.dump(configs_dict, allow_unicode=True))

def configs_read(args):
    with open(args.configs_file) as f:
        configs_dict = yaml.load(f.read(), Loader=yaml.FullLoader)
    configs = argparse.Namespace(**configs_dict)

    return configs

def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out

def celoss(outputs, labels):

    softmax_outputs = F.softmax(outputs, dim=1)

    one_hot_labels = torch.zeros_like(outputs)
    one_hot_labels.scatter_(1, labels.unsqueeze(1), 1)

    loss = -one_hot_labels * torch.log(softmax_outputs + 1e-10)

    return loss

def val_data_process(args):
    filepaths_test = f'./datasets/{args.dataset}/TEST/{args.exp}'
    filepaths_DGOT = f'./datasets/{args.dataset}/DGOT/{args.exp}'

    xtrain = np.load(os.path.join(filepaths_DGOT, 'xtrain.npy')).squeeze()
    ytrain = np.load(os.path.join(filepaths_DGOT, 'ytrain.npy'))
    xtest = np.load(os.path.join(filepaths_test, 'xtest.npy'))
    ytest = np.load(os.path.join(filepaths_test, 'ytest.npy'))

    init_num = [j for _, j in sorted(Counter(ytrain).items())]  # The number of each category
    final_num = int(max(init_num) * 1.1)  # The number of major category

    return {
        'xtrain':xtrain,
        'ytrain':ytrain,
        'xtest':xtest,
        'ytest':ytest,
        'init_num':init_num,
        'final_num':final_num
    }


# %%
def train(args):
    from scripts.dataprocessing import datasets

    from models.GaussionDiffusion import sample_posterior, q_sample_pairs
    from models.GaussionDiffusion import Diffusion_Coefficients, Posterior_Coefficients

    from models.Generator import Unet
    from models.DC_Discriminator import discriminator

    if args.class_num > 2:
        from scripts.evaluate_multi import sample_evaluate
    else:
        from scripts.evaluate_binary import sample_evaluate

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True  # 确定卷积算法固定
    torch.backends.cudnn.benchmark = False  # 不进行卷积算法的预搜索
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    device = args.device
    batch_size = args.batch_size
    nz = args.nz  # latent dimension

    # DataSet
    datapath = f'./datasets/{args.dataset}/DGOT/{args.exp}'
    ds = datasets(datapath)
    val_dict = val_data_process(args)
    data_loader = DataLoader(ds,
                             batch_size=batch_size,
                             shuffle=True,
                             pin_memory=True,
                             drop_last=True,)
    attrvalues = np.eye(args.class_num)  # Conditional information

    # model
    netG = Unet(
        in_ch=1,
        out_ch=1,
        init_dim=args.feature_len,
        nz = args.nz+attrvalues.shape[0],
        init_ch = args.init_ch,
        ch_mult=args.ch_mult,
        resnet_block_groups=args.rbg,
    ).to(device)

    netD = discriminator(
                         nc = 2*args.num_channels,
                         ndf = args.ngf,
                         init_ch=16,
                         time_dime = args.t_emb_dim,
                         act=nn.LeakyReLU(0.2),
                         out_class=args.class_num,
                         ).to(device)

    # optimizer from Wasserstein GAN
    optimizerD = optim.RMSprop(netD.parameters(), lr=args.lr_d, alpha=args.beta1)
    optimizerG = optim.RMSprop(netG.parameters(), lr=args.lr_g, alpha=args.beta2)

    # Copy all the programs and create an independent folders
    exp = args.exp
    parent_dir = "./saved_log/DGOT/{}".format(args.dataset)
    exp_path = os.path.join(parent_dir, exp)
    best_f1 = 0.0

    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    if args.save_configs:
        configs_save(args, exp_path)


    # all diffusion hyper-paramters and T step
    coeff = Diffusion_Coefficients(args, device)
    pos_coeff = Posterior_Coefficients(args, device)
    p2 = args.pw1/(1 + torch.log(args.pw2 + pos_coeff.alphas_cumprod/((1-pos_coeff.alphas_cumprod))))


    # import checkpoint， if present
    if args.resume:
        checkpoint_file = os.path.join(exp_path, 'content.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint['epoch']
        netG.load_state_dict(checkpoint['netG_dict'])
        # load G

        optimizerG.load_state_dict(checkpoint['optimizerG'])

        # load D
        netD.load_state_dict(checkpoint['netD_dict'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])

        global_step = checkpoint['global_step']
        best_f1 = checkpoint['best_f1']
        print("=> loaded checkpoint (epoch {})"
              .format(checkpoint['epoch']))
    else:
        global_step, epoch, init_epoch = 0, 0, 0




    # Train
    with tqdm(initial=init_epoch, total=args.num_epoch) as pbar:
        for epoch in range(init_epoch, args.num_epoch + 1):
            for iteration, (x, y) in enumerate(data_loader):
                for p in netD.parameters():
                    p.requires_grad = True

                condition = torch.tensor(attrvalues[y.cpu().numpy()]).to(device)
                y = y.to(device)
                y = y.to(torch.long)

                netD.zero_grad()

                # sample from p(x_0)
                real_data = x.to(device, non_blocking=True)
                t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
                x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)
                x_t.requires_grad = True

                # train with real
                D_clf, D_real = netD(x_t, t, x_tp1.detach())
                loss = celoss(D_clf, y)*extract(p2, t, D_real.shape) + F.softplus(- D_real)
                errD_real = loss.mean()
                errD_real.backward(retain_graph=True)

                #grad penalty
                if args.lazy_reg is None:
                    grad_real = torch.autograd.grad(
                        outputs=D_real.sum(), inputs=x_t, create_graph=True
                    )[0]
                    grad_penalty = (
                            grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                    ).mean()

                    grad_penalty = args.r1_gamma / 2 * grad_penalty
                    grad_penalty.backward()
                else:
                    if global_step % args.lazy_reg == 0:
                        grad_real = torch.autograd.grad(
                            outputs=D_real.sum(), inputs=x_t, create_graph=True
                        )[0]
                        grad_penalty = (
                                grad_real.view(grad_real.size(0), -1).norm(2, dim=1) ** 2
                        ).mean()

                        grad_penalty = args.r1_gamma / 2 * grad_penalty
                        grad_penalty.backward()

                # train with fake
                latent_z = torch.randn(batch_size, nz, device=device)
                latent_zc = torch.cat([latent_z, condition], 1)
                latent_zc = latent_zc.to(torch.float)

                x_0_predict = netG(x_tp1.detach(), t, latent_zc)
                x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)

                D_g_clf, D_fake = netD(x_pos_sample, t, x_tp1.detach())

                loss = celoss(D_g_clf, y) * extract(p2, t, D_real.shape)+ F.softplus(D_fake)
                errD_fake = loss.mean()
                errD_fake.backward()
                errD = errD_real + errD_fake
                optimizerD.step()

                # update G
                for p in netD.parameters():
                    p.requires_grad = False
                netG.zero_grad()

                t = torch.randint(0, args.num_timesteps, (real_data.size(0),), device=device)
                x_t, x_tp1 = q_sample_pairs(coeff, real_data, t)

                latent_z = torch.randn(batch_size, nz, device=device)
                latent_zc = torch.cat([latent_z, condition], 1)
                latent_zc = latent_zc.to(torch.float)

                x_0_predict = netG(x_tp1.detach(), t, latent_zc)
                x_pos_sample = sample_posterior(pos_coeff, x_0_predict, x_tp1, t)
                output_class, output_logit = netD(x_pos_sample, t, x_tp1.detach())

                loss = celoss(output_class, y) * extract(p2, t, D_real.shape) + F.softplus(-output_logit)
                errG = loss.mean()
                errG.backward()
                optimizerG.step()

                global_step += 1

                pbar.set_description(
                    "Training:Epoch[{:0>3}/{:0>3}] iteration[{:0>3}/{:0>3}]  G Loss: {}  D Loss: {}  D1:{}  D2:{}".format(
                        epoch, args.num_epoch, iteration, len(data_loader), errG.item(), errD.item(),errD_real.item(),errD_fake.item()))
            pbar.update(1)


            if args.save_content:
                if epoch % args.save_content_every == 0:
                    content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                               'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                               'netD_dict': netD.state_dict(),
                               'optimizerD': optimizerD.state_dict(),
                               'best_f1':best_f1}
                    torch.save(content, os.path.join(exp_path, 'content.pth'))

            if epoch % args.save_ckpt_every == 0 and epoch>100:
                with torch.no_grad():
                    classifiers = _MODELS['binary_classification']
                    performance = []
                    for i, cf in enumerate(classifiers):
                        model_param = cf['kwargs']
                        model = cf['class'](**model_param)
                        for k in range(3):
                            results = sample_evaluate(val_dict['init_num'],
                                                      val_dict['final_num'],
                                                      val_dict['xtrain'],
                                                      val_dict['ytrain'],
                                                      val_dict['xtest'],
                                                      val_dict['ytest'],
                                                      model,
                                                      args, netG, pos_coeff,device)
                            performance.append(results)
                    temp = pd.DataFrame(performance)
                    means = temp.mean(axis=0)

                if best_f1 < means['macro_f1']:
                    best_f1 = means['macro_f1']
                    # print('\n',epoch, best_f1)
                    torch.save(netG.state_dict(), os.path.join(exp_path, 'netG.pth'.format(epoch)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser('DGOT parameters')

    # diffusion param
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
    parser.add_argument('--num_timesteps', type=int, default=4)


    # training param
    parser.add_argument('--seed', type=int, default=666, help='seed used for initialization')
    parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=800)
    parser.add_argument('--device',default='cuda:0', help='name of experiment')

    parser.add_argument('--exp', default='exp0', help='name of experiment')
    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=50, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=5, help='save ckpt every x epochs')
    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--lr_d', type=float, default=2e-3, help='learning rate d')
    parser.add_argument('--lr_g', type=float, default=5e-3, help='learning rate g')
    parser.add_argument('--beta1', type=float, default=0.8, help='beta for g_RMSProp')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta for d_RMSProp')

    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None, help='lazy regulariation.')


    # dataset
    parser.add_argument('--dataset', default='abalone_15', help='name of dataset')
    parser.add_argument('--class_num', type=int, default=2)

    # loss param
    parser.add_argument('--pw1', type=float, default=1, help='p2_loss_weight_gamma')
    parser.add_argument('--pw2', type=float, default=1, help='p2_loss_weight_k')

    # generator
    parser.add_argument('--init_ch', type=int, default= 16 )
    parser.add_argument('--ch_mult', nargs='+',type=int, default=[1, 2, 2])
    parser.add_argument('--feature_len', type=int, default=8, help='length of feature')
    parser.add_argument('--nz', type=int, default=50)
    parser.add_argument('--rbg', type=int, default=4)


    # discriminator
    parser.add_argument('--num_channels', type=int, default=1, help='channel of data')
    parser.add_argument('--t_emb_dim', type=int, default=32)
    parser.add_argument('--ngf', type=int, default=32)


    # configs management
    parser.add_argument('--save_configs', action='store_false', default=True)
    parser.add_argument('--use_configs', action='store_true', default=False)
    parser.add_argument('--configs_file',default='.\configs\configs_binary.yaml', help='saving path of configs')

    args = parser.parse_args()


    # import configs, if exist
    if args.use_configs:
        args = configs_read(args)

    train(args)