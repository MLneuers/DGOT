import yaml
import argparse
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import os

from einops import rearrange

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader



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

# %%
def train(rank, gpu, args):
    from datasets_prep.dataprocessing import datasets

    from models.GaussionDiffusion import sample_posterior, q_sample_pairs
    from models.GaussionDiffusion import Diffusion_Coefficients, Posterior_Coefficients

    from models.Generator import Unet
    from models.DC_Discriminator import discriminator

    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))

    batch_size = args.batch_size

    nz = args.nz  # latent dimension
    cdim = args.class_num + 1

    # DataSet
    datapath = f'./datasets_prep/{args.dataset}/BTDG/{args.exp}'
    ds = datasets(datapath)

    # dist
    data_loader = DataLoader(ds,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=4,
                             pin_memory=True,
                             drop_last=True,)

    # Model, loss function and aptimizer
    attrvalues = np.eye(cdim - 1)  # Conditional information

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
                         out_class=cdim-1,
                         ).to(device)

    # optimizer from Wasserstein GAN

    optimizerD = optim.RMSprop(netD.parameters(), lr=args.lr_d, alpha=args.beta1)
    optimizerG = optim.RMSprop(netG.parameters(), lr=args.lr_g, alpha=args.beta2)

    # Copy all the programs and create an independent folders
    exp = args.exp
    parent_dir = "./saved_log/DGOT/{}".format(args.dataset)

    exp_path = os.path.join(parent_dir, exp)
    if rank == 0:
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

                # sample t
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
                # Update D
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

                loss = celoss(output_class, y) + F.mse_loss(real_data, x_0_predict)  * extract(p2, t, D_real.shape)+ F.softplus(-output_logit)

                errG = loss.mean()

                errG.backward()
                optimizerG.step()

                global_step += 1
                if iteration % 10 == 0:
                    if rank == 0:
                        print(
                            "Training:Epoch[{:0>3}/{:0>3}] iteration[{:0>3}/{:0>3}]  G Loss: {}  D Loss: {} D1:{} D2:{}".format(
                                epoch, args.num_epoch, iteration, len(data_loader), errG.item(), errD.item(),errD_real.item(),errD_fake.item()))

                pbar.set_description(
                    "Training:Epoch[{:0>3}/{:0>3}] iteration[{:0>3}/{:0>3}]  G Loss: {}  D Loss: {}  D1:{}  D2:{}".format(
                        epoch, args.num_epoch, iteration, len(data_loader), errG.item(), errD.item(),errD_real.item(),errD_fake.item()))

            pbar.update(1)



            if rank == 0:

                if args.save_content:
                    if epoch % args.save_content_every == 0:
                        print('Saving content.')
                        content = {'epoch': epoch + 1, 'global_step': global_step, 'args': args,
                                   'netG_dict': netG.state_dict(), 'optimizerG': optimizerG.state_dict(),
                                   'netD_dict': netD.state_dict(),
                                   'optimizerD': optimizerD.state_dict()}

                        torch.save(content, os.path.join(exp_path, 'content.pth'))

                if epoch % args.save_ckpt_every == 0:
                    if args.use_ema:
                        optimizerG.swap_parameters_with_ema(store_params_in_ema=True)

                    torch.save(netG.state_dict(), os.path.join(exp_path, 'netG_{}.pth'.format(epoch)))
                    if args.use_ema:
                        optimizerG.swap_parameters_with_ema(store_params_in_ema=True)





if __name__ == '__main__':
    parser = argparse.ArgumentParser('BA-DGAN parameters')

    # diffusion param
    parser.add_argument('--use_geometric', action='store_true', default=False)
    parser.add_argument('--beta_min', type=float, default=0.1, help='beta_min for diffusion')
    parser.add_argument('--beta_max', type=float, default=20., help='beta_max for diffusion')
    parser.add_argument('--num_timesteps', type=int, default=4)


    #training param
    parser.add_argument('--seed', type=int, default=505, help='seed used for initialization')
    parser.add_argument('--batch_size', type=int, default=32, help='input batch size')
    parser.add_argument('--num_epoch', type=int, default=1200)

    parser.add_argument('--exp', default='exp0', help='name of experiment')
    parser.add_argument('--save_content', action='store_true', default=False)
    parser.add_argument('--save_content_every', type=int, default=100, help='save content for resuming every x epochs')
    parser.add_argument('--save_ckpt_every', type=int, default=10, help='save ckpt every x epochs')
    parser.add_argument('--resume', action='store_true', default=False)

    parser.add_argument('--lr_g', type=float, default=1.5e-5, help='learning rate g')
    parser.add_argument('--lr_d', type=float, default=1e-5, help='learning rate d')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta for g_RMSProp')
    parser.add_argument('--beta2', type=float, default=0.9, help='beta for d_RMSProp')


    parser.add_argument('--use_ema', action='store_true', default=False, help='use EMA or not')
    parser.add_argument('--ema_decay', type=float, default=0.9999, help='decay rate for EMA')

    parser.add_argument('--r1_gamma', type=float, default=0.05, help='coef for r1 reg')
    parser.add_argument('--lazy_reg', type=int, default=None, help='lazy regulariation.')


    # ddp param
    parser.add_argument('--num_proc_node', type=int, default=1, help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1, help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0, help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0, help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1', help='address for master')

    # dataset
    parser.add_argument('--dataset', default='TEData', help='name of dataset')
    parser.add_argument('--class_num', type=int, default=0, help='The index of node.')

    #loss param
    parser.add_argument('--pw1', type=float, default=1, help='p2_loss_weight_gamma')
    parser.add_argument('--pw2', type=float, default=1, help='p2_loss_weight_k')

    #generator
    parser.add_argument('--init_ch', type=int, default= 64 )
    parser.add_argument('--ch_mult', nargs='+',type=int, default=[1, 2, 2])
    parser.add_argument('--feature_len', type=int, default=50, help='length of feature')
    parser.add_argument('--nz', type=int, default=100)
    parser.add_argument('--rbg', type=int, default=4)


    #discriminator
    parser.add_argument('--num_channels', type=int, default=1, help='channel of data')
    parser.add_argument('--t_emb_dim', type=int, default=256)
    parser.add_argument('--ngf', type=int, default=64)


    #configs management
    parser.add_argument('--save_configs', action='store_false', default=True)
    parser.add_argument('--use_configs', action='store_true', default=False)
    parser.add_argument('--configs_file',default='.\configs\configs_binary.yaml', help='saving path of configs')

    args = parser.parse_args()

    #ddp
    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node

    #import configs, if exist
    if args.use_configs:
        args = configs_read(args)

    train(0,'0',args)