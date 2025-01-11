import torch
import torch.nn as nn
import math
from einops import rearrange
import numpy as np

# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1) ##按行拼接
        return emb

class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0 ##input=0 return false
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        self.main = nn.Sequential(
            nn.Conv1d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(in_features),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_features, in_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(in_features),
        )

    def forward(self, x):
        return x + self.main(x)

# building block modules

class DownBlock(nn.Module):
    def __init__(
            self,
            in_channel,
            out_channel,
            t_emb_dim=128,
            kernel_size=4,
            stride=2,
            padding=1,
            act=nn.LeakyReLU(0.2),
                ):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.Downsample1 = nn.Conv1d(out_channel, out_channel, kernel_size, stride, padding, bias=False)
        self.Downsample2 = nn.Conv1d(in_channel, out_channel, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm1d(out_channel)
        self.act = act
        self.dense_t1 = nn.Linear(t_emb_dim, out_channel)

    def forward(self, input, t_emb, resblock=True):
        out = self.act(input)

        out = self.conv1(out)
        out += self.dense_t1(t_emb)[..., None]
        out = self.act(out)

        out = self.Downsample1(out)
        out = self.bn(out)

        if resblock:
            skip = self.Downsample2(input)
            return (out + skip) / np.sqrt(2)
        else:
            return out

class discriminator(nn.Module):

    # time dependent discriminator
    def __init__(self,
                 nc,
                 ndf,
                 init_ch=1,
                 time_dime = 32,
                 learned_sinusoidal_cond=False,
                 learned_sinusoidal_dim=16,
                 act = nn.LeakyReLU(0.2),
                 out_class = 1,
                 ):

        super(discriminator,self).__init__()

        #time embedding
        self.time_dim = time_dime

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(init_ch)
            fourier_dim = init_ch

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, self.time_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(self.time_dim, self.time_dim),
            nn.LeakyReLU(0.2)
        )

        self.act = act
        self.conv1 = nn.Conv1d(nc, ndf, 3, 1, 1, bias=False)

        self.down1 = DownBlock(ndf, ndf*2,t_emb_dim=self.time_dim)
        self.down2 = DownBlock(ndf*2, ndf*4,t_emb_dim=self.time_dim)
        self.down3 = DownBlock(ndf*4, ndf*8,t_emb_dim=self.time_dim)

        self.conv2 = nn.Sequential(
            nn.Conv1d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
                                   )

        self.end_linear = nn.Linear(ndf*4,out_class)
        self.end_real = nn.Linear(ndf*4,1)


    def forward(self, x, time, x_t):
        # See note [TorchScript super()]
        x = torch.cat([x,x_t], axis=1)
        t = self.time_mlp(time)

        out = self.conv1(x)

        out = self.down1(out, t, resblock=True)
        out = self.down2(out, t, resblock=True)

        out = self.act(out)

        out = out.view(out.shape[0],out.shape[1],-1).sum(2)
        out_class = self.end_linear(out)
        out_rf = self.end_real(out)

        return out_class, out_rf






