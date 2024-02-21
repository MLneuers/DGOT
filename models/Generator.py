#%%
import math

import torch
from torch import nn
from functools import partial

from einops import rearrange


# helpers functions
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

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

# building block modules

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = nn.Conv1d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)

        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, z_emb_dim= None, groups = 8):
        super().__init__()

        #planA : merging infromation about time and latent z by using twice scale_shift
        self.mlp1 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.mlp2 = nn.Sequential(
            nn.SiLU(),
            nn.Linear(z_emb_dim, dim_out * 2)
        ) if exists(z_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()


    def forward(self, x, time_emb = None, z_emb = None):

        scale_shift1 = None
        scale_shift2 = None

        if exists(self.mlp1) and exists(time_emb):
            time_emb = self.mlp1(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 ')
            scale_shift1 = time_emb.chunk(2, dim = 1)

        if exists(self.mlp2) and exists(z_emb):
            z_emb = self.mlp2(z_emb)
            z_emb = rearrange(z_emb, 'b c -> b c 1 ')
            scale_shift2 = z_emb.chunk(2, dim = 1)

        res = self.res_conv(x)

        h = self.block1(x, scale_shift = scale_shift1)

        h = self.block2(h, scale_shift = scale_shift2)

        h= h + res

        return h

def Downsamle(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), 4, 2, 1)

def Upsample(dim,dim_out=None,op=None):
    return nn.ConvTranspose1d(dim, default(dim_out,dim), 2, stride=2, output_padding=op)

# model

class Unet(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            init_dim, #length
            nz, #latent z
            init_ch = 3,
            ch_mult = [2,4,8],
            resnet_block_groups = 4,
            learned_sinusoidal_cond = False,
            learned_sinusoidal_dim = 16):
        super(Unet, self).__init__()

        self.channels = in_ch
        self.init_conv = nn.Conv1d(in_ch, init_ch, 7,padding=3)

        # hyperparameter for Unet architecture

        chs = [init_ch, *map(lambda m: init_ch * int(m), ch_mult)]
        in_out = list(zip(chs[:-1], chs[1:]))
        length = [init_dim, *map(lambda x: init_dim // int(2**x), range(1,len(ch_mult)+1))]
        op = [*map(lambda x: x % 2, reversed(length[:-1]))]

        # basic block

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = init_ch * 4

        self.learned_sinusoidal_cond = learned_sinusoidal_cond

        if learned_sinusoidal_cond:
            sinu_pos_emb = LearnedSinusoidalPosEmb(learned_sinusoidal_dim)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(init_ch)
            fourier_dim = init_ch

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        #latent z embeddings

        z_dim = init_ch * 4

        self.z_mlp = nn.Sequential(
            nn.Linear(nz, z_dim),
            nn.GELU(),
            nn.Linear(z_dim, z_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for ind, (dim_in, dim_out) in enumerate(in_out):

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim = time_dim, z_emb_dim = z_dim),
                Downsamle(dim_in,dim_out)
            ]))

        mid_dim = chs[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim = time_dim, z_emb_dim = z_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_out, dim_out, time_emb_dim = time_dim, z_emb_dim = z_dim),
                Upsample(dim_out, dim_in, op=op[ind])
            ]))

        self.out_dim = out_ch
        self.final_res_block = block_klass(init_ch * 2, init_ch, time_emb_dim = time_dim)
        self.final_conv = nn.Conv1d(init_ch, self.out_dim, 1)
        self.final_activate = nn.Tanh()

    def forward(self, x, time, latent):

        t = self.time_mlp(time)
        z = self.z_mlp(latent)

        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, downsample in self.downs:
            x = block1(x, t, z)
            x = downsample(x)
            h.append(x)

        x = self.mid_block1(x, t, z)

        for block1, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim = 1)
            x = block1(x, t, z)
            x = upsample(x)

        x = torch.cat((x, r), dim = 1)
        x = self.final_res_block(x, t)
        x = self.final_conv(x)

        x = self.final_activate(x)

        return x


