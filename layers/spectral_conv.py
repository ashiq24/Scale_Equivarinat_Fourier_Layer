"""Implements Spectral Convolution Support."""

import torch
import torch.nn as nn
from layers.spectral_mixer import SpectralMixer_2D
from utils.core_utils import get_mat
import math
from utils.core_utils import resample, get_pweights
from layers.complex_modules import get_block2d, enblock2d


def compl_mul1d(input, weights):
    # Complex multiplication
    return torch.einsum("bix,iox->box", input, weights)


def compl_mul2d(input, weights):
    return torch.einsum("bixy,ioxy->boxy", input, weights)


class SpectralConv2dCircular(nn.Module):
    def __init__(self, in_channel, out_channel, h_modes1, h_modes2, device=None, dtype=torch.cfloat):
        super(SpectralConv2dCircular, self).__init__()
        '''
        Regular Spectral Convolution.
        parameters
        ----------
        in_channel : number of input channels
        out_channel : number of output channels
        h_modes1 : number of horizontal modes
        h_modes2 : number of verticle modes
        '''
        assert h_modes1 == h_modes2

        self.in_channel = int(in_channel)
        self.out_channel = int(out_channel)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.factory_kwargs = {'device': device, 'dtype': dtype}

        self.h_modes1 = int(h_modes1)
        self.h_modes2 = int(h_modes2)
        self.W = nn.Parameter(torch.empty(self.in_channel, self.out_channel,
                                          2*self.h_modes1, 2*self.h_modes2, **factory_kwargs))

    def reset_parameter(self):
        scale = (1 / (2*self.in_channel))**(1.0/2.0)
        torch.nn.init.normal_(self.W, mean=0.0, std=scale)

    def forward(self, x_ft, pool=False):
        '''
        x_ft: input Fourier Co-efficinets 
        '''
        batchsize = x_ft.shape[0]

        if pool:
            dim1 = 2*self.h_modes1
            dim2 = 2*self.h_modes2
        else:
            dim1 = x_ft.shape[-2]
            dim2 = x_ft.shape[-1]
        effective_h_modes1 = min(self.W.shape[-2]//2, x_ft.shape[-2]//2)
        effective_h_modes2 = min(self.W.shape[-1]//2, x_ft.shape[-1]//2)

        out_ft = torch.zeros(batchsize, self.out_channel,
                             dim1, dim2, dtype=x_ft.type, device=x_ft.device)
        out_ft[:, :, :effective_h_modes1, :effective_h_modes2] = \
            compl_mul2d(x_ft[:, :, :effective_h_modes1, :effective_h_modes2],
                        self.W[:, :, :effective_h_modes1, :effective_h_modes2])

        out_ft[:, :, -effective_h_modes1:, :effective_h_modes2] = \
            compl_mul2d(x_ft[:, :, -effective_h_modes1:, :effective_h_modes2],
                        self.W[:, :, -effective_h_modes1:, :effective_h_modes2])

        out_ft[:, :, :effective_h_modes1, -effective_h_modes2:] = \
            compl_mul2d(x_ft[:, :, :effective_h_modes1, -effective_h_modes2:],
                        self.W[:, :, :effective_h_modes1, -effective_h_modes2:])

        out_ft[:, :, -effective_h_modes1:, -effective_h_modes2:] = \
            compl_mul2d(x_ft[:, :, -effective_h_modes1:, -effective_h_modes2:],
                        self.W[:, :, -effective_h_modes1:, -effective_h_modes2:])

        return out_ft


class SpectralConv2dLocalized(nn.Module):
    def __init__(self, in_channel, out_channel, global_modes, local_modes, device=None, dtype=torch.cfloat):
        '''
        Spatially localized Spectral convolution.
        parameters
        ----------
        in_channel : int, number of input channels
        out_channel : int, number of output channels
        Global_modes : int, number of global modes/ determines the total number of modes in the filter
        local_modes : int, number of local modes/ determines the effective spatial size of the filter
        '''
        super(SpectralConv2dLocalized, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_channel = int(in_channel)
        self.out_channel = int(out_channel)
        assert global_modes % 2 == 0
        self.local_h_modes = int(0.5*local_modes)
        self.local_modes = local_modes
        self.global_modes = int(global_modes)
        self.L = nn.Parameter(torch.empty(
            self.in_channel, self.out_channel, local_modes, self.local_modes, **factory_kwargs))
        self.register_buffer(
            "spectral_decay", get_pweights(local_modes, local_modes))
        self.register_buffer("mat", get_mat(
            self.global_modes, self.L.shape[-2]).to(self.L.device, dtype=self.L.dtype))
        self.reset_parameter()

    def reset_parameter_1(self):
        scale = (1/self.in_channel)**0.5
        torch.nn.init.uniform_(self.L, a=-scale, b=-scale)

    def reset_parameter(self):
        # self.global_modes/(2*self.local_h_modes)
        scale = (1/self.in_channel)**0.5
        torch.nn.init.normal_(self.L, std=scale)

    def get_reg_loss(self,):
        return torch.norm(self.L * self.spectral_decay[None, None, :, :]) + torch.norm(torch.fft.ifft2(self.L, norm='forward'))

    def get_filters(self,):
        W = self.mat @ self.L
        W = self.mat @ torch.transpose(W, -1, -2)
        W = torch.transpose(W, -1, - 2)
        return torch.fft.ifft2(W, norm='forward')

    def get_filters_spectral(self,):
        W = self.mat @ self.L
        W = self.mat @ torch.transpose(W, -1, -2)
        W = torch.transpose(W, -1, - 2)
        return W

    def forward(self, x_ft, pool=False):
        '''
        x_ft: input Fourier Co-efficinets 
        '''
        batchsize = x_ft.shape[0]

        if pool:
            dim1 = self.global_modes
            dim2 = self.global_modes
        else:
            dim1 = x_ft.shape[-2]
            dim2 = x_ft.shape[-1]

        W = self.mat @ self.L
        W = self.mat @ torch.transpose(W, -1, -2)

        W = torch.transpose(W, -1, - 2)

        effective_h_modes1 = min(W.shape[-2]//2, x_ft.shape[-2]//2)
        effective_h_modes2 = min(W.shape[-1]//2, x_ft.shape[-1]//2)

        s = x_ft.shape[-1] % 2

        out_ft = torch.zeros(batchsize, self.out_channel,
                             dim1, dim2, dtype=x_ft.dtype, device=self.L.device)

        out_ft[:, :, :effective_h_modes1+s, :effective_h_modes2+s] = \
            compl_mul2d(x_ft[:, :, :effective_h_modes1+s, :effective_h_modes2+s],
                        W[:, :, :effective_h_modes1+s, :effective_h_modes2+s])

        out_ft[:, :, -effective_h_modes1:, :effective_h_modes2+s] = \
            compl_mul2d(x_ft[:, :, -effective_h_modes1:, :effective_h_modes2+s],
                        W[:, :, -effective_h_modes1:, :effective_h_modes2+s])

        out_ft[:, :, :effective_h_modes1+s, -effective_h_modes2:] = \
            compl_mul2d(x_ft[:, :, :effective_h_modes1+s, -effective_h_modes2:],
                        W[:, :, :effective_h_modes1+s, -effective_h_modes2:])

        out_ft[:, :, -effective_h_modes1:, -effective_h_modes2:] = \
            compl_mul2d(x_ft[:, :, -effective_h_modes1:, -effective_h_modes2:],
                        W[:, :, -effective_h_modes1:, -effective_h_modes2:])

        return out_ft

    def get_feature(self, x):
        return torch.fft.ifft2(self.forward(torch.fft.fft2(x))).real


class SpectralConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, mode_size, device=None, dtype=torch.cfloat):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mode_size = mode_size

        self.weights = nn.Parameter(torch.empty(
            in_channels, out_channels, mode_size, **factory_kwargs))
        self.reset_parameter()

    def reset_parameter(self):
        scale = (1 / (2*self.in_channels))**(1.0/2.0)
        torch.nn.init.normal_(self.weights, mean=0.0, std=scale)

    def forward(self, input):
        '''
        input : input fourier coefficients
        '''
        ret = torch.zeros(input.shape[0], self.out_channels,
                          self.mode_size, dtype=input.dtype)
        ret[:, :, :self.mode_size] = compl_mul1d(
            input[:, :, :self.mode_size], self.weights)
        return ret


class SpectralConvLocalized1d(nn.Module):
    def __init__(self, in_channel, out_channel, global_modes, local_modes, device=None, dtype=torch.cfloat):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.local_h_modes = int(0.5*local_modes)
        self.local_modes = local_modes
        self.global_modes = int(global_modes)
        self.L = nn.Parameter(torch.empty(
            self.in_channel, self.out_channel, local_modes, **factory_kwargs))
        self.register_buffer("spectral_decay", get_pweights(
            local_modes, local_modes)[0, :])
        self.reset_parameter()

        self.reset_parameter()
        # TODO: This layer takes fft as input

    def reset_parameter(self):
        scale = self.global_modes/(2*self.local_h_modes)
        torch.nn.init.normal_(self.L, std=scale)

    def get_reg_loss(self,):
        return torch.norm(self.L * self.spectral_decay[None, None, :])

    def get_filters(self,):
        mat = get_mat(self.global_modes,
                      self.L.shape[-1]).to(self.L.device, dtype=self.L.dtype)
        W = self.L @ torch.transpose(mat, -1, -2)
        return torch.fft.ifft(W, norm='forward')

    def forward(self, x_ft, pool=False):
        '''
        x_ft: input Fourier Co-efficinets 
        '''
        batchsize = x_ft.shape[0]

        if pool:
            dim = self.global_modes
        else:
            dim = x_ft.shape[-1]

        mat = get_mat(self.global_modes,
                      self.L.shape[-1]).to(self.L.device, dtype=self.L.dtype)
        # print(mat.type(), FS_.type())
        W = self.L @ torch.transpose(mat, -1, -2)

        effective_h_modes1 = min(W.shape[-1]//2, x_ft.shape[-1]//2)

        s = x_ft.shape[-1] % 2

        out_ft = torch.zeros(batchsize, self.out_channel,
                             dim, dtype=x_ft.dtype, device=self.L.device)

        out_ft[:, :, :effective_h_modes1+s] = \
            compl_mul1d(x_ft[:, :, :effective_h_modes1+s],
                        W[:, :, :effective_h_modes1+s])

        out_ft[:, :, -effective_h_modes1:] = \
            compl_mul1d(x_ft[:, :, -effective_h_modes1:],
                        W[:, :, -effective_h_modes1:])

        return out_ft


class softPaching(nn.Module):
    def __init__(self, dim, number) -> None:
        '''
        Soft Paching Layer

        '''
        super().__init__()
        self.dim = dim
        self.number = number

    def forward(self, x):
        t = None
        ps = self.dim // self.number
        for i in range(self.number):
            for j in range(self.number):
                m = torch.zeros(self.dim, self.dim).to(x.device, dtype=x.dtype)
                m[i*ps: (i+1)*ps, j*ps: (j+1)*ps] = 1.0
                mr = resample(m[None, None, :, :], num=(
                    x.shape[-2], x.shape[-1]), complex=False)[0, 0, :, :]
                x_t = x * mr[None, None, :, :]
                if t is None:
                    t = x_t
                else:
                    t = torch.cat((t, x_t), dim=1)

        return t
