"""Spectral Mixer Implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from scale_eq.utils.core_utils import get_mixer_mask, ComplexGELU, ComplexSELU, get_pweights, ComplexLeakyRELU, ComplexTanh, ComplexRELU, get_pweights_1D


class SpectralMixer_1D(nn.Module):
    def __init__(self, num_channels, mode_size, mixer_band=-1, device=None, dtype=torch.cfloat):
        '''
        num_channels: Number of Co-domain of the input signal
        mode_size: Maximum number of Fourier models to use. Normally can be set to size of the input signal.
        mixer_band: Limits the number of neighboring Fourier models to mix with each other. Deafults to -1 implies no limit on mixing.
        '''
        super(SpectralMixer_1D, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.mode_size = mode_size
        self.W1 = nn.Parameter(torch.empty(
            num_channels, self.mode_size, self.mode_size, **factory_kwargs))
        self.bias1 = nn.Parameter(torch.empty(
            num_channels, self.mode_size, **factory_kwargs))
        self.mixer_band = mixer_band

        self.register_buffer(
            "spectral_decay", get_pweights(mode_size, mode_size))
        self.reset_parameter()

    def get_reg_loss(self,):
        return torch.norm(self.spectral_decay * self.W1) + torch.norm(self.bias1)

    def reset_parameter_1(self):
        scale = (1 / self.mode_size)**0.5
        torch.nn.init.uniform_(self.W1, a=-scale, b=scale)
        torch.nn.init.uniform_(self.bias1, a=-scale, b=scale)

    def reset_parameter(self):
        scale = (2 / self.mode_size)**0.5
        torch.nn.init.normal_(self.W1, std=scale)
        torch.nn.init.normal_(self.bias1, std=scale)

    def forward(self, x, in_modes1=None):
        '''
        x : input Fourier Co-efficient of shape (batch_size, num_channels, Fourier Coefficients)
        in_modes1: Number of modes to mix. If None, all of the Fourier Coefficients will be considered for mixing.
        '''
        if in_modes1 is None:
            in_modes1 = x.shape[-1]
        h_modes1 = min(self.mode_size//2, in_modes1//2)
        s = in_modes1 % 2
        temp = torch.zeros(x.shape[0], x.shape[1],
                           h_modes1*2 + s, dtype=x.dtype, device=x.device)

        eW1 = torch.zeros(x.shape[1], h_modes1*2+s,
                          2*h_modes1+s, dtype=x.dtype, device=x.device)
        bias1 = torch.zeros(x.shape[1],  h_modes1 *
                            2+s, dtype=x.dtype, device=x.device)

        temp[:, :, :h_modes1+s] = x[:, :, :h_modes1+s]
        temp[:, :, -h_modes1:] = x[:, :, -h_modes1:]

        eW1[:, :h_modes1+s, :h_modes1+s] = self.W1[:, :h_modes1+s, :h_modes1+s]
        eW1[:, -h_modes1:, :h_modes1+s] = self.W1[:, -h_modes1:, :h_modes1+s]
        eW1[:, :h_modes1+s, -h_modes1:] = self.W1[:, :h_modes1+s, -h_modes1:]
        eW1[:, -h_modes1:, -h_modes1:] = self.W1[:, -h_modes1:, -h_modes1:]

        bias1[:, :h_modes1+s] = self.bias1[:, :h_modes1+s]
        bias1[:, -h_modes1:] = self.bias1[:, -h_modes1:]

        mask = get_mixer_mask(h_modes1*2+s, dtype=x.dtype,
                              band_width=self.mixer_band).to(x.device)

        temp = ((eW1 * mask[None, :, :])[None, :, :, :] @
                temp[..., None])[:, :, :, 0] + bias1[None, :, :]

        x[:, :, :h_modes1+s,] = temp[:, :, :h_modes1+s]
        x[:, :, -h_modes1:] = temp[:, :, -h_modes1:]

        return x


class SpectralMixer_2D(nn.Module):
    def __init__(self, num_channels, mode_size, mixer_band=-1, device=None, dtype=torch.cfloat):
        '''
        num_channels: Number of Co-domain of the input signal
        mode_size: Maximum number of Fourier models to use. Normally can be set to size of the input signal.
        mixer_band: Limits the number of neighboring Fourier models to mix with each other. Deafults to -1 implies no limit on mixing.
        '''
        super(SpectralMixer_2D, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.mode_size = mode_size
        self.W1 = nn.Parameter(torch.empty(
            num_channels, self.mode_size, self.mode_size, **factory_kwargs))
        self.W2 = nn.Parameter(torch.empty(
            num_channels, self.mode_size, self.mode_size, **factory_kwargs))
        self.bias1 = nn.Parameter(torch.empty(
            num_channels, self.mode_size, self.mode_size, **factory_kwargs))
        self.mixer_band = mixer_band

        self.register_buffer(
            "spectral_decay", get_pweights(mode_size, mode_size))
        self.reset_parameter()

    def get_reg_loss(self,):
        return torch.norm(self.spectral_decay * self.W1) + torch.norm(self.spectral_decay * self.W2) + torch.norm(self.bias1)

    def reset_parameter_1(self):
        scale = (1 / self.mode_size)**0.5
        torch.nn.init.uniform_(self.W1, a=-scale, b=scale)
        torch.nn.init.uniform_(self.W2, a=-scale, b=scale)
        torch.nn.init.uniform_(self.bias1, a=-scale, b=scale)

    def reset_parameter(self):
        scale = (2 / self.mode_size)**0.5
        torch.nn.init.normal_(self.W1, std=scale)
        torch.nn.init.normal_(self.W2,  std=scale)
        torch.nn.init.normal_(self.bias1, std=scale)

    def forward(self, x, in_modes1, in_modes2):
        '''
        x : input Fourier Co-efficient
        in_modes1, in_modes2 : Number of modes to mix
        '''
        h_modes1 = min(self.mode_size//2, in_modes1//2)
        h_modes2 = min(self.mode_size//2, in_modes2//2)
        s = in_modes1 % 2
        temp = torch.zeros(x.shape[0], x.shape[1],  h_modes1*2 + s, 2*h_modes2+s,
                           dtype=x.dtype, device=x.device)

        eW1 = torch.zeros(x.shape[1], h_modes1*2+s,
                          2*h_modes2+s, dtype=x.dtype, device=x.device)
        eW2 = torch.zeros(x.shape[1],  h_modes1*2+s,
                          2*h_modes2+s, dtype=x.dtype, device=x.device)
        bias1 = torch.zeros(x.shape[1],  h_modes1*2+s,
                            2*h_modes2+s, dtype=x.dtype, device=x.device)

        temp[:, :, :h_modes1+s, :h_modes2+s] = x[:, :, :h_modes1+s, :h_modes2+s]
        temp[:, :, -h_modes1:, :h_modes2+s] = x[:, :, -h_modes1:, :h_modes2+s]
        temp[:, :, :h_modes1+s, -h_modes2:] = x[:, :, :h_modes1+s, -h_modes2:]
        temp[:, :, -h_modes1:, -h_modes2:] = x[:, :, -h_modes1:, -h_modes2:]

        eW1[:, :h_modes1+s, :h_modes2+s] = self.W1[:, :h_modes1+s, :h_modes2+s]
        eW1[:, -h_modes1:, :h_modes2+s] = self.W1[:, -h_modes1:, :h_modes2+s]
        eW1[:, :h_modes1+s, -h_modes2:] = self.W1[:, :h_modes1+s, -h_modes2:]
        eW1[:, -h_modes1:, -h_modes2:] = self.W1[:, -h_modes1:, -h_modes2:]

        eW2[:, :h_modes1+s, :h_modes2+s] = self.W2[:, :h_modes1+s, :h_modes2+s]
        eW2[:, -h_modes1:, :h_modes2+s] = self.W2[:, -h_modes1:, :h_modes2+s]
        eW2[:, :h_modes1+s, -h_modes2:] = self.W2[:, :h_modes1+s, -h_modes2:]
        eW2[:, -h_modes1:, -h_modes2:] = self.W2[:, -h_modes1:, -h_modes2:]

        bias1[:, :h_modes1+s, :h_modes2 +
              s] = self.bias1[:, :h_modes1+s, :h_modes2+s]
        bias1[:, -h_modes1:, :h_modes2 +
              s] = self.bias1[:, -h_modes1:, :h_modes2+s]
        bias1[:, :h_modes1+s, -h_modes2:] = self.bias1[:, :h_modes1+s, -h_modes2:]
        bias1[:, -h_modes1:, -h_modes2:] = self.bias1[:, -h_modes1:, -h_modes2:]

        mask = get_mixer_mask(h_modes1*2+s, dtype=x.dtype,
                              band_width=self.mixer_band).to(x.device)

        temp = ((eW1 * mask[None, :, :])[None, :, :, :] @ temp)

        temp = (torch.transpose(
            (eW2*mask[None, :, :])[None, :, :, :] @ torch.transpose(temp, -2, -1), -2, -1))

        x[:, :, :h_modes1+s, :h_modes2+s] = temp[:, :, :h_modes1+s, :h_modes2+s]
        x[:, :, -h_modes1:, :h_modes2+s] = temp[:, :, -h_modes1:, :h_modes2+s]
        x[:, :, :h_modes1+s, -h_modes2:] = temp[:, :, :h_modes1+s, -h_modes2:]
        x[:, :, -h_modes1:, -h_modes2:] = temp[:, :, -h_modes1:, -h_modes2:]

        return x


class SpectralMixer_2D_shifteq(nn.Module):
    def __init__(self, num_channels, mode_size, mixer_band=-1, device=None, dtype=torch.float):
        '''
        This is shift equivariant version of the SpectralMixer_2D class.

        num_channels: Number of Co-domain of the input signal
        mode_size: Maximum number of Fourier models to use. Normally can be set to size of the input signal.
        mixer_band: Limits the number of neighboring Fourier models to mix with each other. Deafults to -1 implies no limit on mixing.
        '''
        super(SpectralMixer_2D_shifteq, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.num_channels = num_channels
        self.mode_size = mode_size
        self.W1 = nn.Parameter(torch.empty(
            num_channels, self.mode_size, self.mode_size, **factory_kwargs))
        self.W2 = nn.Parameter(torch.empty(
            num_channels, self.mode_size, self.mode_size, **factory_kwargs))
        self.bias1 = nn.Parameter(torch.empty(
            num_channels, self.mode_size, self.mode_size, **factory_kwargs))
        self.mixer_band = mixer_band
        self.register_buffer(
            "spectral_decay", get_pweights(mode_size, mode_size))
        self.reset_parameter()

    def get_reg_loss(self,):
        return torch.norm(self.spectral_decay * self.W1) + torch.norm(self.spectral_decay * self.W2) + torch.norm(self.bias1)

    def reset_parameter_1(self):
        scale = (1 / self.mode_size)**0.5
        torch.nn.init.uniform_(self.W1, a=-scale, b=scale)
        torch.nn.init.uniform_(self.W2, a=-scale, b=scale)
        torch.nn.init.uniform_(self.bias1, a=-scale, b=scale)

    def reset_parameter(self):
        scale = (2 / self.mode_size)**0.5
        torch.nn.init.normal_(self.W1, std=scale)
        torch.nn.init.normal_(self.W2,  std=scale)
        torch.nn.init.normal_(self.bias1, std=scale)

    def forward(self, x, in_modes1, in_modes2):
        '''
        x : input Fourier Co-efficient
        in_modes1, in_modes2 : Number of modes to mix
        '''
        h_modes1 = min(self.mode_size//2, in_modes1//2)
        h_modes2 = min(self.mode_size//2, in_modes2//2)
        s = in_modes1 % 2
        temp = torch.zeros(x.shape[0], x.shape[1],  h_modes1*2 + s, 2*h_modes2+s,
                           dtype=x.dtype, device=x.device)

        eW1 = torch.zeros(x.shape[1], h_modes1*2+s, 2 *
                          h_modes2+s, dtype=self.W1.dtype, device=x.device)
        eW2 = torch.zeros(x.shape[1],  h_modes1*2+s, 2 *
                          h_modes2+s, dtype=self.W2.dtype, device=x.device)
        bias1 = torch.zeros(
            x.shape[1],  h_modes1*2+s, 2*h_modes2+s, dtype=self.bias1.dtype, device=x.device)

        temp[:, :, :h_modes1+s, :h_modes2+s] = x[:, :, :h_modes1+s, :h_modes2+s]
        temp[:, :, -h_modes1:, :h_modes2+s] = x[:, :, -h_modes1:, :h_modes2+s]
        temp[:, :, :h_modes1+s, -h_modes2:] = x[:, :, :h_modes1+s, -h_modes2:]
        temp[:, :, -h_modes1:, -h_modes2:] = x[:, :, -h_modes1:, -h_modes2:]

        temp_unit = temp.clone()
        temp_unit = temp_unit/(temp_unit.abs()+1e-5)

        temp = temp.abs()

        eW1[:, :h_modes1+s, :h_modes2+s] = self.W1[:, :h_modes1+s, :h_modes2+s]
        eW1[:, -h_modes1:, :h_modes2+s] = self.W1[:, -h_modes1:, :h_modes2+s]
        eW1[:, :h_modes1+s, -h_modes2:] = self.W1[:, :h_modes1+s, -h_modes2:]
        eW1[:, -h_modes1:, -h_modes2:] = self.W1[:, -h_modes1:, -h_modes2:]

        eW2[:, :h_modes1+s, :h_modes2+s] = self.W2[:, :h_modes1+s, :h_modes2+s]
        eW2[:, -h_modes1:, :h_modes2+s] = self.W2[:, -h_modes1:, :h_modes2+s]
        eW2[:, :h_modes1+s, -h_modes2:] = self.W2[:, :h_modes1+s, -h_modes2:]
        eW2[:, -h_modes1:, -h_modes2:] = self.W2[:, -h_modes1:, -h_modes2:]

        bias1[:, :h_modes1+s, :h_modes2 +
              s] = self.bias1[:, :h_modes1+s, :h_modes2+s]
        bias1[:, -h_modes1:, :h_modes2 +
              s] = self.bias1[:, -h_modes1:, :h_modes2+s]
        bias1[:, :h_modes1+s, -h_modes2:] = self.bias1[:, :h_modes1+s, -h_modes2:]
        bias1[:, -h_modes1:, -h_modes2:] = self.bias1[:, -h_modes1:, -h_modes2:]

        mask = get_mixer_mask(h_modes1*2+s, dtype=self.W1.dtype,
                              band_width=self.mixer_band).to(x.device)

        temp = ((eW1 * mask[None, :, :])[None, :, :, :] @ temp)

        temp = (torch.transpose(
            (eW2*mask[None, :, :])[None, :, :, :] @ torch.transpose(temp, -2, -1), -2, -1))

        temp = temp * temp_unit

        x[:, :, :h_modes1+s, :h_modes2+s] = temp[:, :, :h_modes1+s, :h_modes2+s]
        x[:, :, -h_modes1:, :h_modes2+s] = temp[:, :, -h_modes1:, :h_modes2+s]
        x[:, :, :h_modes1+s, -h_modes2:] = temp[:, :, :h_modes1+s, -h_modes2:]
        x[:, :, -h_modes1:, -h_modes2:] = temp[:, :, -h_modes1:, -h_modes2:]

        return x
