import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.core_utils import get_mixer_mask, ComplexRELU, ComplexLeakyRELU, ComplexSELU, ComplexGELU, get_pweights


class spectralMlp(nn.Module):
    '''
    scale equivarinat MLP layer in Fourier Domain
    modes: Number of Fourier Modes per channe;
    num_feature: total number of feature = modes x num_in_channels
    out_channels: Number of output channels
    mixer_band: Maximum number of Fourier modes to mix with each other
    '''

    def __init__(self, modes, num_feature, out_channel, non_linear, mixer_band=-1, device=None, dtype=torch.cfloat):
        super(spectralMlp, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        assert num_feature % modes == 0
        self.modes = modes
        self.num_feature = num_feature
        self.out_channel = out_channel
        self.W = nn.Parameter(torch.empty(
            out_channel, modes, num_feature, **factory_kwargs))
        self.register_buffer("mask", get_mixer_mask(
            modes, factory_kwargs['dtype'], band_width=mixer_band))
        self.register_buffer("spectral_decay", get_pweights(modes, modes))
        if non_linear:
            self.non_lin = ComplexGELU()
        else:
            self.non_lin = None
        self.reset_parameter()

    def reset_parameter_1(self):
        scale = 1/(self.num_feature)**0.5
        torch.nn.init.uniform_(self.W, a=-scale, b=scale)

    def reset_parameter(self):
        scale = 2/(self.num_feature+self.modes)
        torch.nn.init.normal_(self.W, std=scale**0.5)

    def get_reg_loss(self,):
        return torch.norm(self.spectral_decay.repeat(1, self.num_feature//self.modes)[None, :, :] * self.W)

    def forward(self, x):
        '''
        x: is feature vectors of shape (batch_size, in_channels x Frequency modes per channel) 
        '''
        out = x[:, None, None, :] @ torch.transpose(self.W * self.mask.repeat(
            1, self.num_feature//self.modes)[None, :, :], -2, -1)
        out = out
        if self.non_lin is not None:
            out = self.non_lin(out)

        return out
