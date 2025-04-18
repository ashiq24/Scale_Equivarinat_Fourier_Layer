import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Dropout
from scale_eq.utils.core_utils import *
from scale_eq.layers.spectral_conv import SpectralConvLocalized1d
from scale_eq.layers.spectral_mixer import SpectralMixer_1D
from scale_eq.layers.spectral_mlp import spectralMlp
from scale_eq.layers.scalq_eq_nonlin import scaleEqNonlin1d
from .core import AbstractBaseClassifierModel
from timm.utils import accuracy
import numpy as np
from scale_eq.layers.complex_modules import SpectralDropout, ComplexBatchNorm1dSim, ComplexBatchNorm2dSim, \
    ComplexLayerNorm1d, ComplexLayerNorm2d, complex_tanh, complex_relu, enblock2d, get_block2d, modRelu, relu
from scale_eq.thirdparty_complex.complexLayers import ComplexConv2d, ComplexMaxPool2d, ComplexConv1d
from math import ceil


class FourierSeq1d(AbstractBaseClassifierModel):
    def __init__(self, in_channel, learning_rate, weight_decay, C1=32, C2=64, C3=128, C4=128, FC1=200, dropout_fc1=0.0,
                 dropout_fc2=0.7, activation_con=relu, activation_mlp=relu, mixer_band=-1, normalizer='batch', **kwargs):
        super(FourierSeq1d, self).__init__(**kwargs)

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.in_channel = in_channel

        self.L0 = SpectralConvLocalized1d(1, C1, 28, 17)
        self.NL0 = scaleEqNonlin1d(activation_con, 8, normalizer, C1)

        self.L1 = SpectralConvLocalized1d(C1, C2, 28, 11)
        self.NL1 = scaleEqNonlin1d(activation_con, 8, normalizer, C2)

        self.L2 = SpectralConvLocalized1d(C2, C3, 28, 11)
        self.NL2 = scaleEqNonlin1d(activation_con, 8, normalizer, C3)

        self.L3 = SpectralConvLocalized1d(C3, C4, 28, 11)
        self.NL3 = scaleEqNonlin1d(activation_con, 8, normalizer, C4)

        self.res_dict = nn.Sequential(
            Dropout(dropout_fc1),
            nn.Linear(C4, FC1),
            nn.ReLU(),
            Dropout(dropout_fc2),
            nn.LayerNorm(FC1),
            nn.Linear(FC1, 1)
        )
        self.activation_con = activation_con
        self.activation_mlp = activation_mlp
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def apply_stack(self, x):
        f = None
        for i in range(0, x.shape[1]):
            res = i+8
            j = self.res_dict(x[:, i, :])
            if f is None:
                f = j[:, None, :]
            else:
                f = torch.cat((f, j[:, None, :]), dim=1)
        return f

    def get_weight_group(self,):
        weight_group = []
        for L in self.children():
            weight_group.append({'params': L.parameters()})
        return weight_group

    def forward(self, x):

        x_ft = torch.fft.fft(x, norm='forward').to(
            self.device, dtype=self.L0.L.dtype)

        x0 = self.L0(x_ft)
        x0 = self.NL0(x0)

        x1 = self.L1(x0)
        x1 = self.NL1(x1)

        x2 = self.L2(x1)
        x2 = self.NL2(x2)

        x3 = self.L3(x2)
        x3 = self.NL3(x3)

        # instead of taking the absolute value of Fourier coefficients
        # we can also resample the Funtion on a fix grid by iFFT

        fe = torch.abs(x3)
        fe = fe.mean(dim=2)

        f2 = self.res_dict(fe)

        return f2

    def get_feature(self, x, idx=0):

        x_ft = torch.fft.fft2(x, norm='forward').to(
            self.device, dtype=self.L0.L.dtype)

        x0 = self.L0(x_ft)
        xr0 = self.conv0(x_ft)
        x0 = x0+xr0
        x0 = self.NL0(x0)

        x1_0 = self.L1(x0)
        xr1_1 = self.conv1(x0)
        x1_2 = x1_0+xr1_1
        x1_4 = self.NL1(x1_2)

        x2 = self.L2(x1_4)
        xr2 = self.conv2(x1_4)
        x2 = x2+xr2
        x2 = self.NL2(x2)

        k = [x0, x1_0, xr1_1, x1_2, x1_4]
        return torch.fft.ifft(k[idx], norm="forward")

    def _forward_step(self, batch, batch_idx, stage='train', sync_dist=False):
        x, _, y = batch
        y = torch.tensor(y, dtype=torch.float32).to(self.device)[None, ...]
        # making it a binary classification task
        x = x[:, :, 6000:8000]
        y = y[:, 0:1]
        reg_loss = 0
        for L in self.modules():
            if hasattr(L, 'W1'):
                reg_loss += L.get_reg_loss()
            elif hasattr(L, 'L'):
                reg_loss += L.get_reg_loss()
            elif hasattr(L, 'W'):
                reg_loss += L.get_reg_loss()
            elif hasattr(L, 'weight') and L.weight is not None:
                reg_loss += torch.norm(L.weight)
            elif hasattr(L, 'bias') and L.bias is not None:
                reg_loss += torch.norm(L.bias)

        output = self(x)
        Ensemble = output
        loss = None
        loss = self.criterion(output, y) + self.weight_decay * reg_loss

        acc = accuracy(Ensemble, y)[0]/100

        if stage != "evaluate":
            self.log('%s_loss' % stage, loss, on_step=True,
                     on_epoch=True, logger=True, sync_dist=sync_dist, prog_bar=stage == 'val')
            self.log('%s_acc' % stage, acc, on_step=True,
                     on_epoch=True, logger=True, sync_dist=sync_dist, prog_bar=stage == 'val')
        return Ensemble, loss

    def _forward_step_(self, batch, batch_idx, stage='train', sync_dist=False):
        x, _, y = batch
        y = torch.tensor(y, dtype=torch.float32).to(self.device)[None, ...]

        # making it a binary classification task

        x = x[:, :, 6000:8000]
        y = y[:, 0:1]

        reg_loss = 0
        for L in self.modules():
            if hasattr(L, 'W1'):
                reg_loss += L.get_reg_loss()
            elif hasattr(L, 'L'):
                reg_loss += L.get_reg_loss()
            elif hasattr(L, 'W'):
                reg_loss += L.get_reg_loss()
            elif hasattr(L, 'weight') and L.weight is not None:
                reg_loss += torch.norm(L.weight)
            elif hasattr(L, 'bias') and L.bias is not None:
                reg_loss += torch.norm(L.bias)

        output = self(x)
        Ensemble = None
        loss = None

        for j in range(output.shape[1]):
            if j == 0:
                logits = output[:, j, :]
                if loss is None:
                    loss = self.criterion(logits, y) * (1/output.shape[1])
                with torch.no_grad():
                    Ensemble = logits.clone().detach()
            else:
                logits = output[:, j, :] + Ensemble

                loss += self.criterion(logits, y) * (1/output.shape[1])
                with torch.no_grad():
                    Ensemble = logits.clone().detach()

        loss = loss + self.weight_decay * reg_loss

        acc = accuracy(Ensemble, y)[0]/100

        if stage != "evaluate":
            self.log('%s_loss' % stage, loss, on_step=True,
                     on_epoch=True, logger=True, sync_dist=sync_dist, prog_bar=stage == 'val')
            self.log('%s_acc' % stage, acc, on_step=True,
                     on_epoch=True, logger=True, sync_dist=sync_dist, prog_bar=stage == 'val')
        return Ensemble, loss

    def training_step(self, batch, batch_idx):
        """Training step."""
        _, loss = self._forward_step(
            batch, batch_idx, stage='train', sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        # print("Validation step is being called")
        _, loss = self._forward_step(
            batch, batch_idx, stage='val', sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        _, loss = self._forward_step(
            batch, batch_idx, stage='test', sync_dist=True)
        return loss
