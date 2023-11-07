import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Dropout
from utils.core_utils import resample, get_stack
from layers.spectral_conv import SpectralConv2dCircular, SpectralConv2dLocalized, softPaching
from layers.spectral_mixer import SpectralMixer_2D, SpectralMixer_2D_shifteq
from layers.spectral_mlp import spectralMlp
from .core import AbstractBaseClassifierModel
from timm.utils import accuracy
import numpy as np
from layers.complex_modules import SpectralDropout, ComplexBatchNorm1dSim, ComplexBatchNorm2dSim, \
    ComplexLayerNorm1d, ComplexLayerNorm2d, complex_tanh, complex_relu, enblock2d, get_block2d, modRelu
from thirdparty_complex.complexLayers import ComplexConv2d, ComplexMaxPool2d
from math import ceil


class FourierSeqFrequencyMixer(AbstractBaseClassifierModel):
    def __init__(self, in_channel, learning_rate, weight_decay, C1=32, C2=64, C3=128, C4=128, FC1=200, dropout_fc1=0.0,
                 dropout_fc2=0.7, activation_con=complex_tanh, activation_mlp=complex_relu, mixer_band=-1, **kwargs):
        '''
        This scale equivariant model does not use scale equivariant nonlinearity. Rather it mixes Fourier coefficients
        in an scale-equivariant manner and uses nonlinearity in the Fourier domain. This model's features is not translation
        equivariant (unlike convolution layers). This is addressed in the FourierScaleShifteqMixer model.

        Parameters:
        ----------
        in_channel : int, input channels
        learning_rate : float, learning rate
        weight_decay : float, weight decay
        C1, C2, C3, C4, FC1 : int, width of different layers
        dropout_fc1, dropout_fc2: float, dropput rate for fc layers
        activation_con, activation_mlp : complex activation functions
        mixer_band : int, Number of Fourier coefficients to mix,
                                 -1 means all Fourier coefficients are mixed maintaining scale equivariant.
        '''
        super(FourierSeqFrequencyMixer, self).__init__(**kwargs)

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.in_channel = in_channel

        self.L0 = SpectralConv2dLocalized(1, C1, 32, 21)
        self.conv0 = ComplexConv2d(1, C1, 1, bias=False)
        self.M0 = SpectralMixer_2D(C1, 32, mixer_band=mixer_band)
        self.M00 = SpectralMixer_2D(C1, 32, mixer_band=mixer_band)
        self.B0 = ComplexLayerNorm2d(C1, 32)
        self.B00 = ComplexLayerNorm2d(C1, 32)
        self.mp0 = ComplexMaxPool2d(2)

        self.L1 = SpectralConv2dLocalized(C1, C2, 16, 11)
        self.conv1 = ComplexConv2d(C1, C2, 1, bias=False)
        self.M1 = SpectralMixer_2D(C2, 16, mixer_band=mixer_band)
        self.M11 = SpectralMixer_2D(C2, 16, mixer_band=mixer_band)
        self.B1 = ComplexLayerNorm2d(C2, 16)
        self.B11 = ComplexLayerNorm2d(C2, 16)

        self.L2 = SpectralConv2dLocalized(C2, C3, 16, 11)
        self.conv2 = ComplexConv2d(C2, C3, 1, bias=False)
        self.M2 = SpectralMixer_2D(C3, 16, mixer_band=mixer_band)
        self.M22 = SpectralMixer_2D(C3, 16, mixer_band=mixer_band)
        self.B2 = ComplexLayerNorm2d(C3, 16)
        self.B22 = ComplexLayerNorm2d(C3, 16)

        self.L3 = SpectralConv2dLocalized(C3, C4, 16, 11)
        self.conv3 = ComplexConv2d(C3, C4, 1, bias=False)
        self.M3 = SpectralMixer_2D(C4, 16, mixer_band=mixer_band)
        self.M33 = SpectralMixer_2D(C4, 16, mixer_band=mixer_band)
        self.B3 = ComplexLayerNorm2d(C4, 16)
        self.B33 = ComplexLayerNorm2d(C4, 16)

        self.LD1 = SpectralDropout(dropout_fc1, 'complex')
        self.linear_1 = spectralMlp(
            16, 16*C4, FC1, non_linear=False, mixer_band=mixer_band)

        self.LD2 = SpectralDropout(dropout_fc2, 'complex')
        self.LB2 = ComplexLayerNorm1d(FC1, 16)
        self.linear_2 = spectralMlp(
            16, 16*FC1, 10, non_linear=False, mixer_band=mixer_band)

        self.activation_con = activation_con
        self.activation_mlp = activation_mlp
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x_ft = torch.fft.fft2(x, norm='forward').to(
            self.device, dtype=self.L0.L.dtype)

        x0 = self.L0(x_ft)
        xr0 = self.conv0(x_ft)
        x0 = self.M0(x0, x_ft.shape[-2], x_ft.shape[-1])
        x0 = self.B0(x0)
        x0 = self.activation_con(x0)
        x0 = self.M00(x0, x_ft.shape[-2], x_ft.shape[-1])
        x0 = x0+xr0
        x0 = self.B00(x0)
        x0 = self.activation_con(x0)
        x0 = self.mp0(x0)

        x1 = self.L1(x0)
        xr1 = self.conv1(x0)
        x1 = self.M1(x1, x0.shape[-2], x0.shape[-1])
        x1 = self.B1(x1)
        x1 = self.activation_con(x1)
        x1 = self.M11(x1, x0.shape[-2], x0.shape[-1])
        x1 = x1 + xr1
        x1 = self.B11(x1)
        x1 = self.activation_con(x1)

        x2 = self.L2(x1)
        xr2 = self.conv2(x1)
        x2 = self.M2(x2, x1.shape[-2], x1.shape[-1])
        x2 = self.B2(x2)
        x2 = self.activation_con(x2)
        x2 = self.M22(x2, x1.shape[-2], x1.shape[-1])
        x2 = x2 + xr2
        x2 = self.B22(x2)
        x2 = self.activation_con(x2)

        x3 = self.L3(x2)
        xr3 = self.conv3(x2)
        x3 = self.M3(x3, x2.shape[-2], x2.shape[-1])
        Modes = x2.shape[-1]
        x3 = self.B3(x3, modes=Modes)
        x3 = self.activation_con(x3)
        x3 = self.M33(x3,  x2.shape[-2], x2.shape[-1])
        x3 = x3 + xr3
        x3 = self.B33(x3, modes=Modes)
        x3 = self.activation_con(x3)

        Modes = x3.shape[-1]
        x3 = enblock2d(x3, self.L3.global_modes)
        fe = torch.diagonal(x3, dim1=-2, dim2=-1)

        fe = fe.reshape(fe.shape[0], -1)
        fe = self.LD1(fe)
        f1 = self.linear_1(fe)

        f1 = torch.squeeze(f1)
        f1 = self.LB2(f1, modes=Modes)
        f1 = self.activation_con(f1)
        f1 = f1.reshape(f1.shape[0], -1)

        f1 = self.LD2(f1)

        f2 = self.linear_2(f1)

        return torch.real(f2[:, :, :, :])

    def get_feature(self, x):

        # returning some internal features
        x_ft = torch.fft.fft2(x, norm='forward').to(
            self.device, dtype=self.L0.L.dtype)

        x0 = self.L0(x_ft)
        xr0 = self.conv0(x_ft)
        x0 = self.M0(x0, x_ft.shape[-2], x_ft.shape[-1])
        x0 = self.B0(x0)
        x0 = self.activation_con(x0)
        x0 = self.M00(x0, x_ft.shape[-2], x_ft.shape[-1])
        x0l = x0+xr0
        x0 = self.B00(x0l)
        x0 = self.activation_con(x0)

        return torch.fft.ifft2(x0, norm="forward")

    def get_weight_group(self,):
        weight_group = []
        for L in self.children():
            if hasattr(L, 'norm'):
                # print("it is the nonLin layer")
                for i in L.norm.keys():
                    weight_dec = (int(i)/self.keep_size)**2 * self.weight_decay
                    weight_group.append(
                        {'params': L.norm[i].parameters(), 'weight_decay': weight_dec})
            elif hasattr(L, 'keys'):
                # print('This is module dict')
                for i in L.keys():
                    weight_dec = (int(i)/self.keep_size)**2 * self.weight_decay
                    weight_group.append(
                        {'params': L[i].parameters(), 'weight_decay': weight_dec})
            elif hasattr(L, 'L'):
                # weight decay is 0, as it is handled manually
                weight_group.append(
                    {'params': L.parameters(), 'weight_decay': 0.0})
            else:
                weight_group.append({'params': L.parameters()})
        return weight_group

    def _forward_step(self, batch, batch_idx, stage='train', sync_dist=False):
        x, y = batch

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
        scaling_factor = output.shape[-1] * 0.5
        f_lim = min(output.shape[-1]//2 + 1, int(x.shape[-1]/4)+1)
        for j in range(0, f_lim):
            if j == 0:
                logits = output[:, :, 0, j]
                Ensemble = torch.clone(logits)
            else:
                logits = (output[:, :, 0, j] +
                          output[:, :, 0, -j])/2 + Ensemble
                if j >= 2:
                    if loss is None:
                        loss = self.criterion(logits, y) * (1/f_lim)
                    else:
                        loss += self.criterion(logits, y) * (1/f_lim)
                    with torch.no_grad():
                        Ensemble = logits.clone().detach()
                else:
                    Ensemble = torch.clone(logits)

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
        _, loss = self._forward_step(
            batch, batch_idx, stage='val', sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        """Test step."""
        _, loss = self._forward_step(
            batch, batch_idx, stage='test', sync_dist=True)
        return loss


class FourierScaleShifteqMixer(AbstractBaseClassifierModel):
    def __init__(self, in_channel, learning_rate, weight_decay, C1=32, C2=64, C3=128, C4=128, FC1=200, dropout_fc1=0.0,
                 dropout_fc2=0.7, activation_con=complex_tanh, activation_mlp=nn.functional.relu, mixer_band=-1,  **kwargs):
        '''
        The model is uses a translation and scale equivarinat Fourier mixer.
        parameters
        ----------
        Same as the FourierSeqFrequencyMixer class.
        '''
        super(FourierScaleShifteqMixer, self).__init__(**kwargs)
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.in_channel = in_channel

        normalizer_2d = ComplexLayerNorm2d
        normalizer_1d = ComplexLayerNorm1d
        mixer = SpectralMixer_2D_shifteq

        self.L0 = SpectralConv2dLocalized(1, C1, 32, 11)
        self.conv0 = ComplexConv2d(1, C1, 1, bias=False)
        self.M0 = mixer(C1, 32, mixer_band=mixer_band)
        self.M00 = mixer(C1, 32, mixer_band=mixer_band)
        self.B0 = normalizer_2d(C1, 32)
        self.B00 = normalizer_2d(C1, 32)
        self.activation_con0 = modRelu()
        self.activation_con01 = modRelu()

        self.L1 = SpectralConv2dLocalized(C1, C2, 32, 11)
        self.conv1 = ComplexConv2d(C1, C2, 1, bias=False)
        self.M1 = mixer(C2, 32, mixer_band=mixer_band)
        self.M11 = mixer(C2, 32, mixer_band=mixer_band)
        self.B1 = normalizer_2d(C2, 32)
        self.B11 = normalizer_2d(C2, 32)
        self.activation_con1 = modRelu()
        self.activation_con11 = modRelu()

        self.L2 = SpectralConv2dLocalized(C2, C3, 32, 11)
        self.conv2 = ComplexConv2d(C2, C3, 1, bias=False)
        self.M2 = mixer(C3, 32, mixer_band=mixer_band)
        self.M22 = mixer(C3, 32, mixer_band=mixer_band)
        self.B2 = normalizer_2d(C3, 32)
        self.B22 = normalizer_2d(C3, 32)
        self.activation_con2 = modRelu()
        self.activation_con21 = modRelu()

        self.L3 = SpectralConv2dLocalized(C3, C4, 32, 11)
        self.conv3 = ComplexConv2d(C3, C4, 1, bias=False)
        self.M3 = mixer(C4, 32, mixer_band=mixer_band)
        self.M33 = mixer(C4, 32, mixer_band=mixer_band)
        self.B3 = normalizer_2d(C4, 32)
        self.B33 = normalizer_2d(C4, 32)
        self.activation_con3 = modRelu()
        self.activation_con31 = modRelu()

        self.LD1 = Dropout(dropout_fc1)
        self.linear_1 = spectralMlp(
            32, 32*C4, FC1, non_linear=False, dtype=torch.float, mixer_band=mixer_band)

        self.LD2 = Dropout(dropout_fc2)
        self.LB2 = normalizer_1d(FC1, 32, complex=False)
        self.linear_2 = spectralMlp(
            32, 32*FC1, 10, non_linear=False,  dtype=torch.float, mixer_band=mixer_band)

        self.activation_con = activation_con
        self.activation_mlp = activation_mlp
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x_ft = torch.fft.fft2(x, norm='forward').to(
            self.device, dtype=self.L0.L.dtype)

        x0 = self.L0(x_ft)
        xr0 = self.conv0(x_ft)
        x0 = self.M0(x0, x_ft.shape[-2], x_ft.shape[-1])
        x0 = self.B0(x0)
        x0 = self.activation_con0(x0.clone())
        x0 = self.M00(x0, x_ft.shape[-2], x_ft.shape[-1])
        x0 = x0+xr0
        x0 = self.B00(x0)
        x0 = self.activation_con01(x0)

        x1 = self.L1(x0)
        xr1 = self.conv1(x0)
        x1 = self.M1(x1, x0.shape[-2], x0.shape[-1])
        x1 = self.B1(x1)
        x1 = self.activation_con1(x1)
        x1 = self.M11(x1, x0.shape[-2], x0.shape[-1])
        x1 = x1 + xr1
        x1 = self.B11(x1)
        x1 = self.activation_con11(x1)

        x2 = self.L2(x1)
        xr2 = self.conv2(x1)
        x2 = self.M2(x2, x1.shape[-2], x1.shape[-1])
        x2 = self.B2(x2)
        x2 = self.activation_con2(x2)
        x2 = self.M22(x2, x1.shape[-2], x1.shape[-1])
        x2 = x2 + xr2
        x2 = self.B22(x2)
        x2 = self.activation_con21(x2)

        x3 = self.L3(x2)
        xr3 = self.conv3(x2)
        x3 = self.M3(x3, x2.shape[-2], x2.shape[-1])
        x3 = self.B3(x3)
        x3 = self.activation_con3(x3)
        x3 = self.M33(x3,  x2.shape[-2], x2.shape[-1])
        x3 = x3 + xr3
        x3 = self.B33(x3)
        x3 = self.activation_con31(x3)

        Modes = x3.shape[-1]
        x3 = enblock2d(x3, self.L3.global_modes)
        fe = torch.diagonal(x3, dim1=-2, dim2=-1).abs()

        fe = fe.reshape(fe.shape[0], -1)
        fe = self.LD1(fe)
        f1 = self.linear_1(fe)

        f1 = torch.squeeze(f1)
        f1 = self.LB2(f1, modes=Modes)
        f1 = self.activation_mlp(f1)
        f1 = f1.reshape(f1.shape[0], -1)

        f1 = self.LD2(f1)

        f2 = self.linear_2(f1)

        return f2[:, :, :, :]

    def get_feature(self, x):

        x_ft = torch.fft.fft2(x, norm='forward').to(
            self.device, dtype=self.L0.L.dtype)

        x0 = self.L0(x_ft)
        xr0 = self.conv0(x_ft)
        x0 = self.M0(x0, x_ft.shape[-2], x_ft.shape[-1])
        x0 = self.B0(x0)
        x0 = self.activation_con(x0.clone())
        x0 = self.M00(x0.clone(), x_ft.shape[-2], x_ft.shape[-1])
        x0 = x0+xr0
        x0 = self.B00(x0)
        x0 = self.activation_con(x0.clone())

        x1 = self.L1(x0)
        xr1 = self.conv1(x0)
        x1 = self.M1(x1, x0.shape[-2], x0.shape[-1])
        x1 = self.B1(x1)
        x1 = self.activation_con(x1)
        x1 = self.M11(x1, x0.shape[-2], x0.shape[-1])
        x1 = x1 + xr1
        x1 = self.B11(x1)
        x1 = self.activation_con(x1)

        x2 = self.L2(x1)
        xr2 = self.conv2(x1)
        x2 = self.M2(x2, x1.shape[-2], x1.shape[-1])
        x2 = self.B2(x2)
        x2 = self.activation_con2(x2)
        x2 = self.M22(x2, x1.shape[-2], x1.shape[-1])
        x2 = x2 + xr2
        x2 = self.B22(x2)
        x2 = self.activation_con21(x2)

        x3 = self.L3(x2)
        xr3 = self.conv3(x2)
        x3 = self.M3(x3, x2.shape[-2], x2.shape[-1])
        x3 = self.B3(x3)
        x3 = self.activation_con3(x3)
        x3 = self.M33(x3,  x2.shape[-2], x2.shape[-1])
        x3 = x3 + xr3
        x3 = self.B33(x3)
        x3 = self.activation_con31(x3)

        return torch.fft.ifft2(x3, norm="forward")

    def get_weight_group(self,):
        weight_group = []
        for L in self.children():
            if hasattr(L, 'norm'):
                # print("it is the nonLin layer")
                for i in L.norm.keys():
                    weight_dec = (int(i)/self.keep_size)**2 * self.weight_decay
                    weight_group.append(
                        {'params': L.norm[i].parameters(), 'weight_decay': weight_dec})
            elif hasattr(L, 'keys'):
                # print('This is module dict')
                for i in L.keys():
                    weight_dec = (int(i)/self.keep_size)**2 * self.weight_decay
                    weight_group.append(
                        {'params': L[i].parameters(), 'weight_decay': weight_dec})
            elif hasattr(L, 'L'):
                # weight decay is 0, as it is handled manually
                weight_group.append(
                    {'params': L.parameters(), 'weight_decay': 0.0})
            else:
                weight_group.append({'params': L.parameters()})
        return weight_group

    def _forward_step(self, batch, batch_idx, stage='train', sync_dist=False):
        x, y = batch

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
        scaling_factor = output.shape[-1] * 0.5
        f_lim = min(output.shape[-1]//2 + 1, int(x.shape[-1]/4)+1)
        for j in range(0, f_lim):
            if j == 0:
                logits = output[:, :, 0, j]
                Ensemble = torch.clone(logits)
            else:
                logits = (output[:, :, 0, j] +
                          output[:, :, 0, -j])/2 + Ensemble
                if j >= 2:
                    if loss is None:
                        loss = self.criterion(logits, y) * (1/f_lim)
                    else:
                        loss += self.criterion(logits, y) * (1/f_lim)
                    with torch.no_grad():
                        Ensemble = logits.clone().detach()
                else:
                    Ensemble = torch.clone(logits)

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
