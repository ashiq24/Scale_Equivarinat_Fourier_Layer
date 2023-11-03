import torch
from torch import nn
import random
import torch.nn.functional as F
from torch.nn import Dropout 
from utils.core_utils import resample, get_stack, get_pooled, get_stack_differential
from layers.spectral_conv import SpectralConv2dCircular, SpectralConv2dLocalized, softPaching
from layers.spectral_mixer import SpectralMixer_2D, SpectralMixer_2D_shifteq
from layers.spectral_mlp import spectralMlp
from layers.scalq_eq_nonlin import scaleEqNonlin, scaleEqNonlinMaxp
from .core import AbstractBaseClassifierModel
from timm.utils import accuracy
import numpy as np
import math
from layers.complex_modules import SpectralDropout, ComplexBatchNorm1dSim, ComplexBatchNorm2dSim, \
  ComplexLayerNorm1d, ComplexLayerNorm2d, complex_tanh, complex_relu, enblock2d, get_block2d, modRelu, relu
from thirdparty_complex.complexLayers import ComplexConv2d, ComplexMaxPool2d
from math import ceil

class FourierSeqStl(AbstractBaseClassifierModel):
  def __init__(self, in_channel, learning_rate, weight_decay,C1 = 32, C2 = 64,C3 = 128, C4 = 128, FC1 = 200, dropout_fc1 = 0.0,\
               dropout_fc2 = 0.7, activation_con = relu, activation_mlp = relu, mixer_band = -1, normalizer = 'instance', 
               increment = None, max_res =None, base_res = None, pool_size = None, **kwargs):
    '''
    Model implementation of Fourier Scale Equivariant Model with Scale Equivariant Nonlinearities of STL dataset.

    parameters
    ----------
    in_channel : int, nummber of input channels
    output_channel : int, number of output channels
    learning_rate : float, learning rate
    weight_decay : float, weight decay
    C1, C2, C3, C4 : int, width of the 1st, 2nd, 3rd and 4th layers
    FC1 : int, width of the fully connected layer
    dropout_fc1, dropout_fc2  : float, dropout rate of the fully connected layer
    activation_con, activation_mlp : str, activation function of the convolutional layers and the fully connected layer
    normalizer : str, normalization type of the convolutional layers
    incremnt: int, increment of the NonLinear layers
    max_res : int, maximum resolution under consideration
    base_res : int, minimum resolution under consideration
    pool_size : int, size of the pooling layer for final features before fattenting.
    '''
    super(FourierSeqStl, self).__init__(**kwargs)
    # Log hyperparameters
    self.save_hyperparameters()
    self.learning_rate = learning_rate
    self.in_channel = in_channel

    self.pool_size = pool_size
    self.keep_size  = max_res
    self.increment = increment
    self.base_res = base_res
    self.projector = ComplexConv2d(self.in_channel,C1,1, bias= False)
    self.PN =  scaleEqNonlin(activation_con, self.base_res, normalizer,C1,max_res=self.keep_size, increment= self.increment)

    self.L0_1 = SpectralConv2dLocalized(C1, C2, 64, 7)
    self.L0_2 = SpectralConv2dLocalized(C2, C2, 64, 7)
    self.conv0 = ComplexConv2d(C1,C2,1, bias= False)
    self.NL0_1 = scaleEqNonlin(activation_con, self.base_res,normalizer,C2, max_res=self.keep_size,increment= self.increment)
    self.NL0_2 = scaleEqNonlinMaxp(activation_con, self.base_res,normalizer,C2, max_res=self.keep_size, increment=self.increment)
    
    

    self.L1_1 = SpectralConv2dLocalized(C2, C3, 32, 5)
    self.L1_2 = SpectralConv2dLocalized(C3, C3, 32, 5)
    self.conv1 = ComplexConv2d(C2,C3,1, bias= False)
    self.NL1_1 = scaleEqNonlin(activation_con, math.ceil(self.base_res/2),normalizer,C3, max_res=math.ceil(self.keep_size/2), increment= self.increment//2)
    self.NL1_2 = scaleEqNonlinMaxp(activation_con, math.ceil(self.base_res/2),normalizer,C3, max_res=math.ceil(self.keep_size/2), increment=self.increment//2)


    self.L2_1 = SpectralConv2dLocalized(C3, C4, 16, 5)
    self.L2_2 = SpectralConv2dLocalized(C4, C4, 16, 5)
    self.conv2 = ComplexConv2d(C3,C4,1, bias= False)
    self.NL2_1 = scaleEqNonlin(activation_con, math.ceil(self.base_res/4),normalizer,C4, max_res=math.ceil(self.keep_size/4),increment= self.increment//4)
    self.NL2_2 = scaleEqNonlin(activation_con, math.ceil(self.base_res/4),normalizer,C4, max_res=math.ceil(self.keep_size/4),increment=self.increment//4)

    self.res_dict = nn.Sequential(
          nn.LayerNorm(C4* math.ceil((self.keep_size/4)/self.pool_size)**2 ),
          Dropout(dropout_fc1),
          nn.Linear(C4* math.ceil((self.keep_size/4)/self.pool_size)**2 ,FC1),
          Dropout(dropout_fc2),
          nn.LayerNorm(FC1),
          nn.ReLU(),
          nn.Linear(FC1, 10)
      )
    
    self.activation_con = activation_con
    self.activation_mlp = activation_mlp
    self.weight_decay = weight_decay
    self.criterion = nn.CrossEntropyLoss()
  def apply_block(self, x, L1,L2,N1,N2,C1):
    xr = C1(x)

    x1 = L1(x)
    x1 = N1(x1)
    x1 = L2(x1)
    x_out = xr + x1
    x_out = N2(x_out)
    return x_out
  
  def get_block_activation(self, x,L1,L2,N1,N2,C1, idx):
    xr = C1(x)
    x1 = L1(x)
    x2 = N1(x1)
    x3 = L2(x2)
    x_out = xr + x3
    x4 = N2(x_out)
    k = [x1, x2, x3, x4]
    return torch.fft.ifft2(k[idx],norm = "forward")

   
  def apply_stack(self,x, increment):
    f = None
    for i in range(0,x.shape[1]):
      res = (increment*i)+self.base_res
      j = self.res_dict[str(res)](x[:,i,:])
      if f is None:
        f = j[:,None,:]
      else:
        f = torch.cat((f, j[:,None,:]), dim = 1)
    return f
  
  def apply_stack_shared(self,x):
    f = None
    for i in range(0,x.shape[1]):
      j = self.res_dict(x[:,i,:])
      if f is None:
        f = j[:,None,:]
      else:
        f = torch.cat((f, j[:,None,:]), dim = 1)
    return f
  
  def forward(self, x):
    
    x_ft = torch.fft.fft2(x, norm = 'forward').to( self.device, dtype = self.L0_1.L.dtype)
    
    x_ft = self.projector(x_ft)
    x_ft = self.PN(x_ft)

    x1 = self.apply_block(x_ft, self.L0_1, self.L0_2, self.NL0_1, self.NL0_2, self.conv0)
    x2 = self.apply_block(x1, self.L1_1, self.L1_2, self.NL1_1, self.NL1_2, self.conv1)
    x3 = self.apply_block(x2, self.L2_1, self.L2_2, self.NL2_1, self.NL2_2, self.conv2)

    fe = get_stack(x3, self.pool_size, math.ceil(self.keep_size/4),base_res=self.base_res//4, increment= self.increment//4)


    f2 = self.apply_stack_shared(fe)    
    return f2
  
  def get_feature(self, x, idx = -1):
    x_ft = torch.fft.fft2(x, norm = 'forward').to( self.device, dtype = self.L0_1.L.dtype)
    
    x_ft = self.projector(x_ft)
    x_ft = self.PN(x_ft)

    x1 = self.apply_block(x_ft, self.L0_1, self.L0_2, self.NL0_1, self.NL0_2, self.conv0)
    x2 = self.apply_block(x1, self.L1_1, self.L1_2, self.NL1_1, self.NL1_2, self.conv1)
    x3 = self.apply_block(x2, self.L2_1, self.L2_2, self.NL2_1, self.NL2_2, self.conv2)

    k = [x_ft, x1, x2, x3]
    return torch.fft.ifft2(k[idx],norm = "forward")
  
  def get_block_feature(self, x, idx =-1):
    x_ft = torch.fft.fft2(x, norm = 'forward').to( self.device, dtype = self.L0_1.L.dtype)
    
    x_ft = self.projector(x_ft)
    x_ft = self.PN(x_ft)

    return  self.get_block_activation(x_ft, self.L0_1, self.L0_2, self.NL0_1, self.NL0_2, self.conv0, idx)
    
  
  def get_loss(self, output, y):
    loss = None
    prev_loss = None
    b_loss = None
    for j in range(output.shape[1]):
      if j == 0:
        logits = output[:,j,:]
        if loss is None:
            loss = self.criterion(logits, y)
            prev_loss = loss.clone()
        # with torch.no_grad():
        #   Ensemble = logits.clone().detach()
      else:
        logits = output[:,j,:]
        c_loss  = self.criterion(logits, y)
        loss += c_loss
        if b_loss is None:
          b_loss = torch.max(c_loss - prev_loss, torch.tensor(0.0)).to(output.device)
        else:
          b_loss += torch.max(c_loss - prev_loss, torch.tensor(0.0)).to(output.device)
        prev_loss = c_loss.clone()

      if j == output.shape[1] -1:
        with torch.no_grad():
          Ensemble = logits.clone().detach()
          
    if b_loss is None:
      b_loss = torch.tensor(0.0).to(output.device)

    return loss,b_loss, Ensemble
  
  def get_weight_group(self,):
    weight_group = []
    for L in self.children():
      if hasattr(L, 'norm'):
        #print("it is the nonLin layer")
        for i in L.norm.keys():
          weight_dec = (int(i)/self.keep_size)**2 *self.weight_decay
          weight_group.append({'params': L.norm[i].parameters(), 'weight_decay': weight_dec})
      elif hasattr(L,'keys'):
        #print('This is module dict')
        for i in L.keys():
          weight_dec = (int(i)/self.keep_size)**2 *self.weight_decay
          weight_group.append({'params': L[i].parameters(), 'weight_decay': weight_dec})
      elif hasattr(L,'L'):
        #weight decay is 0, as it is handled manually
        weight_group.append({'params': L.parameters(), 'weight_decay': 0.0})
      else:
        weight_group.append({'params': L.parameters()})
    return weight_group
        


  def _forward_step(self, batch, batch_idx, stage='train', sync_dist=False):
    x, y = batch

    reg_loss = 0
    for L in self.modules():
      if hasattr(L, 'L'):
        reg_loss += L.get_reg_loss()
     
    output = self(x)
    Ensemble = None
    
    loss ,b_loss, Ensemble = self.get_loss(output, y)

    loss = loss + self.weight_decay * 0.001 * reg_loss + 0.001*b_loss

    acc = accuracy(Ensemble, y)[0]/100

    if stage != "evaluate":
      self.log('%s_loss' % stage, loss, on_step=True,
                on_epoch=True, logger=True, sync_dist=sync_dist, prog_bar=stage=='val')
      self.log('%s_acc' % stage, acc, on_step=True,
                on_epoch=True, logger=True, sync_dist=sync_dist, prog_bar=stage=='val')
      self.log('%s_b_loss' % stage, b_loss, on_step=True,
                on_epoch=True, logger=True, sync_dist=sync_dist, prog_bar=stage=='val')
    return Ensemble, loss

  def training_step(self, batch, batch_idx):
    """Training step."""
    _, loss = self._forward_step(
        batch, batch_idx, stage='train', sync_dist=False)
    return loss

  def validation_step(self, batch, batch_idx):
    """Validation step."""
    #print("Validation step is being called")
    _, loss = self._forward_step(
        batch, batch_idx, stage='val', sync_dist=True)
    return loss

  def test_step(self, batch, batch_idx):
    """Test step."""
    _, loss = self._forward_step(
        batch, batch_idx, stage='test', sync_dist=True)
    return loss
