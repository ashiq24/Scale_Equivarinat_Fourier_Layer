import torch
import torch.nn as nn
import math
import random
from torch.nn import Module, Parameter, init
from torch.nn.functional import relu, max_pool2d, avg_pool2d, dropout, dropout2d, interpolate, sigmoid, tanh, gelu,selu

class SpectralDropout(nn.Module):
    '''
    dropout layer for Fourier Features.
    '''
    def __init__(self, p, mode) -> None:
        super(SpectralDropout, self).__init__()

        self.p = p
        self.mode = mode
    
    def forward(self,x):

        if self.training:
            if self.mode == 'complex':
                mask = nn.functional.dropout(torch.ones_like(x.real), p = self.p)
                x = x * mask
            elif random.random()<self.p:
                if self.mode == '1D':
                        s_size = x.shape[-1]
                        d_size = math.floor(random.random()*s_size//2)
                        m = torch.ones(s_size, device=x.device)
                        m[s_size//2-d_size:s_size//2+d_size] = 0
                        x = x * m[None,None,:]

                else:
                    s_size = x.shape[-1]
                    d_size = math.floor(random.random()*s_size//2)
                    m = torch.ones(s_size, device=x.device)
                    m[s_size//2-d_size:s_size//2+d_size] = 0
                    x = x * m[None,None,None,:]
                    x = x * m[None,None,:,None]
        
        return x

class ComplexAvgPool2d(nn.Module):
  def __init__(self, kernel_size, **kwargs) -> None:
    super().__init__()
    self.avg_p = nn.AvgPool2d(kernel_size= kernel_size, **kwargs)
  
  def forward(self, x):
     
    return self.avg_p(torch.real(x)) + 1.0j * self.avg_p(torch.imag(x))
  
class ComplexMaxPool2d(nn.Module):
  def __init__(self, kernel_size, **kwargs) -> None:
    super().__init__()
    self.max_p = nn.MaxPool2d(kernel_size= kernel_size, **kwargs)
  
  def forward(self, x):
     
    return self.max_p(torch.real(x)) + 1.0j * self.max_p(torch.imag(x))


def get_block2d(t, modes):
    h_modes = min(t.shape[-1]//2, modes//2)
    s = modes%2
    if len(t.shape) == 3:
        new_mean = torch.zeros(t.shape[0], h_modes*2 + s, 2*h_modes + s, dtype=t.dtype, device=t.device)
    else:
        new_mean = torch.zeros(t.shape[0],t.shape[1], h_modes*2 +s, 2*h_modes +s , dtype=t.dtype, device=t.device)
    new_mean[..., :h_modes+s, :h_modes+s ] = t[...,:h_modes+s, :h_modes+s ]
    new_mean[ ..., -h_modes:, :h_modes+s] = t[...,-h_modes:, :h_modes+s]
    new_mean[ ..., :h_modes+s, -h_modes: ] = t[...,:h_modes+s, -h_modes: ]
    new_mean[ ..., -h_modes:, -h_modes:] = t[...,-h_modes:, -h_modes:]

    return new_mean

def get_block1d(t, modes):
    h_modes = min(t.shape[-1]//2, modes//2)
    s = modes%2
    if len(t.shape) == 2:
        new_mean = torch.zeros(t.shape[0], h_modes*2+s, dtype=t.dtype, device=t.device)
    else:
        new_mean = torch.zeros(t.shape[0],t.shape[1], h_modes*2+s, dtype=t.dtype, device=t.device)
    new_mean[..., :h_modes+s ] = t[...,:h_modes+s ]
    new_mean[..., -h_modes:] = t[...,-h_modes:]

    return new_mean

def enblock2d(t, modes):
    assert t.shape[-1]<=modes

    s = t.shape[-1]%2

    h_modes = min(t.shape[-1]//2, modes//2)
    s =t.shape[-1]%2

    if len(t.shape) == 3:
        new_mean = torch.zeros(t.shape[0], modes, modes, dtype=t.dtype, device=t.device)
    else:
        new_mean = torch.zeros(t.shape[0],t.shape[1],modes, modes, dtype=t.dtype, device=t.device)

    new_mean[..., :h_modes+s, :h_modes+s ] = t[...,:h_modes+s, :h_modes+s ]
    new_mean[ ..., -h_modes:, :h_modes+s] = t[...,-h_modes:, :h_modes+s]
    new_mean[ ..., :h_modes+s, -h_modes: ] = t[...,:h_modes+s, -h_modes: ]
    new_mean[ ..., -h_modes:, -h_modes:] = t[...,-h_modes:, -h_modes:]

    return new_mean

def enblock1d(t, modes):
    assert t.shape[-1]<=modes
    s = t.shape[-1]%2
    h_modes = min(t.shape[-1]//2, modes//2)
    if len(t.shape) ==2:
        new_mean = torch.zeros(t.shape[0], modes, dtype=t.dtype, device=t.device)
    else:
        new_mean = torch.zeros(t.shape[0],t.shape[1], modes, dtype=t.dtype, device=t.device)

    new_mean[..., :h_modes+s ] = t[...,:h_modes+s ]
    new_mean[..., -h_modes:] = t[...,-h_modes:]

    return new_mean

class ComplexBatchNorm2dSim(Module):
    def __init__(self, num_features, modes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ComplexBatchNorm2dSim, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.modes = modes
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,modes, modes, 2))
            self.bias = Parameter(torch.Tensor(num_features,modes, modes, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, modes, modes, dtype = torch.complex64))
            self.register_buffer('running_covar_real', torch.ones(num_features, modes, modes))
            self.register_buffer('running_covar_imag', torch.ones(num_features, modes, modes))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar_real.zero_()
            self.running_covar_imag.zero_()
            self.running_covar_real[:,:,:] = 1
            self.running_covar_imag[:,:,:] = 1
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight,1)
            init.zeros_(self.bias)

    def forward(self, input, modes = None):
        if modes is not None:
            redo_modes = input.shape[-1]
            input = get_block2d(input, modes)
        exponential_average_factor = 0.0

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            mean = input.mean(0)
        else:
            mean = get_block2d(self.running_mean, input.shape[-1])

        if self.training and self.track_running_stats:
            # update running mean
            mean_t = exponential_average_factor * mean + (1 - exponential_average_factor) * get_block2d(self.running_mean,input.shape[-1])
            with torch.no_grad():
                
                h_modes = input.shape[-1]//2
                s = input.shape[-1]%2
                self.running_mean [:, :h_modes+s, :h_modes+s ] = mean_t[:,:h_modes+s, :h_modes+s ]
                self.running_mean [ :, -h_modes:, :h_modes+s] = mean_t[:,-h_modes:, :h_modes+s]
                self.running_mean [ :, :h_modes+s, -h_modes: ] = mean_t[:,:h_modes+s, -h_modes: ]
                self.running_mean [ :, -h_modes:, -h_modes:] = mean_t[:,-h_modes:, -h_modes:]

            
            

        input = input - mean[None, :, :, :]


        if self.training or (not self.training and not self.track_running_stats):

            #n = input.numel() / input.size(1)
            Crr = input.real.pow(2).mean(0)+self.eps
            Cii = input.imag.pow(2).mean(0)+self.eps
        else:
            Crr = get_block2d( self.running_covar_real, input.shape[-1]) +self.eps
            Cii = get_block2d( self.running_covar_imag, input.shape[-1])+self.eps
            
        if self.training and self.track_running_stats:
            running_covar_real_temp = exponential_average_factor * Crr + (1 - exponential_average_factor) * get_block2d(self.running_covar_real, input.shape[-1])

            running_covar_imag_temp = exponential_average_factor * Cii  + (1 - exponential_average_factor) * get_block2d(self.running_covar_imag, input.shape[-1])
            with torch.no_grad():
                

                h_modes = input.shape[-1]//2
                s = input.shape[-1]%2
                self.running_covar_real [:, :h_modes+s, :h_modes+s ] = running_covar_real_temp[:,:h_modes+s, :h_modes+s ]
                self.running_covar_real [ :, -h_modes:, :h_modes+s] = running_covar_real_temp[:,-h_modes:, :h_modes+s]
                self.running_covar_real [ :, :h_modes+s, -h_modes: ] = running_covar_real_temp[:,:h_modes+s, -h_modes: ]
                self.running_covar_real [ :, -h_modes:, -h_modes:] = running_covar_real_temp[:,-h_modes:, -h_modes:]

                self.running_covar_imag [:, :h_modes+s, :h_modes+s ] = running_covar_imag_temp[:,:h_modes+s, :h_modes+s ]
                self.running_covar_imag [ :, -h_modes:, :h_modes+s] = running_covar_imag_temp[:,-h_modes:, :h_modes+s]
                self.running_covar_imag [ :, :h_modes+s, -h_modes: ] = running_covar_imag_temp[:,:h_modes+s, -h_modes: ]
                self.running_covar_imag [ :, -h_modes:, -h_modes:] = running_covar_imag_temp[:,-h_modes:, -h_modes:]
            Crr = running_covar_real_temp
            Cii = running_covar_imag_temp
                

        if self.training and self.track_running_stats:
            input = input + mean[None,:,:,:] - mean_t[None,:,:,:]

        input = (input.real/Crr[None,:,:,:]) + 1.0j*(input.imag/Cii[None,:,:,:])

        if self.affine:
            Wr = get_block2d(self.weight[:,:,:,0], input.shape[-1])
            Wi = get_block2d(self.weight[:,:,:,1], input.shape[-1])
            Br = get_block2d(self.bias[:,:,:,0], input.shape[-1])
            Bi = get_block2d(self.bias[:,:,:,1], input.shape[-1])
            input = (Wr[None,:,:,:]*input.real + Br[None,:,:,:]) \
                    +1.0j*(Wi[None,:,:,:]*input.imag+ Bi[None,:,:,:])
        del Crr, Cii

        if modes is not None:
            input = enblock2d(input, redo_modes)
        return input

class ComplexBatchNorm1dSim(Module):
    def __init__(self, num_features, modes, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True):
        super(ComplexBatchNorm1dSim, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.modes = modes
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,modes, 2))
            self.bias = Parameter(torch.Tensor(num_features,modes, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features, modes, dtype = torch.complex64))
            self.register_buffer('running_covar_real', torch.ones(num_features, modes))
            self.register_buffer('running_covar_imag', torch.ones(num_features, modes))
            self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        else:
            self.register_parameter('running_mean', None)
            self.register_parameter('running_covar', None)
            self.register_parameter('num_batches_tracked', None)
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_covar_real.zero_()
            self.running_covar_imag.zero_()
            self.running_covar_real[:,:] = 1
            self.running_covar_imag[:,:] = 1
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            init.constant_(self.weight,1)
            init.zeros_(self.bias)

    def forward(self, input, modes = None):
        exponential_average_factor = 0.0
        if modes is not None:
            redo_modes = input.shape[-1]
            input = get_block1d(input, modes)

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        if self.training or (not self.training and not self.track_running_stats):
            mean = input.mean(0)
        else:
            mean = get_block1d(self.running_mean, input.shape[-1])

        if self.training and self.track_running_stats:
            # update running mean
            mean_t = exponential_average_factor * mean + (1 - exponential_average_factor) * get_block1d(self.running_mean,input.shape[-1])
            with torch.no_grad():
                
                h_modes = input.shape[-1]//2
                s = input.shape[-1]%2
                self.running_mean [:, :h_modes+s] = mean_t[:,:h_modes+s]
                self.running_mean [ :, -h_modes:] = mean_t[:,-h_modes:]

            #mean = mean_t
        input = input - mean[None, :, :]

        if self.training or (not self.training and not self.track_running_stats):

            #n = input.numel() / input.size(1)
            Crr = input.real.pow(2).mean(0)+self.eps
            Cii = input.imag.pow(2).mean(0)+self.eps
        else:
            Crr = get_block1d( self.running_covar_real, input.shape[-1]) +self.eps
            Cii = get_block1d( self.running_covar_imag, input.shape[-1])+self.eps
            
        if self.training and self.track_running_stats:
            running_covar_real_temp = exponential_average_factor * Crr + (1 - exponential_average_factor) * get_block1d(self.running_covar_real, input.shape[-1])

            running_covar_imag_temp = exponential_average_factor * Cii  + (1 - exponential_average_factor) * get_block1d(self.running_covar_imag, input.shape[-1])
            with torch.no_grad():
                

                h_modes = input.shape[-1]//2
                s = input.shape[-1]%2
                self.running_covar_real [:, :h_modes+s,  ] = running_covar_real_temp[:,:h_modes+s ]
                self.running_covar_real [ :, -h_modes:] = running_covar_real_temp[:,-h_modes:]

                self.running_covar_imag [:, :h_modes+s] = running_covar_imag_temp[:,:h_modes+s]
                self.running_covar_imag [ :, -h_modes:] = running_covar_imag_temp[:,-h_modes:]
            Crr = running_covar_real_temp
            Cii = running_covar_imag_temp
                


        if self.training and self.track_running_stats:
            input = input + mean[None,:,:] - mean_t[None,:,:]

        input = (input.real/Crr[None,:,:]) + 1.0j*(input.imag/Cii[None,:,:])

        if self.affine:
            Wr = get_block1d(self.weight[:,:,0], input.shape[-1])
            Wi = get_block1d(self.weight[:,:,1], input.shape[-1])
            Br = get_block1d(self.bias[:,:,0], input.shape[-1])
            Bi = get_block1d(self.bias[:,:,1], input.shape[-1])
            input = (Wr[None,:,:,]*input.real + Br[None,:,:]) \
                    +1.0j*(Wi[None,:,:]*input.imag+ Bi[None,:,:])
        del Crr, Cii
        if modes is not None:
            input = enblock1d(input, redo_modes)
        return input
    

class ComplexLayerNorm2d(Module):
    def __init__(self, num_features, modes, eps=1e-5, affine=True):
        super(ComplexLayerNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.modes = modes
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,modes, modes, 2))
            self.bias = Parameter(torch.Tensor(num_features,modes, modes, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
 
        self.reset_parameters()


    def reset_parameters(self):
        if self.affine:
            init.constant_(self.weight,1)
            init.zeros_(self.bias)

    def forward(self, input, modes = None):
        if modes is not None:
            redo_modes = input.shape[-1]
            input = get_block2d(input, modes)




        mean = input.mean(1)
        #print(mean.shape)

        input = input - mean[:, None, :, :]

        Crr = input.real.pow(2).mean(1)+self.eps
        Cii = input.imag.pow(2).mean(1)+self.eps

        input = (input.real/Crr[:,None,:,:]) + 1.0j*(input.imag/Cii[:,None,:,:])

        if self.affine:
            Wr = get_block2d(self.weight[:,:,:,0], input.shape[-1])
            Wi = get_block2d(self.weight[:,:,:,1], input.shape[-1])
            Br = get_block2d(self.bias[:,:,:,0], input.shape[-1])
            Bi = get_block2d(self.bias[:,:,:,1], input.shape[-1])
            
            input = (Wr[None,:,:,:]*input.real) \
                    +1.0j*(Wi[None,:,:,:]*input.imag)
        del Crr, Cii

        if modes is not None:
            input = enblock2d(input, redo_modes)
        return input
    
class ComplexLayerNorm1d(Module):
    def __init__(self, num_features, modes, eps=1e-5, affine=True, complex = True):
        super(ComplexLayerNorm1d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.modes = modes
        if self.affine:
            self.weight = Parameter(torch.Tensor(num_features,modes, 2))
            self.bias = Parameter(torch.Tensor(num_features,modes, 2))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        self.complex = complex
        self.reset_parameters()
        #self.register_buffer("spectral_decay", get_pweights(modes,modes))

    def get_reg_loss(self,):
        pass

    def reset_parameters(self):
        if self.affine:
            init.constant_(self.weight,1)
            init.zeros_(self.bias)

    def forward(self, input, modes = None):
        if modes is not None:
            redo_modes = input.shape[-1]
            input = get_block1d(input, modes)




        mean = input.mean(1)
        #print(mean.shape)

        input = input - mean[:, None, :]

        Crr = input.real.pow(2).mean(1)+self.eps
        if self.complex:
            Cii = input.imag.pow(2).mean(1)+self.eps

        #print(Crr.shape)

        if self.complex:
            input = (input.real/Crr[:,None,:]) + 1.0j*(input.imag/Cii[:,None,:])
        else:
            input = (input.real/Crr[:,None,:])

        if self.affine:
            if self.complex:
                Wr = get_block1d(self.weight[:,:,0], input.shape[-1])
                Wi = get_block1d(self.weight[:,:,1], input.shape[-1])
                Br = get_block1d(self.bias[:,:,0], input.shape[-1])
                Bi = get_block1d(self.bias[:,:,1], input.shape[-1])
                input = (Wr[None,:,:,]*input.real ) \
                        +1.0j*(Wi[None,:,:]*input.imag)
            else:
                Wr = get_block1d(self.weight[:,:,0], input.shape[-1])
                Br = get_block1d(self.bias[:,:,0], input.shape[-1])
                input = (Wr[None,:,:,]*input.real ) 


        if modes is not None:
            input = enblock1d(input, redo_modes)
        return input
    
def complex_relu(input):
    return relu(input.real)+1.0j*relu(input.imag)
def complex_selu(input):
    return selu(input.real)+1.0j*selu(input.imag)
def complex_gelu(input):
    return gelu(input.real)+1.0j*gelu(input.imag)

def complex_sigmoid(input):
    return sigmoid(input.real).type(torch.complex64)+1j*sigmoid(input.imag).type(torch.complex64)
def complex_tanh(input):
    return tanh(input.real).type(torch.complex64)+1j*tanh(input.imag).type(torch.complex64)

def complex_tanh_se(input):
    inp = input.clone()
    inp_abs = input.clone()
    inp = inp/(inp.abs()+1e-5)
    abs = inp_abs.abs()
    absn = tanh(abs)
    return (absn * inp).clone()

class modRelu(Module):
    def __init__(self) -> None:
        super().__init__()
        self.b = nn.Parameter(0*torch.randn(1))
    def forward(self, input):
        inp = input.clone()
        inp_abs = input.clone()
        inp = inp/(inp.abs()+1e-5)
        abs = inp_abs.abs()
        absn = relu(abs + self.b)
        return (absn * inp).clone()
activation_map = {'gelu_c': complex_gelu, 'relu_c': complex_relu, 'selu_c':complex_selu, 'tanh_c':complex_tanh, 'tanh_se': complex_tanh_se, 'mod_relu': modRelu,\
                  'gelu': gelu, 'relu': relu, 'selu':selu, 'tanh':tanh,}

def get_activation(name):
    global activation_map
    return activation_map[name]
