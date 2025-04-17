import json
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import math

def loadcfg(fn):
    try:
        with open(fn, 'r') as f:
            return json.load(f)
    except TypeError as e:
        logging.error(f'Filename to config file required. {e}')
        raise

def resample_1d(x: torch.tensor, num: int, complex = True, skip_nyq = False):
        '''
        parameters:
        ----------
        x: torch.tensor, data to resample
        num: int, Final size of the resampled data
        complex: bool, treat the data as complex function
        skip_nyq: bool, skip the Nyquist frequency
        '''
        if not complex:
                X = torch.fft.rfft(x, norm ='forward')
                Y = torch.zeros(X.shape[0], X.shape[1], num//2 + 1, dtype=X.dtype)
                #print("printing data type ", Y.dtype)
                en = min(num//2, x.shape[-1]//2)

                Y[:,:,:en+1] = X[:,:,:(en+1)]


                if skip_nyq:
                        Y[:,:,-1] = 0.0

                y = torch.fft.irfft(Y, n = num,norm ='forward').to(x.device)
                return y
        
        X = torch.fft.fft(x, norm ='forward')
        Y = torch.zeros(X.shape[0], X.shape[1],num, dtype=X.dtype)
        en = min(num//2, x.shape[-1]//2)

        Y[:,:,:en] = X[:,:,:en]
        Y[:,:,-en:] = X[:,:,-en:]

        if skip_nyq:
                Y[:,:,-en] = 0.0
        if complex:
                return torch.fft.ifft(Y, norm ='forward').to(x.device)

def resample(x, num, complex = True, skip_nyq = False):
        '''
        parameters:
        ----------
        x: torch.tensor, data to resample shpae (batch, channel, H, W)
        num: int, Final size of the resampled data (num[0], num[1])
        complex: bool, treat the data as complex function
        skip_nyq: bool, skip the Nyquist frequency
        '''
        if not complex:
                assert num[0] == num[1]
                X = torch.fft.rfft2(x, norm ='forward')
                Y = torch.zeros(X.shape[0], X.shape[1],num[0], num[1]//2 + 1, dtype=X.dtype)
                en = min(num[1]//2, x.shape[-1]//2)
                if num[0]%2==0:
                        Y[:,:,:en,:en+1] = X[:,:,:en,:(en+1)]
                        Y[:,:,-en:,:(en+1)] = X[:,:,-en:,:(en+1)]
                else:
                        Y[:,:,:en+1,:en+1] = X[:,:,:en+1,:(en+1)]
                        Y[:,:,-en:,:(en+1)] = X[:,:,-en:,:(en+1)]

                if skip_nyq:
                        Y[:,:,-en,:] = 0.0
                        Y[:,:,:,-1] = 0.0
                        if num[0]%2 ==1:
                                Y[:,:,en,:] = 0.0
                y = torch.fft.irfft2(Y, s= (num[0], num[1]),norm ='forward').to(x.device)
                assert y.shape[-1] == y.shape[-2]
                return y
        X = torch.fft.fft2(x, norm ='forward')
        Y = torch.zeros(X.shape[0], X.shape[1],num[0], num[1], dtype=X.dtype)
        en = min(num[0]//2, x.shape[-1]//2)
        Y[:,:,:en,:en] = X[:,:,:en,:en]
        Y[:,:,-en:,:en] = X[:,:,-en:,:en]
        Y[:,:,:en,-en:] = X[:,:,:en,-en:]
        Y[:,:,-en:,-en:] = X[:,:,-en:,-en:]
        if skip_nyq:
                Y[:,:,-en,:] = 0.0
                Y[:,:,:,-en] = 0.0
        if complex:
                return torch.fft.ifft2(Y, norm ='forward').to(x.device)

def down_sample_keepsize_1d(X, num, keep_size, skip_nyq = False):
        '''
        parameters:
        ----------
        X: torch.tensor complex, Fourier Coefficients of the Function
        num: int, Number of Fourier modes to keep
        keep_size: int, Output size of the data
        skip_nyq: bool, skip the Nyquist frequency
        '''
        assert num <= keep_size
        Y = torch.zeros(X.shape[0], X.shape[1],keep_size//2 + 1, dtype=X.dtype)
        en = min(num[1]//2, X.shape[-2]//2)
        Y[:,:,:en+1] = X[:,:,:(en+1)]
        if skip_nyq:
                Y[:,:,-1] = 0.0
        y = torch.fft.irfft(Y, n= keep_size,norm ='forward').to(X.device)
        return y

def down_sample_keepsize(X, num, keep_size, skip_nyq = False, isfreq = True):
        '''
        Note the input is Fourier Coefficients
        '''
        assert num[0] == num[1]
        assert keep_size >= num[0]
        if not isfreq:
             X = torch.fft.fft2(X, norm = 'forward')

        Y = torch.zeros(X.shape[0], X.shape[1],keep_size, keep_size//2 + 1, dtype=X.dtype)
        en = min(num[1]//2, X.shape[-2]//2)
        if num[0]%2==0:
                Y[:,:,:en,:en+1] = X[:,:,:en,:(en+1)]
                Y[:,:,-en:,:(en+1)] = X[:,:,-en:,:(en+1)]
        else:
                Y[:,:,:en+1,:en+1] = X[:,:,:en+1,:(en+1)]
                Y[:,:,-en:,:(en+1)] = X[:,:,-en:,:(en+1)]
        if skip_nyq:
                Y[:,:,-en,:] = 0.0
                Y[:,:,:,-1] = 0.0
                if num[0]%2 ==1:
                        Y[:,:,en,:] = 0.0
        y = torch.fft.irfft2(Y, s= (keep_size, keep_size),norm ='forward').to(X.device)
        assert y.shape[-1] == y.shape[-2]
        return y


def make_real(FS):
        '''
        Enforce conjugate symmetry in Fourier Coefficients FS
        '''
        FS[:,:,0,0] = torch.real(FS[:,:,0,0])
        FS[:,:,FS.shape[-2]//2,0] = torch.real(FS[:,:,FS.shape[-2]//2,0])
        FS[:,:,0,FS.shape[-1]//2] = torch.real(FS[:,:,0,FS.shape[-1]//2])
        FS[:,:,FS.shape[-2]//2,FS.shape[-1]//2] = torch.real(FS[:,:,FS.shape[-2]//2,FS.shape[-1]//2])
        FS[:,:,FS.shape[-2]//2+1:,0] = torch.conj(torch.flip(FS[:,:,1:FS.shape[-2]//2,0], dims = [-1]))
        FS[:,:,FS.shape[-2]//2+1:,FS.shape[-1]//2] = torch.conj(torch.flip(FS[:,:,1:FS.shape[-2]//2,FS.shape[-1]//2], dims = [-1]))
        FS[:,:, 1:, FS.shape[-1]//2 + 1: ] = torch.flip(torch.conj(FS[:,:, 1:, 1:FS.shape[-1]//2 ]), dims = [-2, -1])
        FS[:,:,0,FS.shape[-1]//2+1:] = torch.conj(torch.flip(FS[:,:,0,1:FS.shape[-1]//2], dims = [-1]))

        return FS

def sliding_windows(a, W):
        a = np.asarray(a)
        p = np.zeros(W-1,dtype=a.dtype)
        b = np.concatenate((p,a,p))
        s = b.strides[0]
        strided = np.lib.stride_tricks.as_strided
        M = strided(b[W-1:], shape=(W,len(a)+W-1), strides=(-s,s))[:, :W]
        return torch.tensor(M.T.tolist(), dtype=torch.float)

def get_diag_mask(mode_size, band_size =3, dtype = torch.cfloat):
    '''
    Produce a mask to limit the mixing of Fourier Coefficients
    '''
    mask = sliding_windows([1]*band_size, mode_size//2+1)
    if mode_size%2==0:
        mask = torch.nn.functional.pad(
            mask[None, :, :], [0, 0, 0, mode_size//2 - 1], mode='reflect')[0]
    else:
         mask = torch.cat((mask, torch.flip(mask, dims=[-2])[:-1,:]),-2)
    if mode_size%2==0:
        mask = torch.cat((mask, torch.flip(mask[:, :], dims=[-1])[:,1:-1]), -1)
    else:
        mask = torch.cat((mask, torch.flip(mask[:, :], dims=[-1])[:,:-1]), -1)
    mask[0, 0] = 1.0
    return mask.to(dtype = dtype)

def get_mixer_mask(mode_size, dtype, band_width = -1):
    mask = torch.ones((mode_size//2+1, mode_size//2+1),
                    dtype=dtype)
    mask[:, :] = torch.tril(mask[:, :], diagonal=0)
    if mode_size%2==0:
        mask = torch.nn.functional.pad(
            mask[None, :, :], [0, 0, 0, mode_size//2 - 1], mode='reflect')[0]
    else:
         mask = torch.cat((mask, torch.flip(mask, dims=[-2])[:-1,:]),-2)
    if mode_size%2==0:
        mask = torch.cat((mask, torch.flip(mask[:, :], dims=[-1])[:,1:-1]), -1)
    else:
        mask = torch.cat((mask, torch.flip(mask[:, :], dims=[-1])[:,:-1]), -1)
    mask[0, 0] = 1.0
    if band_width != -1:
         M = get_diag_mask(mode_size, band_size=band_width,dtype=dtype)
         mask = mask * M
    return mask

def get_pweights(dim1,dim2):
        gridx = torch.tensor(np.linspace(0, 1, dim1), dtype=torch.float)
        gridx = gridx.reshape(dim1, 1).repeat([ 1, dim2])
        gridy = torch.tensor(np.linspace(0, 1, dim2), dtype=torch.float)
        gridy = gridy.reshape(1, dim2).repeat([dim1, 1])
        return 30.0*torch.exp(-10*(1*(gridx-0.5)**2+1.0*(gridy-0.5)**2))

def get_gaussian(dim1,dim2, width, real):
        gridx = torch.abs(torch.tensor(np.linspace(0, 1, dim1), dtype=torch.float))
        gridx = gridx.reshape(dim1, 1).repeat([ 1, dim2])
        gridy = torch.abs(torch.tensor(np.linspace(0, 1, dim2), dtype=torch.float))
        gridy = gridy.reshape(1, dim2).repeat([dim1, 1])
        gf = torch.exp(-2*np.pi*((gridx-0.5)**2+(gridy-0.5)**2)/(width/dim1)**2)
        gf = torch.fft.ifftshift(gf)
        if real:
             gf = gf[:,:dim2//2+1]
        return gf

def get_pweights_1D(dim1):
        gridx = torch.tensor(np.linspace(0, 1, dim1), dtype=torch.float)
        return 10.0*torch.exp(-10*((gridx-0.5)**2))

def get_row_element(x,N):
    if x == 1 or x == 0:
        return np.exp(-1j*x*(N-1)/2)
    else:
        return 1/N * np.exp(-1j*x*(N-1)/2)* np.sin(N*x/2)/np.sin(x/2)
def get_mat(L,N):
    return torch.tensor([[get_row_element(2*np.pi*(l/L - n/N),N) for n in range(N)] for l in range(L)])


class ComplexGELU(nn.Module):
    def __init__(self, approximate='none'):
        super(ComplexGELU, self).__init__()
        self.approximate = approximate

    def forward(self, input):
        return F.gelu(torch.real(input))\
                + 1.0j*F.gelu(torch.imag(input))

class ComplexSELU(nn.Module):
    def __init__(self):
        super(ComplexSELU, self).__init__()

    def forward(self, input):
        return F.selu(torch.real(input))\
                + 1.0j*F.selu(torch.imag(input))

class ComplexRELU(nn.Module):
    def __init__(self):
        super(ComplexRELU, self).__init__()
    def forward(self, input):
        return F.relu(torch.real(input))\
                + 1.0j*F.relu(torch.imag(input))
    
class ComplexLeakyRELU(nn.Module):
    def __init__(self, slope = 0.1):
        super(ComplexLeakyRELU, self).__init__()
        self.nl = torch.nn.LeakyReLU(negative_slope=slope)
    def forward(self, input):
        return self.nl(torch.real(input))\
                + 1.0j*self.nl(torch.imag(input))
    
class ComplexTanh(nn.Module):
    def __init__(self):
        super(ComplexTanh, self).__init__()
    def forward(self, input):
        return F.tanh(torch.real(input))\
                + 1.0j*F.tanh(torch.imag(input))



class AddUniformNoise(object):
        '''
        Uniform noise augmentation.
        '''
        def __init__(self, mean=0., std=1., prob = 0.5):
                self.std = std
                self.mean = mean
                self.prob = prob
                
        def __call__(self, tensor):
                std = 0
                if random.random()< self.prob:

                     std = self.std[0] + random.random()*(self.std[1]- self.std[0])
                     
                return tensor + (torch.rand(tensor.size()) -0.5) * std + self.mean
        
        def __repr__(self):
                return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class AddCircularShift(object):
        '''
        circular shift Augmentation.
        '''
        def __init__(self, mean=0., x_shift=0.25,y_shift = 0.25, prob = 0.7):
                self.x_shift = x_shift
                self.y_shift = y_shift
                self.prob = prob
                
        def __call__(self, tensor):
                if random.random()< self.prob:
                    sx = int(self.x_shift * random.random() * tensor.shape[-2])
                    sy = int(self.y_shift * random.random() * tensor.shape[-1])

                    if random.random()>0.5:
                     sx = -1*sx
                    if random.random()>0.5:
                     sy = -1*sy

                    if sx**2 <1 or sy**2 <1:
                        return tensor
                    else:
                        return torch.roll(tensor, shifts=(sx,sy), dims=(-2,-1))
                     
                return tensor
        
        def __repr__(self):
                return self.__class__.__name__ + '(x shift={0}, y shift={1})'.format(self.x_shift, self.y_shift)


def get_stack(x_fft, pool_window, keep_size, base_res = 4, increment = 1):
    x = torch.real(torch.fft.ifft2(x_fft))
    x_fft = torch.fft.irfft2(x)
    stack = None
    for i in range(base_res, x_fft.shape[-2]+increment,increment):
        i = min(i, keep_size)
        x_res = down_sample_keepsize(x_fft,[i,i],keep_size).clone()
        pooled = F.max_pool2d(x_res, pool_window, ceil_mode = True)
        if stack is None:
            stack = pooled.reshape(x_fft.shape[0], 1, -1)
        else:
            stack = torch.cat((stack, pooled.reshape(x_fft.shape[0], 1, -1)), dim = 1)
    
    return stack

def get_stack_differential(x_fft, pool_window, keep_size, base_res = 4, increment=2):
    stack = None
    x_res = None
    x_prev = None
    for i in range(base_res, x_fft.shape[-2]+increment,increment):

        if x_res is None:
            x_res = down_sample_keepsize(x_fft,[i,i],keep_size)
            x_prev = x_res.clone().detach()
        else:
            x_ = down_sample_keepsize(x_fft,[i,i],keep_size)
            x_res = x_ - x_prev
            x_prev = x_.clone().detach()

        pooled = F.max_pool2d(x_res, pool_window, ceil_mode= True)
        if stack is None:
            stack = pooled.reshape(x_fft.shape[0], 1, -1)
        else:
            stack = torch.cat((stack, pooled.reshape(x_fft.shape[0], 1, -1)), dim = 1)
    
    return stack

def get_stack_1d(x_fft, pool_window, keep_size, base_res = 4):
    stack = None
    for i in range(base_res, x_fft.shape[-1]+1,1):

        x_res = down_sample_keepsize_1d(x_fft,i,keep_size).clone()
        pooled = F.max_pool1d(x_res, pool_window)
        if stack is None:
            stack = pooled.reshape(x_fft.shape[0], 1, -1)
        else:
            stack = torch.cat((stack, pooled.reshape(x_fft.shape[0], 1, -1)), dim = 1)
    
    return stack

def get_pooled(x_fft, res, pool_size):
    x = torch.real(torch.fft.ifft2(x_fft))
    x = resample(x, [res, res], complex= False)
    
    x = F.max_pool2d(x, pool_size)
    return x.reshape(x.shape[0], -1)

def get_pooled_1d(x_fft, res, pool_size):
    x = torch.real(torch.fft.ifft(x_fft))
    x = resample_1d(x, res, complex= False)
    
    x = F.max_pool1d(x, pool_size)
    return x.reshape(x.shape[0], -1)



class Cutout(object):
        """Randomly mask out one or more patches from an image.

        Args:
                n_holes (int): Number of patches to cut out of each image.
                length (int): The length (in pixels) of each square patch.
        """

        def __init__(self, n_holes, length):
                self.n_holes = n_holes
                self.length = length

        def __call__(self, img):
                """
                Args:
                        img (Tensor): Tensor image of size (C, H, W).
                Returns:
                        Tensor: Image with n_holes of dimension length x length cut out of it.
                """
                h = img.size(1)
                w = img.size(2)

                mask = np.ones((h, w), np.float32)

                for n in range(self.n_holes):
                        y = np.random.randint(h)
                        x = np.random.randint(w)

                        y1 = np.clip(y - self.length // 2, 0, h)
                        y2 = np.clip(y + self.length // 2, 0, h)
                        x1 = np.clip(x - self.length // 2, 0, w)
                        x2 = np.clip(x + self.length // 2, 0, w)

                        mask[y1: y2, x1: x2] = 0.

                mask = torch.from_numpy(mask)
                mask = mask.expand_as(img)
                img = img * mask

                return img

        def __repr__(self):
                s = 'Cutout(n_holes={}, length={})'.format(self.n_holes, self.length)
                return s

def scale_resize(x, res, resize_res, mode, resize_mode):
    if mode == "ideal":
        img = resample(x[None,...],(res,res), complex= False)[0]
    elif mode == "bicubic" or mode == "linear":
        img = F.interpolate(x[None,...], size=(res, res), mode = mode, align_corners=True,  antialias=True)[0]
    else:
         raise Exception()
    
    if resize_mode == "pad":
        l = resize_res - img.shape[-2]
        t = resize_res - img.shape[-1]
        img = F.pad(img, pad = [l//2, l - l//2, t//2, t - t//2])
    elif resize_mode == 'resize':
        img = resample(img,(resize_res,resize_res), complex= False)
    return img
        
class AddScaling(object):
        '''
        scaling Augmentation.
        '''
        def __init__(self, low = 0.3, high=1.0, prob = 0.7,mode = 'ideal', resize_mode = 'None'):
                self.low = low
                self.high = high
                self.prob = prob
                self.resize_mode = resize_mode
                self.mode = mode
                
        def __call__(self, tensor):
                if random.random()< self.prob:
                    low_res = int(self.low * tensor.shape[-1])
                    high_res = int(self.high * tensor.shape[-1])
                    t_res = random.randint(low_res, high_res+1)

                    tensor = scale_resize(tensor, t_res, high_res, self.mode, self.resize_mode)

                    return tensor
                return tensor
        
        def __repr__(self):
                return self.__class__.__name__ 