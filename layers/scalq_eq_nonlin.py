import torch
import torch.nn as nn
from utils.core_utils import resample, resample_1d, down_sample_keepsize
import torch.nn.functional as F
import math

class scaleEqNonlin(nn.Module):
    def __init__(self, non_lin, base_res, normalization = None, channels = None, max_res = 28, increment = 1) -> None:
        '''
        parameter
        ----------
        non_lin : Nonlinearity funtion to be converted to scale equivarinat
        base_res : int, Lower resolution under considertion for scale equivariance
        normalization : None, 'instance' or 'batch'. 
        max_res : int, Highest resolution under considertion for Non-trivial scale equivarinace
        increment : int, Increment for the resolutions from base to max_res. Determines the set of scales/resoltuin to 
                    consider for scale equivariance. Default is 1. i.e. every scales/resolutions are considered.
        '''
        super().__init__()
        self.non_lin = non_lin
        self.base_res = base_res
        self.norm = normalization
        self.increment = increment
        print("Making Non_lin layer, Max res", max_res, " Increment", self.increment)
        if normalization is not None:
            if normalization == 'instance':
                self.norm = nn.ModuleDict()
                for res in range(self.base_res, max_res+self.increment, self.increment):
                    self.norm[str(res)] = nn.InstanceNorm2d(channels,affine= True)
            else:
                self.norm = nn.BatchNorm2d(channels)
                
        assert self.base_res%2 ==0 


    def forward(self, x_ft):
        '''
        parameters
        x_ft : complex torch.Tensor, Fourier Transform of the input
        '''
        x = torch.real(torch.fft.ifft2(x_ft, norm = 'forward'))
        x_out = torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3]//2 +1, dtype=x_ft.dtype).to(x_ft.device)

        assert x.shape[-1]>= self.base_res

        #copying the base res case
        xbase = resample(x, [self.base_res, self.base_res], complex= False)
        if self.norm is not None:
            xbase = self.norm[str(self.base_res)](xbase)
        xbase = self.non_lin(xbase)


        xbase_rft = torch.fft.rfft2(xbase, norm = 'forward')
        x_out[..., :self.base_res//2, :self.base_res//2 +1] = xbase_rft[..., :self.base_res//2, :self.base_res//2 +1]
        x_out[..., -(self.base_res//2):, :self.base_res//2 +1] = xbase_rft[..., -(self.base_res//2):, :self.base_res//2 +1]

        X = torch.fft.rfft2(x, norm = 'forward')
        if self.increment ==1:
            for res in range(self.base_res+1, x.shape[-1]+1):
                x_res = down_sample_keepsize(X, [res, res], res) # resample(x, [res,res], complex= False) # 
                if self.norm is not None:
                    x_res = self.norm[str(res)](x_res)
                x_res = self.non_lin(x_res)
                x_res_rft =  torch.fft.rfft2(x_res, norm = 'forward')
                s = res%2
                if s ==1:
                    x_out[..., res//2, :res//2+1] = x_res_rft[..., res//2, :res//2+1]
                    x_out[..., -(res//2), :res//2+1] = x_res_rft[..., -(res//2), :res//2+1]
                    x_out[..., :res//2+1, res//2] = x_res_rft[..., :res//2+1, res//2]
                    x_out[..., -(res//2):, res//2] = x_res_rft[..., -(res//2):, res//2]
                else:
                    x_out[..., -(res//2), :res//2+1] = x_res_rft[..., -(res//2), :res//2+1]
                    x_out[..., -(res//2):, res//2] = x_res_rft[..., -(res//2):, res//2]
                    x_out[..., :(res//2), res//2] = x_res_rft[..., :(res//2), res//2]
        else:
            prev = self.base_res//2 +1
            for res in range(self.base_res+self.increment, x.shape[-1]+self.increment, self.increment):
                x_res = down_sample_keepsize(X, [res-1, res-1], res) # resample(x, [res,res], complex= False) #                 
                if self.norm is not None:
                    x_res = self.norm[str(res)](x_res)
                x_res = self.non_lin(x_res)
                x_res_rft =  torch.fft.rfft2(x_res, norm = 'forward')
                t_res = min(res, x_out.shape[-2])
                
                x_out[..., prev-1 :t_res//2, :t_res//2+1] = x_res_rft[..., prev-1:t_res//2, :t_res//2+1]
                x_out[..., -(t_res//2):-(prev -1), :t_res//2+1] = x_res_rft[..., -(t_res//2):-(prev -1), :t_res//2+1]
                x_out[..., :t_res//2, prev-1: t_res//2+1] = x_res_rft[..., :t_res//2, prev-1: t_res//2+1]
                x_out[..., -(t_res//2):, prev-1: t_res//2+1] = x_res_rft[..., -(t_res//2):, prev-1: t_res//2+1]

                prev = t_res//2 +1

        x_out = torch.fft.fft2(torch.fft.irfft2(x_out, s= (x.shape[-2], x.shape[-1]), norm = 'forward'), norm = 'forward')
        return x_out
    
    def get_feature(self, x):
        return torch.fft.ifft2(self.forward(torch.fft.fft2(x))).real

class scaleEqNonlinMaxp(nn.Module):
    '''
    parameter
    ----------
    pool_window : int, Pooling window size
    '''
    def __init__(self, non_lin, base_res, normalization = None, channels = None, pool_window = 2, max_res = 28, increment = 1) -> None:
        super().__init__()
        self.non_lin = non_lin
        self.base_res = base_res
        self.norm = normalization
        self.pool_window = pool_window
        self.increment = increment
        if normalization is not None:
            if normalization == 'instance':
                self.norm = nn.ModuleDict()
                for res in range(self.base_res, max_res+self.increment, self.increment):
                    self.norm[str(res)] = nn.InstanceNorm2d(channels,affine= True)
            else:
                self.norm = nn.BatchNorm2d(channels)
                
        

        assert self.base_res%2 ==0 


    def forward(self, x_ft):
        '''
        parameters
        x_ft : complex torch.Tensor, Fourier Transform of the input
        '''
        x = torch.real(torch.fft.ifft2(x_ft, norm = 'forward'))

        out_res = math.ceil(x.shape[-2]/self.pool_window)
        x_out = torch.zeros(x.shape[0], x.shape[1], out_res, out_res//2 +1, dtype=x_ft.dtype).to(x_ft.device)

        assert x.shape[-1]>= self.base_res

        #copying the base res case
        xbase = resample(x, [self.base_res, self.base_res], complex= False)
        if self.norm is not None:
            xbase = self.norm[str(self.base_res)](xbase)
        xbase = self.non_lin(xbase)
        xbase = down_sample_keepsize(xbase, [math.ceil(self.base_res/self.pool_window),math.ceil(self.base_res/self.pool_window)],self.base_res, isfreq= False)
        xbase = F.max_pool2d(xbase, self.pool_window, ceil_mode = True)

        res = xbase.shape[-1]
        xbase_rft = torch.fft.rfft2(xbase, norm = 'forward')
        x_out[..., :res//2, :res//2 +1] = xbase_rft[..., :res//2, :res//2 +1]
        x_out[..., -(res//2):, :res//2 +1] = xbase_rft[..., -(res//2):, :res//2 +1]

        X = torch.fft.rfft2(x, norm = 'forward')
        if self.increment == 1:
            for res in range(self.base_res+self.pool_window, x.shape[-1]+self.pool_window, self.pool_window):
                x_res = down_sample_keepsize(X, [res, res], res) # resample(x, [res,res], complex= False) # 

                if self.norm is not None:
                    x_res = self.norm[str(res)](x_res)
                
                x_res = self.non_lin(x_res)
                x_res = down_sample_keepsize(x_res, [math.ceil(res/self.pool_window),math.ceil(res/self.pool_window)],res, isfreq= False)
                x_res = F.max_pool2d(x_res, self.pool_window, ceil_mode = True)
                
                c_res = x_res.shape[-1]
                x_res_rft =  torch.fft.rfft2(x_res, norm = 'forward')

                s = c_res%2
                if s ==1:
                    x_out[..., c_res//2, :c_res//2+1] = x_res_rft[..., c_res//2, :c_res//2+1]
                    x_out[..., -(c_res//2), :c_res//2+1] = x_res_rft[..., -(c_res//2), :c_res//2+1]
                    x_out[..., :c_res//2+1, c_res//2] = x_res_rft[..., :c_res//2+1, c_res//2]
                    x_out[..., -(c_res//2):, c_res//2] = x_res_rft[..., -(c_res//2):, c_res//2]


                else:
                    x_out[..., -(c_res//2), :c_res//2+1] = x_res_rft[..., -(c_res//2), :c_res//2+1]
                    x_out[..., -(c_res//2):, c_res//2] = x_res_rft[..., -(c_res//2):, c_res//2]
                    x_out[..., :(c_res//2), c_res//2] = x_res_rft[..., :(c_res//2), c_res//2]
        else:
            prev = res//2 + 1
            for res in range(self.base_res+self.increment, x.shape[-1]+self.increment, self.increment):
                x_res = down_sample_keepsize(X, [res-1, res-1], res) # resample(x, [res,res], complex= False) # 

                if self.norm is not None:
                    x_res = self.norm[str(res)](x_res)
                
                x_res = self.non_lin(x_res)
                x_res = down_sample_keepsize(x_res, [math.ceil(res/self.pool_window),math.ceil(res/self.pool_window)],res, isfreq= False)
                x_res = F.max_pool2d(x_res, self.pool_window, ceil_mode = True)
                
                c_res = x_res.shape[-1]
                x_res_rft =  torch.fft.rfft2(x_res, norm = 'forward')
                t_res = min(x_out.shape[-2],c_res)

                x_out[..., prev-1 :t_res//2, :t_res//2+1] = x_res_rft[..., prev-1:t_res//2, :t_res//2+1]
                x_out[..., -(t_res//2):-(prev -1), :t_res//2+1] = x_res_rft[..., -(t_res//2):-(prev -1), :t_res//2+1]
                x_out[..., :t_res//2, prev-1: t_res//2+1] = x_res_rft[..., :t_res//2, prev-1: t_res//2+1]
                x_out[..., -(t_res//2):, prev-1: t_res//2+1] = x_res_rft[..., -(t_res//2):, prev-1: t_res//2+1]
                
                prev = t_res//2 +1
            

        x_out = torch.fft.fft2(torch.fft.irfft2(x_out, s= (out_res, out_res), norm = 'forward'), norm = 'forward')
        return x_out
    def get_feature(self, x):
        return torch.fft.ifft2(self.forward(torch.fft.fft2(x))).real

class scaleEqNonlin1d(nn.Module):
    def __init__(self, non_lin, base_res, normalization = None, channels = None) -> None:
        super().__init__()
        self.non_lin = non_lin
        self.base_res = base_res
        self.norm = normalization
        if normalization is not None:
            if normalization == 'instance':
                self.norm = nn.InstanceNorm1d(channels,affine= True)
            else:
                self.norm = nn.BatchNorm1d(channels)
                
        

        assert self.base_res%2 ==0 


    def forward(self, x_ft):
        '''
        parameters
        x_ft : complex torch.Tensor, Fourier Transform of the input
        '''
        x = torch.real(torch.fft.ifft(x_ft, norm = 'forward'))
        x_out = torch.zeros(x.shape[0], x.shape[1], x.shape[2]//2 + 1, dtype=x_ft.dtype).to(x_ft.device)

        assert x.shape[-1]>= self.base_res

        #copying the base res case
        xbase = resample_1d(x, self.base_res, complex= False)
        if self.norm is not None:
            xbase = self.norm(xbase)
        xbase = self.non_lin(xbase)


        xbase_rft = torch.fft.rfft(xbase, norm = 'forward')
        x_out[..., :self.base_res//2,] = xbase_rft[..., :self.base_res//2]
        x_out[..., -(self.base_res//2):] = xbase_rft[..., -(self.base_res//2):]


        for res in range(self.base_res+1, x.shape[-1]+1):
            x_res = resample_1d(x, res, complex= False)

            if self.norm is not None:
                x_res = self.norm(x_res)
            
            x_res = self.non_lin(x_res)
            
            # if self.training:
            #     x_res = F.dropout2d(x_res, 0.2)

            x_res_rft =  torch.fft.rfft2(x_res, norm = 'forward')
            s = res%2
            if s ==1:
                x_out[..., :res//2+1] = x_res_rft[..., :res//2+1]
                x_out[..., -(res//2):] = x_res_rft[..., -(res//2):]
            else:
                x_out[..., -(res//2):] = x_res_rft[..., -(res//2):]
                x_out[..., :(res//2)] = x_res_rft[..., :(res//2)]
        x_out = torch.fft.fft(torch.fft.irfft(x_out, n=x.shape[-1], norm = 'forward'), norm = 'forward')
        return x_out