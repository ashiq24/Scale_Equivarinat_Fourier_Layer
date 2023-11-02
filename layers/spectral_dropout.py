import torch
import math
import torch.nn as nn
import random

class SpectralDropout(nn.Module):
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





    