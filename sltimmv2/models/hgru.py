from functools import partial
from typing import Any, Callable, Sequence, Tuple

import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as jnp


import timm
from timm.models._registry import register_model


ModuleDef = Any
jnp.random.seed(42)

## COULDN"T USE IT (pytorch restrictions)

# def chrono_init(key, shape, dtype):
# 	# dtype = dtypes.canonicalize_dtype(dtype) ## not required
# 	# return jnp.log(jax.random.uniform(jax.random.PRNGKey(42), shape, dtype) * 7 + 1)
#     return jnp.log(jax.random.uniform(1, 7-1, shape)).astype(dtype) # Tmax = 7

# def chrono_init2(key, shape, dtype):
# 	# dtype = dtypes.canonicalize_dtype(dtype) ## not required
# 	# return -jnp.log(jax.random.uniform(jax.random.PRNGKey(42), shape, dtype) * 7 + 1)
# 	return -jnp.log(jax.random.uniform(1, 7-1, shape)).astype(dtype) # Tmax = 7

class hConvGruCell(nn.Module):
    
    def __init__(self, input_size, hidden_size, kernel_size, batchnorm=True, layernorm = False, timesteps=8):
        
        super().__init__()


        self.padding = kernel_size // 2
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.timesteps = timesteps
        self.batchnorm = batchnorm
        self.layernorm = layernorm

        ## UPDATE RESET GATE
        self.u1_gate = nn.Conv2d(hidden_size, hidden_size, 1)
        self.u2_gate = nn.Conv2d(hidden_size, hidden_size, 1)
		
        init.orthogonal_(self.u1_gate.weight)
        init.orthogonal_(self.u2_gate.weight)
		
        init.uniform_(self.u1_gate.bias.data, 1, 8.0 - 1)
        self.u1_gate.bias.data.log()
        self.u2_gate.bias.data =  -self.u1_gate.bias.data

        self.w_gate_inh = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
        self.w_gate_exc = nn.Parameter(torch.empty(hidden_size , hidden_size , kernel_size, kernel_size))
		
        init.orthogonal_(self.w_gate_inh)
        init.orthogonal_(self.w_gate_exc)
		
        self.w_gate_inh.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)
        self.w_gate_exc.register_hook(lambda grad: (grad + torch.transpose(grad,1,0))*0.5)


        if self.batchnorm:
            #self.bn = nn.ModuleList([nn.GroupNorm(25, 25, eps=1e-03) for i in range(32)])
            self.bn = nn.ModuleList([nn.BatchNorm2d(self.hidden_size, eps=1e-03) for i in range(32)])
        else:
            self.n = nn.Parameter(torch.randn(self.timesteps,1,1))
        for bn in self.bn:
            init.constant_(bn.weight, 0.1)
			
        
        if self.layernorm:
            self.ln = nn.ModuleList([nn.LayerNorm(25, eps=1e-03) for i in range(32)])
        else:
            self.n = nn.Parameter(torch.randn(self.timesteps,1,1))
        for ln in self.bn:
            init.constant_(ln.weight, 0.1)
			

        ## HYPERPARAMETERS
        self.alpha = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.gamma = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.kappa = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.w = nn.Parameter(torch.empty((hidden_size,1,1)))
        self.mu= nn.Parameter(torch.empty((hidden_size,1,1)))
        
        init.constant_(self.alpha, 0.1)
        init.constant_(self.gamma, 1.0)
        init.constant_(self.kappa, 0.5)
        init.constant_(self.w, 0.5)
        init.constant_(self.mu, 1)


    def forward(self, input_, prev_state2, timestep=0):
        
        if timestep == 0:
            prev_state2 = torch.empty_like(input_)
            init.xavier_normal_(prev_state2)
			
        i = timestep
        if self.batchnorm:
            
            g1_t = torch.sigmoid(self.bn[i*4+0](self.u1_gate(prev_state2)))
            c1_t = self.bn[i*4+1](F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding))
            next_state1 = F.relu(input_ - F.relu(c1_t*(self.alpha*prev_state2 + self.mu)))
            
            g2_t = torch.sigmoid(self.bn[i*4+2](self.u2_gate(next_state1)))
            c2_t = self.bn[i*4+3](F.conv2d(next_state1, self.w_gate_exc, padding=self.padding))
            h2_t = F.relu(self.kappa*next_state1 + self.gamma*c2_t + self.w*next_state1*c2_t)
            prev_state2 = (1 - g2_t)*prev_state2 + g2_t*h2_t

        else:
            g1_t = F.sigmoid(self.u1_gate(prev_state2))
            c1_t = F.conv2d(prev_state2 * g1_t, self.w_gate_inh, padding=self.padding)
            next_state1 = F.tanh(input_ - c1_t*(self.alpha*prev_state2 + self.mu))
            # g2_t = F.sigmoid(self.bn[i*4+2](self.u2_gate(next_state1)))
            g2_t = F.sigmoid(self.u2_gate(next_state1))
            c2_t = F.conv2d(next_state1, self.w_gate_exc, padding=self.padding)
            h2_t = F.tanh(self.kappa*(next_state1 + self.gamma*c2_t) + (self.w*(next_state1*(self.gamma*c2_t))))
            prev_state2 = self.n[i]*((1 - g2_t)*prev_state2 + g2_t*h2_t)

        return prev_state2



class hConvGru(nn.Module):

    def __init__(self, timesteps=8, filt_size = 9):
            super().__init__()
            
            self.num_classes = 2
            self.hidden_size = 16 ## previous gpu implementation had hidden_size = 25
            self.kernel_size = filt_size
            self.timesteps = timesteps
            self.dtype = torch.float32
            

            self.conv0 = nn.Conv2d(3, self.hidden_size, kernel_size= 7, padding=3) ## previous gpu implementation had serre gabor initialization
            
            self.bnip = nn.BatchNorm2d(self.hidden_size)
            self.rnncell1 = hConvGruCell(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size)
            self.bnop = nn.BatchNorm2d(self.hidden_size)
            
            self.conv6 = nn.Conv2d(self.hidden_size, 2, kernel_size=1) ## previous gpu implementation had xavier intialization
            self.dense1 = nn.Linear(2, self.num_classes)

    def forward(self, x):
        
            x = self.conv0(x)
            x = self.bnip(x) ## jax had running average
            print(f'First conv layer passed output shape {x.shape}')
            internal_state = torch.zeros_like(x)
            for i in range(self.timesteps):
                internal_state = self.rnncell1(x, internal_state, timestep = i)
            
            x = self.bnop(internal_state)
            x = self.conv6(x)
            print(f'Shape before mean {x.shape}')
            x = torch.mean(x, axis = (2,3))
            print(f'Completed Convgru penultimate {x.shape}')
            x = self.dense1(x).type(self.dtype)
            
            return x


class hConvGruResNet(nn.Module):

    def __init__(self):
        
            self.num_classes = 2,
            self.hidden_size = 64, 
            self.kernel_size = 3,
            self.timesteps = 8,
            self.dtype = torch.float32 

            self.activ = nn.ReLU

            self.conv0_1 = nn.Conv2d(3, int(self.hidden_size//2), kernel_size = 3, stride = 2, padding = 1, dtype=self.dtype)
            self.conv0_2 = nn.Conv2d(int(self.hidden_size//2), self.hidden_size, kernel_size = 3, stride = 2, padding = 1, dtype=self.dtype)
            self.conv0_3 = nn.Conv2d(self.hidden_size, self.hidden_size, kernel_size = 3, stride = 2, padding = 1, dtype=self.dtype)
            self.bninp_1 = nn.BatchNorm2d(self.hidden_size//2, momentum=0.9, eps=1e-5, dtype=self.dtype)
            self.bninp_2 = nn.BatchNorm2d(self.hidden_size, momentum=0.9, eps=1e-5, dtype=self.dtype)
            self.bninp_3 = nn.BatchNorm2d(self.hidden_size, momentum=0.9, eps=1e-5, dtype=self.dtype)

            self.rnncell1 = hConvGruCell(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size)

            self.conv1 = nn.Conv2d(self.hidden_size, self.hidden_size*2, kernel_size = 3, stride = 3, padding = 1, dtype=self.dtype)
            self.bn1 = nn.BatchNorm2d(self.hidden_size*2, momentum=0.9, eps=1e-5, dtype=self.dtype)
            self.rnncell2 = hConvGruCell(self.hidden_size*2, self.hidden_size*2, kernel_size=self.kernel_size)

            self.conv2 = nn.Conv2d(self.hidden_size*2, self.hidden_size*4, kernel_size = 3, stride = 3, padding = 1, dtype=self.dtype)
            self.bn2 = nn.BatchNorm2d(self.hidden_size*4, momentum=0.9, eps=1e-5, dtype=self.dtype)
            self.rnncell3 = hConvGruCell(self.hidden_size*4, self.hidden_size*4, kernel_size=self.kernel_size)

            self.conv3 = nn.Conv2d(self.hidden_size*4, self.hidden_size*8, kernel_size = 3, stride = 3, padding = 1, dtype=self.dtype)
            self.bn3 = nn.BatchNorm2d(self.hidden_size*8, momentum=0.9, eps=1e-5, dtype=self.dtype)
            self.rnncell4 = hConvGruCell(self.hidden_size*8, self.hidden_size*8, kernel_size=self.kernel_size)


            self.conv4 = nn.Conv2d(self.hidden_size*8, self.hidden_size*16*2, kernel_size = 1, dtype=self.dtype)

            self.bnop_1 = nn.BatchNorm2d(self.hidden_size*16*2, momentum=0.9, epsilon=1e-5, dtype=self.dtype)
            #self.bnop_2 = nn.BatchNorm(momentum=0.9, epsilon=1e-5, dtype=self.dtype)
            self.bnop_3 = nn.BatchNorm2d(self.hidden_size *16*2, momentum=0.9, eps=1e-5, dtype=self.dtype)
            #self.conv6 = nn.Conv(2048, [1, 1], dtype=self.dtype)
            self.dense1 = nn.Dense(self.num_classes, dtype=self.dtype)


    def forward(self, x):
            x = self.conv0_1(x)
            x = self.activ(self.bninp_1(x))

            x = self.conv0_2(x)
            x = self.activ(self.bninp_2(x))

            n = x.shape[2]
            maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = (n+1)//2)

            x = maxpool(x)
            x = self.conv0_3(x)
            x = self.activ(self.bninp_3(x))

            state = torch.zeros_like(x)
            for i in range(self.timesteps):
                state = self.rnncell1(x, state, timestep = i)

            x = self.conv1(state)
            x = self.activ(self.bn1(x))

            state = torch.zeros_like(x)
            for i in range(self.timesteps):
                state = self.rnncell2(x, state, timestep = i)

            x = self.conv2(state)
            x = self.activ(self.bn2(x))

            state = torch.zeros_like(x)
            for i in range(self.timesteps):
                state = self.rnncell3(x, state, timestep = i)
                
            x = self.conv3(state)
            x = self.activ(self.bn3(x))
                
            state = torch.zeros_like(x)
            for i in range(self.timesteps):
                state = self.rnncell4(x, state, timestep = i)

            x = self.conv4(state)
            x = self.activ(self.bnop_1(x))

            x = torch.mean(x, axis = (2,3))

            x = self.bnop_3(x)
            x = self.dense1(x)

            return x
    

__all__ = []

@register_model
def hgru(pretrained = False, **kwargs):
    model = hConvGru()
    return model

@register_model
def hconvgru_resnet(pretrained = False, **kwargs):
    model = hConvGruResNet()
    return model
			
## changes 
	# g2_t = F.sigmoid(self.bn[i*4+2](self.u2_gate(next_state1))) removed coz of batchnorm