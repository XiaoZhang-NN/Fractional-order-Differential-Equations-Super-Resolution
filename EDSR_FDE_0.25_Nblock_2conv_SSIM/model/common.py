import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        self.requires_grad = False


############################################调用fde#########################################
class ResBlock_fde(nn.Module):
    def __init__(
        self, conv, n_feats, kernel_size, alpha, num_for,
        bias=True, act=nn.PReLU(1, 0.25), res_scale=1):

        super(ResBlock_fde, self).__init__()

        self.conv1 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu1 = nn.PReLU(n_feats, 0.25)
        self.conv2 = conv(n_feats, n_feats, kernel_size, bias=bias)
        self.relu2 = nn.PReLU(n_feats, 0.25)
        self.h = 0.25        
        self.alpha = alpha
        self.alpha_index = int(self.alpha*10-1)
        self.f = 0
        self.yn_p = 0
        self.num_for = num_for
        self.a1 = (self.h**self.alpha)/(self.alpha*(self.alpha+1))

    def forward(self, x):
        #有for循环尽量不要用张量，要考虑并行化，可能会把第一个维度当作batchsize并行，有for尽量改用列表存储
        #学会用try except
        #各种列表、张量相互切换着调用
        #多百度，用一个关键词或者整个报错贴上去
        # ff = torch.zeros(self.num_for , x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        # yn = torch.zeros(self.num_for + 1 , x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        self.yn_p = 0
        self.f = 0
        ff=[[] for i in range(self.num_for)] 
        yn=[[] for i in range(self.num_for+1)] 
        ###########################################################################################
        self.gamma = [9.5135, 4.5908, 2.9916, 2.2182, 1.7725, 1.4892, 1.2981, 1.1642, 1.0686]
        ##########################################1./gamma#########################################
        self.gamma_1 = [0.1051, 0.2178 ,0.3343 ,0.4508, 0.5642, 0.6715, 0.7704, 0.8589, 0.9358]

        yn[0] = x
        for i in range (self.num_for):

            a = [[] for _ in range(i+1)] 
            b = [[] for _ in range(i+1)] 
            for m in range(i+1):

                if m == 0:
                    a[m] = self.a1*(i**(self.alpha+1) - (i-self.alpha)*((i+1)**self.alpha))
                else:
                    a[m] = ((i-m+2)**(self.alpha+1))- 2*((i-m+1)**(self.alpha+1))+ ((i-m)** (self.alpha+1))

                b[m] = (self.h**self.alpha)/self.alpha*(((i+1-m)**self.alpha)- ((i-m)** self.alpha))

                ff[m] = self.conv2(self.relu2(self.conv1(self.relu1(yn[m]))))

                self.f = self.f + a[m] * ff[m]
                self.yn_p = self.yn_p + ff[m] * b[m]

            f_p = self.conv2(self.relu2(self.conv1(self.relu1(yn[0] + self.gamma_1[self.alpha_index]*self.yn_p))))
            yn[i+1] = yn[0] + self.gamma_1[self.alpha_index]* (self.f + self.a1 * f_p) 
        return yn[self.num_for]

############################################调用fde#########################################

class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feats))

                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feats))

            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)

