#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List, Tuple

from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from torch.nn.init import xavier_normal_
import warnings
# from ..model import ResidualAttentionBlock 

import copy
from timm.models.layers import DropPath
import random

def drop_token(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    # shape = (x.shape[0],) + (x.shape[1],) + (1,)  # work with diff dim tensors, not just 2D ConvNets
    shape = (1,)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropToken(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropToken, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_token(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def tensor_init(p, init, gain=1, std=1, a=1, b=1):
    if init == 'ortho':
        nn.init.orthogonal_(p)
    elif init == 'uniform':
        nn.init.uniform_(p, a=a, b=b)
    elif init == 'normal':
        nn.init.normal_(p, std=std)
    elif init == 'zero':
        nn.init.zeros_(p)
    elif init == 'he_uniform':
        nn.init.kaiming_uniform_(p, a=a)
    elif init == 'he_normal':
        nn.init.kaiming_normal_(p, a=a)
    elif init == 'xavier_uniform':
        nn.init.xavier_uniform_(p, gain=gain)
    elif init == 'xavier_normal':
        nn.init.xavier_normal_(p, gain=gain)
    elif init == 'trunc_normal':
        nn.init.trunc_normal_(p, std=std)
    else:
        assert NotImplementedError

def tensor_prompt(x, y=None, z=None, w=None, init='xavier_uniform', gain=1, std=1, a=1, b=1):
    if y is None:
        p = torch.nn.Parameter(torch.FloatTensor(x), requires_grad=True)
    elif z is None:
        p = torch.nn.Parameter(torch.FloatTensor(x,y), requires_grad=True)
    elif w is None:
        p = torch.nn.Parameter(torch.FloatTensor(x,y,z), requires_grad=True)
    else:
        p = torch.nn.Parameter(torch.FloatTensor(x,y,z,w), requires_grad=True)

    if p.dim() > 2:
        tensor_init(p[0], init, gain=gain, std=std, a=a, b=b)
        for i in range(1, x): p.data[i] = p.data[0]
    else:
        tensor_init(p, init)
    
    return p

class StochasticAdapter(nn.Module):
    def __init__(self, embed_dim, r=64, init_value=0.1, eval_scale=0.5, drop_path=0, scale_train=True, bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.r = r
        self.bias = bias
        self.scale_train = scale_train
        self.drop_path = drop_path
        self.init_value = init_value
        self.s = nn.Parameter(init_value * torch.ones(1), requires_grad = self.scale_train) if self.scale_train else init_value
        self.loss = torch.zeros([1])
        self.drop_paths = DropPath(drop_path, scale_by_keep=True)
        self.eval_scale = eval_scale

        if embed_dim < r:
            self.d = nn.Linear(embed_dim, r, bias=bias)
            self.u = nn.Linear(r, embed_dim, bias=bias)
            nn.init.xavier_uniform_(self.d.weight)
            nn.init.zeros_(self.u.weight)
            if bias:
                nn.init.zeros_(self.d.bias)
                nn.init.zeros_(self.u.bias)
        else:
            self.f = nn.Linear(embed_dim, embed_dim, bias=bias)
            nn.init.zeros_(self.f.weight)
            if bias:
                nn.init.zeros_(self.f.bias)


    def forward(self, x):
        if self.embed_dim < self.r:
            z = self.u(self.d(x))
        else:
            z = self.f(x)
        z = self.drop_paths(z)

        scale = self.s if self.training else self.s * self.eval_scale
        x = x + z * scale

        return x