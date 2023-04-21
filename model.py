import ai.model as m
import torch
from torch import nn
import torch.nn.functional as F


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MODELS
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def hardcoded(learn_greyscale=True):
    return m.Model(m.seq(
        ToGrey(learn_greyscale),
        Sobel(),
        AbsSum(),
        ToRGB(),
    ))


def simplest():
    return m.Model(m.seq(
        ToGrey(),
        m.conv(1, 2, padtype='replicate'),
        AbsSum(),
        ToRGB(),
    ))


def lightest():
    return m.Model(m.seq(
        ToGrey(False),
        LightConv(),
        AbsSum(False),
        ToRGB(),
    ))


def general(nc_in=3, nc_out=3, k=3, actv='tanh', padtype='replicate'):
    return m.Model(m.conv(nc_in, nc_out, k=k, actv=actv, padtype=padtype))
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# MODULES
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class ToGrey(m.Module):
    def __init__(s, learn=True):
        super().__init__()
        s._learn = learn
        if learn:
            s._net = m.conv(3, 1, k=1)

    def forward(s, img):
        if s._learn:
            return s._net(img)
        return torch.sum(img, dim=1, keepdim=True) / 3.


class ToRGB(m.Module):
    def forward(s, x):
        return x.repeat(1, 3, 1, 1)


class AbsSum(m.Module):
    def __init__(s, learn=True):
        super().__init__()
        s._learn = learn
        s.init_params()

    def init_params(s):
        s._scale = nn.Parameter(torch.tensor(1.)) if s._learn else .35
        s._offset = nn.Parameter(torch.tensor(0.)) if s._learn else -.87

    def forward(s, x):
        x = torch.sum(torch.abs(x), dim=1, keepdim=True)
        return s._scale * x + s._offset


class StaticConv(m.Module):
    def __init__(s, w, pad=(1,1,1,1)):
        super().__init__()
        s.register_buffer('_w', torch.tensor(w))
        s._pad = pad

    def forward(s, x):
        return F.conv2d(F.pad(x, s._pad, 'replicate'), s._w)


class Sobel(StaticConv):
    def __init__(s):
        super().__init__([
            [[
                [-1., 0., 1.],
                [-2., 0., 2.],
                [-1., 0., 1.],
            ]],
            [[
                [-1., -2., -1.],
                [0., 0., 0.],
                [1., 2., 1.],
            ]],
        ])


class LightConv(m.Module):
    def __init__(s):
        super().__init__()
        s.init_params()

    def init_params(s):
        s._a = nn.Parameter(torch.rand(3))
        s._b = nn.Parameter(torch.rand(3))

    def forward(s, x):
        w = torch.cat([
            torch.outer(s._a, s._b).reshape(1, 1, 3, 3),
            torch.outer(s._b, s._a).reshape(1, 1, 3, 3),
        ], dim=0)
        return F.conv2d(F.pad(x, (1,1,1,1), 'replicate'), w)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
