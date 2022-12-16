import torch
import torch.nn as nn

from tsai.imports import Module
from tsai.models.layers import ConvBlock, BN1d
from tsai.models.utils import Squeeze, Add

class ResBlock(Module):
    def __init__(self, ni, nf, kss=[7, 5, 3]):
        self.convblock1 = ConvBlock(ni, nf, kss[0])
        self.convblock2 = ConvBlock(nf, nf, kss[1])
        self.convblock3 = ConvBlock(nf, nf, kss[2], act=None)

        # expand channels for the sum if necessary
        self.shortcut = BN1d(ni) if ni == nf else ConvBlock(ni, nf, 1, act=None)
        self.add = Add()
        self.act = nn.ReLU()

    def forward(self, x):
        res = x
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.add(x, self.shortcut(res))
        x = self.act(x)
        return x

class ResNetBodyNoChannelPool(Module):
    def __init__(self, c_in):
        nf = 64
        kss=[7, 5, 3]
        # 3， 64
        self.resblock1 = ResBlock(c_in, nf, kss=kss)
        # 64, 128
        self.resblock2 = ResBlock(nf, nf * 2, kss=kss)
        # 128, 128
        self.resblock3 = ResBlock(nf * 2, nf * 2, kss=kss)

    def forward(self, x):
        #print("0 x shape : ", x.shape)
        x = self.resblock1(x)
        #print("1 x shape : ", x.shape)
        x = self.resblock2(x)
        #print("2 x shape : ", x.shape)
        x = self.resblock3(x)
        #print("3 x shape : ", x.shape)
        return x


class ResNetBody(Module):
    def __init__(self, c_in):
        self.body = ResNetBodyNoChannelPool(c_in)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.squeeze = Squeeze(-1)

    def forward(self, x):
        x = self.body(x)
        x = self.squeeze(self.gap(x))
        return x

class SegmentationHead(Module):
    # c_out = 1, output_len = 45
    def __init__(self, c_out, output_len, kss=[3, 3, 3]):
        self.convblock1 = ConvBlock(128, 64, kss[0])
        self.convblock2 = ConvBlock(64, 64, kss[1])
        self.convblock3 = ConvBlock(64, c_out, kss[2], act=None)

        self.upsample = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Upsample(size=200, mode='linear')
        )


    def forward(self, x):
        #print("head 0 shape : ", x.shape)
        x = self.convblock1(x)
        #print("head 1 shape : ", x.shape)
        x = self.convblock2(x)
        #print("head 2 shape : ", x.shape)
        x = self.convblock3(x)
        #print("head 3 shape : ", x.shape)
        x = self.upsample(x)
        #print("head 4 shape : ", x.shape)
        #print("head 5 shape : ", x.squeeze().shape)
        return x.squeeze()