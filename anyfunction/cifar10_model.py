from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from anyf import *

import math


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N,C,H,W = x.size()
        g = self.groups
        return x.view(N,g,C//g,H,W).permute(0,2,1,3,4).reshape(N,C,H,W)


class FoldedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,\
            dilation=1, groups=1, bias=False, fold_ratio=2):
        super(FoldedConv2d, self).__init__()

        assert(in_channels % fold_ratio == 0)
        assert(out_channels % fold_ratio == 0)

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.folded_in_channels = in_channels//fold_ratio
        self.folded_out_channels = out_channels//fold_ratio
        self.fold_ratio = fold_ratio

        self.conv = nn.Conv2d(self.folded_in_channels, self.folded_out_channels, kernel_size,\
            stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        # self.p = PositionalEncoding(self.folded_in_channels, fold_ratio * 8 + 1, base_freq=10000.)
        self.p = UniformRandomPositionalEncoding(self.folded_in_channels, fold_ratio + 1)
        self.shuffler = ShuffleBlock(self.fold_ratio)

    
    def forward(self, x):
        ishape = x.shape
        h_out = math.floor((x.shape[2] + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) / self.conv.stride[0] + 1)
        w_out = math.floor((x.shape[3] + 2 * self.conv.padding[1] - self.conv.dilation[1] * (self.conv.kernel_size[1] - 1) - 1) / self.conv.stride[1] + 1)

        # out = torch.zeros((x.shape[0], self.out_channels, h_out, w_out)).to(x.device)

        # x = self.shuffler(x)
        # x = x.reshape(x.shape[0], self.fold_ratio, self.folded_in_channels, x.shape[2], x.shape[3])
        # for i in range(self.fold_ratio):
        #     pe_offset = self.p.pe[0, i, :self.folded_in_channels].reshape(1, self.folded_in_channels, 1, 1)
        #     # input = x[:, i, :, :, :]
        #     # print(pe_offset.shape)
        #     # print(input.shape)
        #     # print(out[:, i*self.folded_out_channels:(i+1)*self.folded_out_channels, :, :].shape)
        #     # print(i*self.folded_out_channels, (i+1)*self.folded_out_channels)
        #     # print(self.in_channels, self.folded_in_channels, self.out_channels, self.folded_out_channels)
        #     # exit(1)
        #     out[:, i*self.folded_out_channels:(i+1)*self.folded_out_channels, :, :] = self.conv.forward(x[:, i, :, :, :] + pe_offset)
        # return out

        x = self.shuffler(x)
        x = x.reshape(x.shape[0], self.fold_ratio, self.folded_in_channels, x.shape[2], x.shape[3])
        pe_offset = self.p.pe[0, :self.fold_ratio, :self.folded_in_channels].reshape(1, self.fold_ratio, self.folded_in_channels, 1, 1)
        
        x = x + pe_offset

        x = x.reshape(-1, self.folded_in_channels, ishape[2], ishape[3])
        x = self.conv(x)
        x = x.reshape(ishape[0], self.out_channels, h_out, w_out)
        return x


class FoldedDWConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,\
            dilation=1, groups=1, bias=False, fold_ratio=2, d_model=32*32):
        super(FoldedDWConv2d, self).__init__()

        assert(in_channels % fold_ratio == 0)
        assert(out_channels % fold_ratio == 0)
        assert(groups == in_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.d_model = d_model

        self.folded_in_channels = in_channels//fold_ratio
        self.folded_out_channels = out_channels//fold_ratio
        self.fold_ratio = fold_ratio

        self.conv = nn.Conv2d(self.folded_in_channels, self.folded_out_channels, kernel_size,\
            stride=stride, padding=padding, dilation=dilation, groups=self.folded_in_channels, bias=bias)
        self.p = PositionalEncoding(d_model, fold_ratio + 1, base_freq=2.)
        # self.p = UniformRandomPositionalEncoding(d_model, fold_ratio + 1)
    
    def forward(self, x):
        ishape = x.shape
        ih = ishape[2]
        iw = ishape[3]
        h_out = math.floor((x.shape[2] + 2 * self.conv.padding[0] - self.conv.dilation[0] * (self.conv.kernel_size[0] - 1) - 1) / self.conv.stride[0] + 1)
        w_out = math.floor((x.shape[3] + 2 * self.conv.padding[1] - self.conv.dilation[1] * (self.conv.kernel_size[1] - 1) - 1) / self.conv.stride[1] + 1)

        pe_offset = self.p.pe[0, :self.fold_ratio, :ih*iw].reshape(1, self.fold_ratio, -1, ih, iw)
        x = x.reshape(ishape[0], self.fold_ratio, -1, ih, iw)

        # x = x + pe_offset
        x = x.reshape(-1, self.folded_in_channels, ih, iw)

        x = self.conv(x)
        x = x.reshape(ishape[0], self.out_channels, h_out, w_out)
        return x


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1, fold_ratio=2):
        super(Block, self).__init__()
        # self.conv1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False)
        self.conv1 = FoldedDWConv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False, fold_ratio=2)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.conv2 = FoldedConv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False, fold_ratio=fold_ratio)
        self.bn2 = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        return out


class MobileNet(nn.Module):
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, num_classes=10):
        super(MobileNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out



class BasicCNN(nn.Module):
    def __init__(self):
        super(BasicCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        # self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2 = FoldedConv2d(32, 64, 3, fold_ratio=4)
        self.fc1   = nn.Linear(64*6*6, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        return out



def Net(*args, **kwargs):
    # net = BasicCNN()
    net = MobileNet()
    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total parameters: {}".format(pytorch_total_params))
    return net
