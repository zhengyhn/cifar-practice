import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from cnn import CNN
from enum import Enum
import random

class BlockType(Enum):
    PRE_ACT_BASIC = 'pre_act_basic',


class PreActBasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(PreActBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(True)
        self.norm1 = nn.BatchNorm2d(in_channel)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.shortcut = None
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        a = self.relu(self.norm1(x))
        shortcut = self.shortcut(a) if self.shortcut else x
        a = self.conv1(a)
        a = self.conv2(self.relu(self.norm2(a)))
        a += shortcut
        #a = self.relu(self.norm1(x))
        #shortcut1 = self.shortcut(a) if self.shortcut else x
        #a = self.conv1(a)
        #a += shortcut1
        #x = a
        #a = self.conv2(self.relu(self.norm2(x)))
        #a += shortcut1 + x
        return a


class PreActResNet(CNN):
    def __init__(self, num_label):
        self.layers = [
            ['conv', 3, 64, 3, 1],
        ]
        out_dim = self.add_res_layers()
        self.layers.extend([
            #['norm'], ['relu'],
            ['avgPool', 4, 1],
            #['conv', 512, 256, 1, 0], ['norm'], ['relu'],
            #['dropout', 0.5],
            #['conv', 256, 256, 1, 0], ['norm'], ['relu'],
            #['dropout', 0.5],
            ['flatten'],
            ['fc', out_dim, num_label]
        ])
        CNN.__init__(self)

    def add_res_layers(self):
        pass

    def _make_layers(self, type, in_channel, out_channel, num_layer):
        first_stride = 1
        if in_channel != out_channel:
            first_stride = 2
        self.layers.append([type, in_channel, out_channel, first_stride])
        for i in range(num_layer - 1):
            if type == BlockType.PRE_ACT_BASIC:
                self.layers.append([type, out_channel, out_channel, 1])
            #elif type == BlockType.BOTTLE_NECK:
            #    self.layers.append([type, out_channel * 4, out_channel, 1])

    def build_layer_internal(self, layer):
        activation = None
        if layer[0] == BlockType.PRE_ACT_BASIC:
            activation = PreActBasicBlock(*layer[1:])
        return activation

