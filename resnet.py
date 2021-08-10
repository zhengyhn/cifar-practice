import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from cnn import CNN
from enum import Enum

class BlockType(Enum):
    BASIC = 'res_basic',
    BOTTLE_NECK = 'res_bottle',


class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(True)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        a = self.relu(self.norm1(self.conv1(x)))
        a = self.norm2(self.conv2(a))
        a += self.shortcut(x)
        a = self.relu(a)
        return a


class BottleNeckBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(BottleNeckBlock, self).__init__()
        expand_channel = out_channel * 4
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, stride=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=stride, padding=1, bias=True)
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=expand_channel, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(True)
        self.norm1 = nn.BatchNorm2d(out_channel)
        self.norm2 = nn.BatchNorm2d(out_channel)
        self.norm3 = nn.BatchNorm2d(expand_channel)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channel != expand_channel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels=in_channel, out_channels=expand_channel, kernel_size=1, stride=stride, bias=True),
                nn.BatchNorm2d(expand_channel)
            )

    def forward(self, x):
        a = self.relu(self.norm1(self.conv1(x)))
        a = self.relu(self.norm2(self.conv2(a)))
        a = self.norm3(self.conv3(a))
        a += self.shortcut(x)
        a = self.relu(a)
        return a


class ResNet(CNN):
    def __init__(self, num_label):
        self.layers = [
            ['conv', 3, 64, 3, 1], ['norm'], ['relu'],
        ]
        out_dim = self.add_res_layers()
        self.layers.extend([
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
            if type == BlockType.BASIC:
                self.layers.append([type, out_channel, out_channel, 1])
            elif type == BlockType.BOTTLE_NECK:
                self.layers.append([type, out_channel * 4, out_channel, 1])

    def build_layer_internal(self, layer):
        activation = None
        if layer[0] == BlockType.BASIC:
            activation = BasicBlock(*layer[1:])
        if layer[0] == BlockType.BOTTLE_NECK:
            activation = BottleNeckBlock(*layer[1:])
        return activation

    def extract_feature(self, X):
        z = X
        for module in self.module_list[:-2]:
            z = module(z)
        return z


