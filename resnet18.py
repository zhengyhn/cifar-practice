import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from cnn import CNN

class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride):
        super(ResBlock, self).__init__()
        #self.kernel_size = kernel_size
        #self.channel = channel
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


class ResNet(CNN):
    def __init__(self, num_label):
        self.layers = [
                ['conv', 3, 64, 5, 2], ['norm'], ['relu'],
                ['resblock', 64, 64, 1],
                ['resblock', 64, 64, 1],
                ['resblock', 64, 128, 2],
                ['resblock', 128, 128, 1],
                ['resblock', 128, 256, 2],
                ['resblock', 256, 256, 1],
                ['resblock', 256, 512, 2],
                ['resblock', 512, 512, 1],
                ['avgPool', 4, 1],
                ['conv', 512, 256, 1, 0], ['norm'], ['relu'],
                ['dropout', 0.5],
                ['conv', 256, 256, 1, 0], ['norm'], ['relu'],
                ['dropout', 0.5],
                ['flatten'],
                ['fc', 256, num_label]
        ]
        CNN.__init__(self)

    def build_layer_internal(self, layer):
        activation = None
        if layer[0] == 'resblock':
            activation = ResBlock(*layer[1:])
        return activation

solver = Solver(train_percentage=0.95, train_batch_size=512)
model = ResNet(solver.num_label)
solver.train_model(model, warmup_epochs=5, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=0.001, epochs=70, checkpoint='checkpoint/resnet18')
solver.test(model)
new_model = solver.caribrate(model)
solver.test_caribrate(new_model)
