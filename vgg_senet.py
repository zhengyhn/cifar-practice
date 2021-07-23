import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from cnn import CNN

class Scale(nn.Module):
    def __init__(self, origin):
        super(Scale, self).__init__()
        self.origin = origin

    def forward(self, input):
        #return input.reshape(input.shape[0], input.shape[1], 1, 1) * self.origin
        return input * self.origin


class SEBlock(nn.Module):
    def __init__(self, ratio, device):
        super(SEBlock, self).__init__()
        self.ratio = ratio
        self.device = device

    def forward(self, input):
        _, C, H, W = input.shape
        down_size = int(C // self.ratio)
        modules = [
                nn.AvgPool2d(kernel_size=H),
                nn.Conv2d(in_channels=C, out_channels=down_size, kernel_size=1),
                #nn.BatchNorm2d(down_size),
                nn.ReLU(True),
                nn.Conv2d(in_channels=down_size, out_channels=C, kernel_size=1),
                #nn.BatchNorm2d(C),
                nn.Sigmoid(),
                Scale(input),
        ]
        output = input
        for module in modules:
            output = module.to(self.device)(output)
        return output


class VggSENet(CNN):
    def __init__(self, num_label):
        self.layers = [
                ['conv', 3, 64, 3, 1], ['norm'], ['relu'], 
                ['conv', 64, 64, 3, 1], ['norm'], ['relu'],
                ['seblock', 4],
                ['maxPool', 2, 2],
                ['conv', 64, 128, 3, 1], ['norm'], ['relu'],
                ['conv', 128, 128, 3, 1], ['norm'], ['relu'],
                ['conv', 128, 128, 3, 1], ['norm'], ['relu'],
                ['seblock', 4],
                ['maxPool', 2, 2],
                ['conv', 128, 256, 3, 1], ['norm'], ['relu'],
                ['conv', 256, 256, 3, 1], ['norm'], ['relu'],
                ['conv', 256, 256, 3, 1], ['norm'], ['relu'],
                ['maxPool', 2, 2],
                ['conv', 256, 512, 3, 1], ['norm'], ['relu'],
                ['conv', 512, 512, 3, 1], ['norm'], ['relu'],
                ['conv', 512, 512, 3, 1], ['norm'], ['relu'],
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
        if layer[0] == 'seblock':
            activation = SEBlock(layer[1], self.device)
        return activation

solver = Solver(train_percentage=0.95, train_batch_size=512)
model = VggSENet(solver.num_label)
solver.train_model(model, warmup_epochs=10, num_epoch_to_log=5, 
        learning_rate=1e-3, weight_decay=0.001, epochs=70, checkpoint='checkpoint/vgg_senet')
solver.test(model)
new_model = solver.caribrate(model)
solver.test_caribrate(new_model)
