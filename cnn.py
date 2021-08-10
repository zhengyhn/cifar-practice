import torch
import numpy as np
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        activations = []
        for i in range(len(self.layers)):
            layer = self.layers[i]
            activation = None
            if layer[0] == 'conv':
                activation = nn.Conv2d(in_channels=layer[1], out_channels=layer[2], kernel_size=layer[3], padding=layer[4])
            elif layer[0] == 'norm':
                activation = nn.BatchNorm2d(self.layers[i - 1][2])
            elif layer[0] == 'relu':
                activation = nn.ReLU(True)
            elif layer[0] == 'maxPool':
                activation = nn.MaxPool2d(*layer[1:])
            elif layer[0] == 'avgPool':
                activation = nn.AvgPool2d(*layer[1:])
            elif layer[0] == 'dropout':
                activation = nn.Dropout(*layer[1:])
            elif layer[0] == 'fc':
                activation = nn.Linear(*layer[1:])
            elif layer[0] == 'flatten':
                activation = Flatten()
            a = self.build_layer_internal(layer)
            if a:
                activation = a
            if activation:
                activations.append(activation.to(self.device))
        #print(activations)
        self.module_list = nn.ModuleList(activations)

    def build_layer_internal(self, layer):
        pass

    def forward(self, X):
        z = X
        for module in self.module_list:
            z = module(z)
        return z


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


