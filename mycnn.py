import torch
import numpy as np
import torch.nn as nn
from trainer import Trainer

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class CNN(Trainer):
    def __init__(self):
        Trainer.__init__(self, train_percentage=0.95, train_batch_size=512)
        layers = [
                ['conv', 3, 64, 3, 1], ['norm'], ['relu'],
                ['conv', 64, 64, 3, 1], ['norm'], ['relu'],
                ['conv', 64, 64, 3, 1], ['norm'], ['relu'],
                #['dropout', 0.2],
                ['maxPool', 2, 2],
                ['conv', 64, 128, 3, 1], ['norm'], ['relu'],
                ['conv', 128, 128, 3, 1], ['norm'], ['relu'],
                ['conv', 128, 128, 3, 1], ['norm'], ['relu'],
                #['dropout', 0.2],
                ['maxPool', 2, 2],
                ['conv', 128, 256, 3, 1], ['norm'], ['relu'],
                ['conv', 256, 256, 3, 1], ['norm'], ['relu'],
                ['conv', 256, 256, 3, 1], ['norm'], ['relu'],
                #['dropout', 0.2],
                ['maxPool', 2, 2],
                ['conv', 256, 512, 3, 1], ['norm'], ['relu'],
                ['conv', 512, 512, 3, 1], ['norm'], ['relu'],
                ['conv', 512, 512, 3, 1], ['norm'], ['relu'],
                ['avgPool', 4, 1],
                ['conv', 512, 64, 1, 0], ['norm'], ['relu'],
                ['dropout', 0.5],
                ['conv', 64, 64, 1, 0], ['norm'], ['relu'],
                ['dropout', 0.5],
                ['flatten'],
                #['flatten'],
                #['fc', 1 * 1 * 512, 64], ['relu'],
                #['dropout', 0.5],
                #['fc', 64, 64], ['relu'],
                #['dropout', 0.5],
                ['fc', 64, self.num_label]
        ]
        activations = []
        for i in range(len(layers)):
            layer = layers[i]
            activation = None
            if layer[0] == 'conv':
                activation = nn.Conv2d(in_channels=layer[1], out_channels=layer[2], kernel_size=layer[3], padding=layer[4]).to(self.device)
            elif layer[0] == 'norm':
                activation = nn.BatchNorm2d(layers[i - 1][2]).to(self.device)
            elif layer[0] == 'relu':
                activation = nn.ReLU(True)
            elif layer[0] == 'maxPool':
                activation = nn.MaxPool2d(layer[1], layer[2]).to(self.device)
            elif layer[0] == 'avgPool':
                activation = nn.AvgPool2d(layer[1], layer[2]).to(self.device)
            elif layer[0] == 'dropout':
                activation = nn.Dropout(layer[1])
            elif layer[0] == 'flatten':
                activation = Flatten()
            elif layer[0] == 'fc':
                activation = nn.Linear(layer[1], layer[2]).to(self.device)
            activations.append(activation)
        self.module_list = nn.ModuleList(activations)

    def map_feature(self, inputs):
        return torch.tensor(inputs)

    def forward(self, X):
        z = X
        for module in self.module_list:
            z = module(z)
        return z

cnn = CNN()
cnn.to(cnn.device)
model = cnn.train_model(num_epoch_to_log=5, learning_rate=1e-3, weight_decay=0.001, epochs=60)
cnn.test(model)
