import torch
import numpy as np
import torch.nn as nn
from trainer import Trainer
import torch.nn.functional as F

torch.backends.cudnn.enabled = False

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class UnFlatten(nn.Module):
    def __init__(self, H, W):
        super(UnFlatten, self).__init__()
        self.H = H
        self.W = W

    def forward(self, input):
        return input.view(input.shape[0], -1, self.H, self.W)

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

class NonLocalBlock(nn.Module):
    def __init__(self, C):
        super(NonLocalBlock, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.down_size = C // 2
        self.theta = nn.Conv2d(in_channels=C, out_channels=self.down_size, kernel_size=1)
        self.phi = nn.Conv2d(in_channels=C, out_channels=self.down_size, kernel_size=1)
        self.g = nn.Conv2d(in_channels=C, out_channels=self.down_size, kernel_size=1)
        self.w = nn.Conv2d(in_channels=self.down_size, out_channels=C, kernel_size=1)
        nn.init.constant_(self.w.weight, 0)
        nn.init.constant_(self.w.bias, 0)

    def forward(self, x):
        N, C, H, W = x.shape
        theta_x = self.theta(x).view(N, C, -1).permute(0, 2, 1)
        phi_x = self.pool(self.phi(x)).view(N, C, -1)
        #phi_x = self.phi(x).view(N, C, -1)
        y = torch.matmul(theta_x, phi_x) / np.sqrt(phi_x.shape[-1])
        g_x = self.pool(self.g(x)).view(N, C, -1).permute(0, 2, 1)
        #g_x = self.g(x).view(N, C, -1).permute(0, 2, 1)
        #print(y.shape, g_x.shape)
        z = torch.matmul(F.softmax(y, dim=-1), g_x)
        return self.w(z.view(N, H, W, self.down_size).permute(0, 3, 1, 2)) + x

class CNN(Trainer):
    def __init__(self):
        Trainer.__init__(self, train_percentage=0.95, train_batch_size=512)
        layers = [
                ['conv', 3, 64, 3, 1], ['norm'], ['relu'], 
                ['nonlocal', 64],
                #['conv', 3, 64, 3, 1], ['norm'], ['relu'], 
                #['conv', 64, 64, 3, 1], ['norm'], ['relu'], 
                ##['conv', 64, 64, 3, 1], ['norm'], ['relu'],
                ##['seblock'],
                #['maxPool', 2, 2],
                #['conv', 64, 128, 3, 1], ['norm'], ['relu'], 
                #['conv', 128, 128, 3, 1], ['norm'], ['relu'], 
                ##['conv', 128, 128, 3, 1], ['norm'], ['relu'], 
                ##['seblock'],
                ##['nonlocal', 128],
                #['maxPool', 2, 2],
                #['conv', 128, 256, 3, 1], ['norm'], ['relu'], 
                #['conv', 256, 256, 3, 1], ['norm'], ['relu'], 
                ##['conv', 256, 256, 3, 1], ['norm'], ['relu'], 
                ##['seblock'],
                ##['nonlocal', 256],
                #['maxPool', 2, 2],
                #['conv', 256, 512, 3, 1], ['norm'], ['relu'], 
                #['conv', 512, 512, 3, 1], ['norm'], ['relu'], 
                #['conv', 512, 512, 3, 1], ['norm'], ['relu'], 
                #['seblock'],
                #['nonlocal', 512],
                ['avgPool', 32, 1],
                #['conv', 512, 256, 1, 0], ['norm'], ['relu'],
                #['dropout', 0.5],
                #['conv', 256, 256, 1, 0], ['norm'], ['relu'],
                #['dropout', 0.5],
                ['flatten'],
                ['fc', 64, 64], ['relu'],
                ['dropout', 0.5],
                ['fc', 64, self.num_label]
        ]
        activations = []
        for i in range(len(layers)):
            layer = layers[i]
            activation = None
            if layer[0] == 'conv':
                activation = nn.Conv2d(in_channels=layer[1], out_channels=layer[2], kernel_size=layer[3], padding=layer[4])
            elif layer[0] == 'norm':
                activation = nn.BatchNorm2d(layers[i - 1][2])
            elif layer[0] == 'relu':
                activation = nn.ReLU(True)
            elif layer[0] == 'maxPool':
                activation = nn.MaxPool2d(layer[1], layer[2])
            elif layer[0] == 'avgPool':
                activation = nn.AvgPool2d(layer[1], layer[2])
            elif layer[0] == 'dropout':
                activation = nn.Dropout(layer[1])
            elif layer[0] == 'flatten':
                activation = Flatten()
            elif layer[0] == 'fc':
                activation = nn.Linear(layer[1], layer[2])
            elif layer[0] == 'seblock':
                activation = SEBlock(4, self.device)
            elif layer[0] == 'nonlocal':
                activation = NonLocalBlock(layer[1])
            if activation:
                activations.append(activation.to(self.device))
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
model = cnn.train_model(warmup_epochs=5, num_epoch_to_log=1, learning_rate=1e-3, weight_decay=0.001, epochs=50)
cnn.test(model)
