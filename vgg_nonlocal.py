import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from cnn import CNN

torch.backends.cudnn.enabled = False

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


class VggNonLocalNet(CNN):
    def __init__(self, num_label):
        self.layers = [
                ['conv', 3, 64, 3, 1], ['norm'], ['relu'],
                ['conv', 64, 64, 3, 1], ['norm'], ['relu'],
                ['nonlocal', 64],
                ['maxPool', 2, 2],
                ['conv', 64, 128, 3, 1], ['norm'], ['relu'],
                ['conv', 128, 128, 3, 1], ['norm'], ['relu'],
                ['conv', 128, 128, 3, 1], ['norm'], ['relu'],
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
        if layer[0] == 'nonlocal':
            activation = NonLocalBlock(layer[1])
        return activation

solver = Solver(train_percentage=0.95, train_batch_size=512)
model = VggNonLocalNet(solver.num_label)
solver.train_model(model, warmup_epochs=10, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=0.001, epochs=70, checkpoint='checkpoint/vgg_nonlocal')
solver.test(model)
new_model = solver.caribrate(model)
solver.test_caribrate(new_model)
