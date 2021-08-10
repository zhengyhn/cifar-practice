import torch
import numpy as np
import torch.nn as nn
from solver import Solver
import torch.nn.functional as F
from cnn import CNN
from dataset import AbstractDataset, CIFAR10Dataset
import math

class Vgg(CNN):
    def __init__(self, dataset: AbstractDataset):
        H, W, C = dataset.num_dims()
        out_dim = H // 8
        self.layers = [
                ['conv', C, 64, 3, 1], ['norm'], ['relu'],
                ['conv', 64, 64, 3, 1], ['norm'], ['relu'],
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
                ['avgPool', out_dim, 1],
                ['flatten'],
                ['fc', 512, dataset.num_labels()]
        ]
        CNN.__init__(self)

if __name__ == '__main__':
    dataset = CIFAR10Dataset(train_percentage=0.95)
    solver = Solver(dataset, train_batch_size=512)
    model = Vgg(dataset.num_label())
    solver.train_model(model, epochs=100, warmup_epochs=10, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=1e-4, checkpoint='checkpoint/vgg')
    solver.test(model)
    #new_model = solver.caribrate(model)
    #solver.test_caribrate(new_model)
