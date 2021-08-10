import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from stomach_dataset import StomachDataset
from resnet import BlockType, BasicBlock
from torchvision import models
from cnn import CNN
import torch.multiprocessing as mp

class BigResNet(CNN):
    def __init__(self, num_label):
        self.layers = [
            ['conv', 512, 64, 1, 0], ['norm'], ['relu'],
        ]
        self._make_layers(BlockType.BASIC, 64, 64, 2)
        self._make_layers(BlockType.BASIC, 64, 128, 2)
        self._make_layers(BlockType.BASIC, 128, 256, 2)
        self._make_layers(BlockType.BASIC, 256, 512, 2)
        self.layers.extend([
            ['avgPool', 4, 1],
            ['flatten'],
            ['fc', 512, num_label]
        ])
        CNN.__init__(self)

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

if __name__ == '__main__':
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    dataset = StomachDataset()
    trainset, validset, testset = dataset.get()
    solver = Solver(dataset, checkpoint='checkpoint/stomach', train_batch_size=4)
    model = BigResNet(dataset.num_label())
    solver.train_model(model, warmup_epochs=10, num_epoch_to_log=1, learning_rate=1e-3, weight_decay=1e-4, epochs=50)
    solver.test(model)
