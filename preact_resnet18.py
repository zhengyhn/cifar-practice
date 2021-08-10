import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from preact_resnet import PreActResNet, BlockType
from dataset import CIFAR10Dataset

class PreActResNet18(PreActResNet):
    def add_res_layers(self):
        self._make_layers(BlockType.PRE_ACT_BASIC, 64, 64, 2)
        self._make_layers(BlockType.PRE_ACT_BASIC, 64, 128, 2)
        self._make_layers(BlockType.PRE_ACT_BASIC, 128, 256, 2)
        self._make_layers(BlockType.PRE_ACT_BASIC, 256, 512, 2)
        return 512


if __name__ == '__main__':
    dataset = CIFAR10Dataset(train_percentage=0.95)
    solver = Solver(dataset, checkpoint='checkpoint/preact_resnet18', train_batch_size=512)
    model = PreActResNet18(dataset.num_label())
    #solver.train_model(model, warmup_epochs=10, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=1e-4, epochs=100)
    solver.test(model)
    #new_model = solver.caribrate(model)
    #solver.test_caribrate(new_model)
