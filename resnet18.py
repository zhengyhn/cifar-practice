import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from resnet import ResNet, BlockType

class ResNet18(ResNet):
    def add_res_layers(self):
        self._make_layers(BlockType.BASIC, 64, 64, 2)
        self._make_layers(BlockType.BASIC, 64, 128, 2)
        self._make_layers(BlockType.BASIC, 128, 256, 2)
        self._make_layers(BlockType.BASIC, 256, 512, 2)
        return 512

if __name__ == '__main__':
    solver = Solver(train_percentage=0.95, train_batch_size=512)
    model = ResNet18(solver.num_label)
    solver.train_model(model, warmup_epochs=10, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=1e-4, epochs=100, checkpoint='checkpoint/resnet18')
    solver.test(model)
    new_model = solver.caribrate(model)
    solver.test_caribrate(new_model)
