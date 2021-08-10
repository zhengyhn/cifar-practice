import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from resnet import ResNet, BlockType

class ResNet34(ResNet):
    def add_res_layers(self):
        self._make_layers(BlockType.BASIC, 64, 64, 3)
        self._make_layers(BlockType.BASIC, 64, 128, 4)
        self._make_layers(BlockType.BASIC, 128, 256, 6)
        self._make_layers(BlockType.BASIC, 256, 512, 3)

solver = Solver(train_percentage=0.95, train_batch_size=512)
model = ResNet34(solver.num_label)
solver.train_model(model, warmup_epochs=10, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=0.00, epochs=150, checkpoint='checkpoint/resnet34')
solver.test(model)
new_model = solver.caribrate(model)
solver.test_caribrate(new_model)
