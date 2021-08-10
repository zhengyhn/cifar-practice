import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from resnet import ResNet, BlockType

class ResNet101(ResNet):
    def add_res_layers(self):
        self._make_layers(BlockType.BOTTLE_NECK, 64, 64, 3)
        self._make_layers(BlockType.BOTTLE_NECK, 256, 128, 4)
        self._make_layers(BlockType.BOTTLE_NECK, 512, 256, 23)
        self._make_layers(BlockType.BOTTLE_NECK, 1024, 512, 3)
        self.layers.append(['conv', 2048, 512, 1, 0])


solver = Solver(train_percentage=0.95, train_batch_size=128)
model = ResNet101(solver.num_label)
solver.train_model(model, warmup_epochs=10, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=0.001, epochs=70, checkpoint='checkpoint/resnet101')
solver.test(model)
new_model = solver.caribrate(model)
solver.test_caribrate(new_model)
