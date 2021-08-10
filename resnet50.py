import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from solver import Solver
from resnet import ResNet, BlockType

class ResNet50(ResNet):
    def add_res_layers(self):
        self._make_layers(BlockType.BOTTLE_NECK, 64, 64, 3)
        self._make_layers(BlockType.BOTTLE_NECK, 256, 128, 4)
        self._make_layers(BlockType.BOTTLE_NECK, 512, 256, 6)
        self._make_layers(BlockType.BOTTLE_NECK, 1024, 512, 3)
        return 2048


solver = Solver(train_percentage=0.95, train_batch_size=512)
model = ResNet50(solver.num_label)
solver.train_model(model, warmup_epochs=10, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=1e-4, epochs=200, checkpoint='checkpoint/resnet50')
solver.test(model)
new_model = solver.caribrate(model)
solver.test_caribrate(new_model)
