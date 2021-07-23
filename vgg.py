import torch
import numpy as np
import torch.nn as nn
from solver import Solver
import torch.nn.functional as F
from cnn import CNN

class Vgg(CNN):
    def __init__(self, num_label):
        self.layers = [
                ['conv', 3, 64, 3, 1], ['norm'], ['relu'],
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
                ['avgPool', 4, 1],
                ['conv', 512, 256, 1, 0], ['norm'], ['relu'],
                ['dropout', 0.5],
                ['conv', 256, 256, 1, 0], ['norm'], ['relu'],
                ['dropout', 0.5],
                ['flatten'],
                ['fc', 256, num_label]
        ]
        CNN.__init__(self)

solver = Solver(train_percentage=0.95, train_batch_size=512)
model = Vgg(solver.num_label)
solver.train_model(model, epochs=10, warmup_epochs=5, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=0.001, checkpoint='checkpoint/vgg')
solver.test(model)
new_model = solver.caribrate(model)
solver.test_caribrate(new_model)
