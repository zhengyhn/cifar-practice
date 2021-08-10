import torch
import numpy as np
import torch.nn as nn
from solver import Solver

class MLP(nn.Module):
    def __init__(self, num_feature, num_label):
        nn.Module.__init__(self)
        self.hidden = nn.Linear(num_feature, 1024)
        self.fc = nn.Linear(self.hidden.out_features, num_label)

    def forward(self, X):
        X = X.view(len(X), -1)
        a = torch.relu(self.hidden(X))
        predict = self.fc(a)
        return predict

if __name__ == '__main__':
    solver = Solver(train_batch_size=64)
    model = MLP(solver.num_feature, solver.num_label)
    solver.train_model(model, epochs=20, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=0, checkpoint='checkpoint/mlp')
    solver.test(model)
    new_model = solver.caribrate(model)
    solver.test_caribrate(new_model)
