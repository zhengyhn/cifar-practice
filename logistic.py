import torch
import torch.nn as nn
from solver import Solver

class LogisticRegression(nn.Module):
    def __init__(self, num_feature, num_label):
        nn.Module.__init__(self)
        self.fc = nn.Linear(num_feature, num_label)

    def forward(self, X):
        X = X.view(len(X), -1)
        predict = torch.sigmoid(self.fc(X))
        return predict

solver = Solver(train_percentage=0.8, train_batch_size=64)
model = LogisticRegression(solver.num_feature, solver.num_label)
solver.train_model(model, learning_rate=2e-5, weight_decay=0.01, epochs=2, warmup_epochs=1, checkpoint='checkpoint/logistic')
solver.test(model)
new_model = solver.caribrate(model)
solver.test_caribrate(new_model)
