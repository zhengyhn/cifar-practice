import torch
import torch.nn as nn
from solver import Solver
from dataset import AbstractDataset, CIFAR10Dataset

class LogisticRegression(nn.Module):
    def __init__(self, dataset: AbstractDataset):
        nn.Module.__init__(self)
        H, W, C = dataset.num_dims()
        self.fc = nn.Linear(H * W * C, dataset.num_labels())

    def forward(self, X):
        X = X.view(len(X), -1)
        predict = torch.sigmoid(self.fc(X))
        return predict

if __name__ == '__main__':
    dataset = CIFAR10Dataset(train_percentage=0.8)
    solver = Solver(dataset, train_batch_size=64)
    model = LogisticRegression(solver.num_feature, solver.num_label)
    solver.train_model(model, learning_rate=2e-5, weight_decay=0.01, epochs=5, warmup_epochs=1, checkpoint='checkpoint/logistic')
    solver.test(model)
    new_model = solver.caribrate(model)
    solver.test_caribrate(new_model)
