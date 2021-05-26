import torch
import numpy as np
import torch.nn as nn
from trainer import Trainer

class MLP(Trainer):
    def __init__(self):
        Trainer.__init__(self, train_batch_size=64)
        self.hiddens = [nn.Linear(self.num_feature, 1024).to(self.device)]
        self.num_hidden = len(self.hiddens)
        self.fc = nn.Linear(self.hiddens[-1].out_features, self.num_label)

    def forward(self, X):
        a = torch.relu(self.hiddens[0](X))
        predict = self.fc(a)
        return predict

mlp = MLP()
mlp.to(mlp.device)
model = mlp.train_model(epochs=100, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=0)
mlp.test(model)
