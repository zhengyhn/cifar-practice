import torch
import numpy as np
import torch.nn as nn
from trainer import Trainer

class MLP(Trainer):
    def __init__(self):
        Trainer.__init__(self, train_batch_size=400)
        # self.hiddens = [nn.Linear(self.num_feature, 1024), nn.BatchNorm1d(1024), nn.Linear(1024, 512)]
        # self.hiddens = [nn.Linear(self.num_feature, 1024)]
        self.hiddens = [nn.Linear(self.num_feature, 1024).to(self.device)]
        self.num_hidden = len(self.hiddens)
        self.fc = nn.Linear(self.hiddens[-1].out_features, self.num_label)
        # self.fc = nn.Linear(self.num_feature, self.num_label)

    def forward(self, X):
        # predict = torch.sigmoid(self.fc(X))
        a = torch.relu(self.hiddens[0](X))
        for i in range(1, self.num_hidden):
            # dropout = nn.Dropout(0.5)
            # a = self.hiddens[i](dropout(a))
            a = self.hiddens[i](a)
            # bn = torch.nn.BatchNorm1d(a.shape[1], affine=True)
            # a = torch.relu(bn(a))
            a = torch.relu(a)
        # predict = torch.softmax(self.fc(a), dim=1)
        predict = self.fc(a)
        return predict

mlp = MLP()
mlp.to(mlp.device)
model = mlp.train_model(num_epoch_to_log=5, learning_rate=1e-3)
mlp.test(model)
