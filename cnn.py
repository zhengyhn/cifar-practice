import torch
import numpy as np
import torch.nn as nn
from trainer import Trainer

class CNN(Trainer):
    def __init__(self):
        Trainer.__init__(self, train_percentage=0.8, train_batch_size=4)
        self.conv11 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=7, padding=3)
        self.conv12 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv21 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 8 * 24, 256)
        self.fc2 = nn.Linear(256, self.num_label)

    def map_feature(self, inputs):
        return torch.tensor(inputs)

    def forward(self, X):
        z = self.conv12(self.conv11(X))
        # dropout = nn.Dropout(0.8)
        # z = dropout(z)
        a = self.pool1(torch.relu(z))
        z = self.conv21(a)
        a = self.pool2(torch.relu(z))
        predict = self.fc2(torch.relu(self.fc1(a.reshape(len(X), -1))))
        return predict

cnn = CNN()
cnn.to(cnn.device)
model = cnn.train_model(num_epoch_to_log=1, learning_rate=1e-3, weight_decay=0, epochs=20)
cnn.test(model)
