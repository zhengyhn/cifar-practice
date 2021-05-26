import torch
import numpy as np
import torch.nn as nn
from trainer import Trainer

class CNN(Trainer):
    def __init__(self):
        Trainer.__init__(self, train_percentage=0.9, train_batch_size=64)
        self.conv10 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=3, padding=1)
        self.conv11 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=3, padding=1)
        self.conv1_bn = nn.BatchNorm2d(12)
        self.pool1 = nn.AvgPool2d(3, 1)
        self.conv20 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5)
        self.conv21 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=7)
        self.conv2_bn = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(3, 1)
        self.fc1 = nn.Linear(18 * 18 * 48, 1024)
        self.fc2 = nn.Linear(1024, self.num_label)

    def map_feature(self, inputs):
        return torch.tensor(inputs)

    def forward(self, X):
        z = self.conv11(self.conv10(X))
        # z = self.conv10(X)
        z = self.conv1_bn(z)
        z = torch.relu(z)
        dropout = nn.Dropout(0.5)
        z = dropout(z)
        a = self.pool1(z)
        z = self.conv21(self.conv20(a))
        # z = self.conv20(a)
        z = self.conv2_bn(z)
        z = torch.relu(z)
        z = dropout(z)
        a = self.pool2(z)
        a = torch.relu(self.fc1(a.reshape(len(X), -1)))
        predict = self.fc2(a)
        return predict

cnn = CNN()
cnn.to(cnn.device)
model = cnn.train_model(num_epoch_to_log=5, learning_rate=1e-3, weight_decay=0, epochs=100)
cnn.test(model)
