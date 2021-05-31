import torch
import numpy as np
import torch.nn as nn
from trainer import Trainer

class CNN(Trainer):
    def __init__(self):
        Trainer.__init__(self, train_percentage=0.99, train_batch_size=1024)
        self.conv10 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=9, padding=4)
        self.conv11 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=7, padding=3)
        self.conv1_bn = nn.BatchNorm2d(12)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv20 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, padding=2)
        self.conv21 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=3, padding=1)
        self.conv2_bn = nn.BatchNorm2d(48)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(8 * 8 * 48, 1024)
        self.fc2 = nn.Linear(1024, self.num_label)

    def map_feature(self, inputs):
        return torch.tensor(inputs)

    def forward(self, X):
        z = torch.relu(self.conv10(X))
        z = self.conv11(z)
        # z = self.conv10(X)
        z = self.conv1_bn(z)
        z = torch.relu(z)
        dropout = nn.Dropout(0.5)
        z = dropout(z)
        z = self.pool1(z)
        z = torch.relu(self.conv20(z))
        z = self.conv21(z)
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
model = cnn.train_model(num_epoch_to_log=1, learning_rate=1e-3, weight_decay=0.01, epochs=30)
cnn.test(model)
