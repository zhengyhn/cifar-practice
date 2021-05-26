import torch
import torch.nn as nn
from trainer import Trainer

class LogisticRegression(Trainer):
    def __init__(self):
        Trainer.__init__(self, train_percentage=0.8, train_batch_size=64)
        self.fc = nn.Linear(self.num_feature, self.num_label)

    def forward(self, X):
        predict = torch.sigmoid(self.fc(X))
        # predict = torch.relu(self.fc(X))
        return predict

logistic = LogisticRegression()
logistic.to(logistic.device)
model = logistic.train_model(learning_rate=3e-5, weight_decay=0.01)
logistic.test(model)
