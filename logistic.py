import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import optimizer

transform=transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ])
trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1000,
                                          shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000,
                                          shuffle=False, num_workers=4)

# images, labels = iter(trainloader).next()
# print(images.shape, labels.shape)
# images = torchvision.utils.make_grid(images)
# plt.matshow(images[1].numpy().T)
#
# plt.show()

class LogisticRegression(nn.Module):
    def __init__(self, num_feature, num_label):
        super(LogisticRegression, self).__init__()
        self.fc = nn.Linear(num_feature, num_label)

    def forward(self, X):
        return torch.sigmoid(self.fc(X))

    def accuracy(self, predict, y):
        t = predict.max(-1)[1] == y
        return torch.mean(t.float())

logistic = LogisticRegression(3 * 32 * 32, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(logistic.parameters(), lr=1e-3)
epochs = 400
for e in range(epochs):
    for i in range(3):
        data = iter(trainloader).next()
    # for i, data in enumerate(trainloader, 0):
        logistic.train()
        inputs, labels = data
        inputs = inputs.reshape(len(inputs), -1)
        # print(inputs.shape)
        # print(labels.shape)
        inputs, labels = Variable(inputs), Variable(labels)

        outputs = logistic(inputs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    if e % 5 == 0:
        test_x, test_labels = iter(testloader).next()
        test_y = logistic(test_x.reshape(len(test_x), -1))
        print("{}, {}, {}".format(e, loss.item(), logistic.accuracy(test_y, test_labels)))
