import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import copy

class Trainer(nn.Module):
    def __init__(self, train_percentage=0.95, num_feature=3 * 32 * 32, train_batch_size=400):
        super(Trainer, self).__init__()
        self.num_feature = num_feature
        self.num_label = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.RandomCrop(32, padding=4),
            #transforms.ColorJitter(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        # init train set
        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=train_transform)
        #validateset = copy.deepcopy(trainset)
        #random_indexes = np.random.permutation(range(0, len(trainset.data)))
        #num_train = int(len(trainset.data) * train_percentage)
        #num_val = min(len(trainset.data) - num_train, int(num_train * 0.1))
        #train_indexes = random_indexes[:num_train]
        #trainset.data = trainset.data[train_indexes]
        #trainset.targets = list(np.array(trainset.targets)[train_indexes])
        #self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, drop_last=True, pin_memory=True)
        #validate_indexes = random_indexes[num_train:num_train + num_val]
        #validateset.data = validateset.data[validate_indexes]
        #validateset.targets = list(np.array(validateset.targets)[validate_indexes])
        #self.validateloader = torch.utils.data.DataLoader(validateset, batch_size=len(validateset.data), pin_memory=True)

        trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data))
        train_x, train_labels = iter(trainloader).next()
        train_x = train_x.to(self.device)
        train_labels = train_labels.to(self.device)
        num_train = int(len(train_x) * train_percentage)
        num_val = min(len(train_x) - num_train, int(num_train * 0.1))
        random_indexes = np.random.permutation(range(0, len(train_x)))
        train_indexes = random_indexes[:num_train]
        self.trainset = torch.utils.data.TensorDataset(train_x[train_indexes], train_labels[train_indexes])
        self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=train_batch_size,
                                                        shuffle=True, drop_last=True)
        validate_indexes = random_indexes[num_train:num_train + num_val]
        self.validateset = torch.utils.data.TensorDataset(train_x[validate_indexes], train_labels[validate_indexes])
        self.validateloader = torch.utils.data.DataLoader(self.validateset, batch_size=len(validate_indexes))

        # init test set
        testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=len(testset.data),
                                                  shuffle=False, num_workers=2)

    def map_feature(self, inputs):
        inputs = inputs.reshape(len(inputs), -1)
        # inputs, _, _ = torch.pca_lowrank(inputs, self.num_feature)
        return inputs.to(self.device)

    def train_model(self, learning_rate=1e-4, epochs=100, num_epoch_to_log=5, weight_decay=0):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0,
                                                               last_epoch=-1)
        for epoch in range(epochs):
            self.train()
            for inputs, labels in self.trainloader:
                inputs = self.map_feature(inputs)
                inputs, labels = Variable(inputs).to(self.device), Variable(labels).to(self.device)
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            scheduler.step()
            if epoch % num_epoch_to_log == 0:
                with torch.no_grad():
                    self.eval()
                    val_x, val_labels = iter(self.validateloader).next()
                    val_x = self.map_feature(val_x)
                    val_y = self(val_x.to(self.device))
                    print("epoch: {}, loss: {}, train acc: {}, val acc: {}"
                          .format(epoch, loss.item(), self.get_accuracy(outputs, labels),
                              self.get_accuracy(val_y, val_labels)))
        return self

    def get_accuracy(self, predict, labels):
        y = predict.max(-1)[1]
        return np.sum((labels.cpu().numpy() == y.cpu().numpy()).astype(int)) / len(labels)

    def test(self, model):
        test_x, test_labels = iter(self.testloader).next()
        test_x = self.map_feature(test_x)
        y = model(test_x.to(self.device))
        predict = y.max(-1)[1].cpu().numpy()
        accuracy = np.sum((test_labels.cpu().numpy() == np.array(predict)).astype(int)) / len(test_labels)
        print("Test accuracy: {}".format(accuracy))
