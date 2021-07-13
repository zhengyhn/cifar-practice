import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math
from apex import amp

from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

class LabelSmoothCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, label, smoothing=0.1):
        pred = F.softmax(pred, dim=1)
        one_hot_label = F.one_hot(label, pred.size(1)).float()
        smoothed_one_hot_label = (1.0 - smoothing) * one_hot_label + smoothing / pred.size(1)
        loss = (-torch.log(pred)) * smoothed_one_hot_label
        return loss.sum(axis=1, keepdim=False).mean()

class Trainer(nn.Module):
    def __init__(self, train_percentage=0.95, num_feature=3 * 32 * 32, train_batch_size=400):
        super(Trainer, self).__init__()
        self.num_feature = num_feature
        self.num_label = 10
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(),
            transforms.RandomCrop(32, padding=2),
            #transforms.RandomRotation((5, -5)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.2, 0.2, 0.2)),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=train_transform)
        # load from disk
        validateset = copy.deepcopy(trainset)
        random_indexes = np.random.permutation(range(0, len(trainset.data)))
        num_train = int(len(trainset.data) * train_percentage)
        num_val = min(len(trainset.data) - num_train, int(num_train * 0.1))
        num_train = len(trainset.data) - num_val
        train_indexes = random_indexes[:num_train]
        trainset.data = trainset.data[train_indexes]
        trainset.targets = list(np.array(trainset.targets)[train_indexes])
        #self.trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_batch_size, shuffle=True, drop_last=True, pin_memory=True)
        self.trainloader = DataLoaderX(trainset, batch_size=train_batch_size, shuffle=True, drop_last=True, pin_memory=True, num_workers=8)

        validate_indexes = random_indexes[num_train:num_train + num_val]
        validateset.data = validateset.data[validate_indexes]
        validateset.targets = list(np.array(validateset.targets)[validate_indexes])
        #self.validateloader = torch.utils.data.DataLoader(validateset, batch_size=len(validateset.data), pin_memory=True, num_workers=4)
        self.validateloader = DataLoaderX(validateset, batch_size=len(validateset.data), pin_memory=True, num_workers=8)

        # load from memory
        #trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(trainset.data), shuffle=True)
        #train_x, train_labels = iter(trainloader).next()
        #train_x = train_x.to(self.device)
        #train_labels = train_labels.to(self.device)
        #num_train = int(len(train_x) * train_percentage)
        #num_val = min(len(train_x) - num_train, int(num_train * 0.01))
        #num_train = len(trainset.data) - num_val
        #self.trainset = torch.utils.data.TensorDataset(train_x[:num_train], train_labels[:num_train])
        #self.trainloader = torch.utils.data.DataLoader(self.trainset, batch_size=train_batch_size,
        #                                                shuffle=True, drop_last=True)
        #self.validateset = torch.utils.data.TensorDataset(train_x[num_train:num_train + num_val],
        #                                                  train_labels[num_train:num_train + num_val])
        #self.validateloader = torch.utils.data.DataLoader(self.validateset, batch_size=num_val)

        # init test set
        testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform)
        self.testloader = torch.utils.data.DataLoader(testset, batch_size=512,
                                                  shuffle=False, num_workers=2)

    def map_feature(self, inputs):
        inputs = inputs.reshape(len(inputs), -1)
        # inputs, _, _ = torch.pca_lowrank(inputs, self.num_feature)
        return inputs.to(self.device)

    def train_model(self, learning_rate=1e-4, epochs=100, num_epoch_to_log=5, weight_decay=0, warmup_epochs=10):
        criterion = nn.CrossEntropyLoss()
        #criterion = LabelSmoothCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)
        warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else \
                0.5 * (math.cos((epoch - warmup_epochs) / (epochs - warmup_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0,
        #                                                       last_epoch=-1)
        #model, optimizer = amp.initialize(self, optimizer, opt_level='O1')
        model = self
        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
        for epoch in range(epochs):
            self.train()
            for inputs, labels in self.trainloader:
                inputs = self.map_feature(inputs)
                inputs, labels = Variable(inputs).to(self.device), Variable(labels).to(self.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                optimizer.zero_grad()
                loss.backward()
                #with amp.scale_loss(loss, optimizer) as scaled_loss:
                #    scaled_loss.backward()
                optimizer.step()
            scheduler.step(epoch)
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
        correct = 0
        total = 0
        with torch.no_grad():
            self.eval()
            for test_x, test_labels in self.testloader:
                test_x = self.map_feature(test_x)
                y = model(test_x.to(self.device))
                predict = y.max(-1)[1].cpu().numpy()
                correct += np.sum((test_labels.cpu().numpy() == np.array(predict)).astype(int))
                total += len(test_labels)
        print("Test accuracy: {}".format(correct / total))
