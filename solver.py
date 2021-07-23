import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import copy
import math
#from apex import amp
from abc import abstractmethod, ABC
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from enum import Enum
from temperature_scaling import TemperatureScalingModel


class Solver():
    def __init__(self, train_percentage=0.95, num_feature=3 * 32 * 32, train_batch_size=400):
        #super(Trainer, self).__init__()
        self.num_feature = num_feature
        self.num_label = 10
        self.train_percentage = train_percentage
        self.train_batch_size = train_batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_dataset()

    def init_dataset(self):
        num_worker = torch.get_num_threads() if torch.cuda.is_available() else 0

        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation((15, -15)),
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
        num_train = int(len(trainset.data) * self.train_percentage)
        num_val = min(len(trainset.data) - num_train, int(num_train * 0.1))
        #num_train = len(trainset.data) - num_val
        train_indexes = random_indexes[:num_train]
        trainset.data = trainset.data[train_indexes]
        trainset.targets = list(np.array(trainset.targets)[train_indexes])
        self.trainloader = DataLoaderX(trainset, batch_size=self.train_batch_size, shuffle=True, 
                                       pin_memory=True, num_workers=num_worker)

        validate_indexes = random_indexes[num_train:num_train + num_val]
        validateset.data = validateset.data[validate_indexes]
        validateset.targets = list(np.array(validateset.targets)[validate_indexes])
        self.validateloader = DataLoaderX(validateset, batch_size=len(validateset.data), pin_memory=True,
                                          num_workers=num_worker)

        # init test set
        testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform)
        self.testloader = DataLoaderX(testset, batch_size=500, shuffle=False, num_workers=num_worker)

    def train_model(self, model, learning_rate=1e-4, epochs=100, num_epoch_to_log=5, weight_decay=0, warmup_epochs=10, checkpoint=''):
        model = model.to(self.device)
        criterion = nn.CrossEntropyLoss()
        #criterion = LabelSmoothCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else \
                0.5 * (math.cos((epoch - warmup_epochs) / (epochs - warmup_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0,
        #                                                       last_epoch=-1)
        train_runner = TrainRunner(self.device)
        valid_runner = ValidRunner(self.device)
        self.best_model_file = checkpoint + '/best_model.pth'
        self.best_model_caribrated_file = checkpoint + '/best_model_caribrated.pth'
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        best_accuracy = 0.0
        for epoch in range(epochs):
            total_loss, all_logits, all_predicts, all_labels = train_runner.run(model, self.trainloader, criterion, optimizer)
            scheduler.step()
            if epoch % num_epoch_to_log == 0:
                train_runner.show_metrics(total_loss, all_logits, all_predicts, all_labels, epoch)
                total_loss, all_logits, all_predicts, all_labels = valid_runner.run(model, self.validateloader)
                accuracy = valid_runner.show_metrics(total_loss, all_logits, all_predicts, all_labels, epoch)
                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), self.best_model_file)

    def caribrate(self, model):
        valid_runner = ValidRunner(self.device)
        state_dict = torch.load(self.best_model_file)
        model.load_state_dict(state_dict)
        new_model = valid_runner.caribrate(model, self.validateloader)
        torch.save(new_model.state_dict(), self.best_model_caribrated_file)
        return new_model


    def test(self, model):
        state_dict = torch.load(self.best_model_file)
        model.load_state_dict(state_dict)
        self._test(model)

    def test_caribrate(self, model):
        state_dict = torch.load(self.best_model_caribrated_file)
        model.load_state_dict(state_dict)
        self._test(model)

    def _test(self, model):
        test_runner = TestRunner(self.device)
        total_loss, all_logits, all_predicts, all_labels = test_runner.run(model, self.testloader)
        test_runner.show_metrics(total_loss, all_logits, all_predicts, all_labels, epoch=0)



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


class Stage(Enum):
    TRAIN = 0,
    VALID = 1,
    TEST = 2

class AbstractRunner(ABC):
    def __init__(self, device):
        self.device = device
        self.stage = None

    def run(self, model, dataloader, criterion=None, optimizer=None):
        num_gpu = torch.cuda.device_count()
        if num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
        self.set_mode(model)
        all_labels = []
        all_predicts = []
        all_logits = []
        total_loss = 0.0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            with torch.torch.set_grad_enabled(self.is_train()):
                logits = model(inputs)
                _, predicts = torch.max(logits, dim=1)
                loss = self.backward(logits, labels, criterion, optimizer)
                if loss:
                    total_loss += loss
            all_labels.append(labels)
            all_predicts.append(predicts)
            all_logits.append(logits)
        all_logits = torch.cat(all_logits)
        all_predicts = torch.cat(all_predicts)
        all_labels = torch.cat(all_labels)
        #print(all_labels.shape, all_predicts.shape, all_logits.shape)
        total_loss /= len(all_predicts)
        return total_loss, all_logits, all_predicts, all_labels

    def show_metrics(self, total_loss, all_logits, all_predicts, all_labels, epoch=0):
        accuracy = (all_predicts == all_labels).sum().float() / len(all_predicts)
        TP = ((all_labels == 1) & (all_predicts == 1)).sum().float()
        FP = ((all_labels == 0) & (all_predicts == 1)).sum().float()
        TN = ((all_labels == 0) & (all_predicts == 0)).sum().float()
        FN = ((all_labels == 1) & (all_predicts == 0)).sum().float()
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        specifiy = TN / (TN + FP) if (TN + FP) > 0 else 0
        F1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        ece = self.cal_ece(all_logits, all_labels)
        print("Epoch: %d, Stage: %s, loss: %.6f, accuracy: %.6f, precision: %.6f, sensitivity/recall: %.6f, "\
                "specifity: %.6f, F1: %.6f, ECE: %.9f"
            % (epoch, self.stage.name, total_loss, accuracy, precision, recall, specifiy, F1, ece
        ))
        return accuracy

    def cal_ece(self, logits, labels):
        n_bins = 15
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        confidences, predicts = torch.max(logits, dim=1)
        accuracies = (predicts == labels)
        ece = torch.tensor(0.0)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
        return ece

    @abstractmethod
    def set_mode(self, model):
        pass

    def is_train(self):
        return False

    def backward(self, outputs, labels, criterion, optimizer):
        pass


class TrainRunner(AbstractRunner):
    def __init__(self, device):
        super(TrainRunner, self).__init__(device)
        self.stage = Stage.TRAIN

    def set_mode(self, model):
        model.train()

    def is_train(self):
        return True

    def backward(self, outputs, labels, criterion, optimizer):
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss


class ValidRunner(AbstractRunner):
    def __init__(self, device):
        super(ValidRunner, self).__init__(device)
        self.stage = Stage.VALID

    def set_mode(self, model):
        model.eval()

    def caribrate(self, model, validateloader):
        total_loss, all_logits, all_predicts, all_labels = self.run(model, validateloader)
        ece_before = self.cal_ece(all_logits, all_labels)
        print("Before scaling, ECE: ", ece_before.item())
        self.show_metrics(total_loss, all_logits, all_predicts, all_labels, epoch=0)

        new_model = TemperatureScalingModel(model, self.device)
        new_model.caribrate(all_logits, all_labels)
        total_loss, all_logits, all_predicts, all_labels = self.run(new_model, validateloader)
        ece_after = self.cal_ece(all_logits, all_labels)
        print("After scaling, ECE: ", ece_after.item())
        self.show_metrics(total_loss, all_logits, all_predicts, all_labels, epoch=0)

        return new_model



class TestRunner(ValidRunner):
    def __init__(self, device):
        super(TestRunner, self).__init__(device)
        self.stage = Stage.TEST
