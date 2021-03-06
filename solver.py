import os
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
#import copy
import math
from abc import abstractmethod, ABC
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator
from enum import Enum
from temperature_scaling import TemperatureScalingModel
from tqdm import tqdm
import time


class Solver():
    def __init__(self, dataset, checkpoint, train_batch_size=400):
        self.dataset = dataset
        if not os.path.exists(checkpoint):
            os.makedirs(checkpoint)
        self.train_batch_size = train_batch_size
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.best_model_file = checkpoint + '/best_model.pth'
        self.best_model_caribrated_file = checkpoint + '/best_model_caribrated.pth'
        #self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.criterion = LabelSmoothCELoss()
        self.init_dataset()

    def init_dataset(self):
        num_worker = torch.get_num_threads() if torch.cuda.is_available() else 0
        print('num_worker: ', num_worker)

        trainset, validset, testset = self.dataset.get()
        self.trainloader = DataLoaderX(trainset, batch_size=self.train_batch_size,
                                   pin_memory=True, num_workers=num_worker, shuffle=True)
        self.validateloader = DataLoaderX(validset, batch_size=len(validset), pin_memory=True,
                                      num_workers=num_worker)
        self.testloader = DataLoaderX(testset, batch_size=512, shuffle=False, pin_memory=True, num_workers=num_worker)

    def train_model(self, model, learning_rate=1e-4, epochs=100, num_epoch_to_log=5, weight_decay=0, warmup_epochs=10):
        model = model.to(self.device)
        num_gpu = torch.cuda.device_count()
        if num_gpu > 0:
            cudnn.benchmark = True
        if num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        #optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=learning_rate, weight_decay=weight_decay)
        warm_up_with_cosine_lr = lambda epoch: (epoch + 1) / warmup_epochs if epoch < warmup_epochs else \
                0.5 * (math.cos((epoch - warmup_epochs) / (epochs - warmup_epochs) * math.pi) + 1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
        #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0,
        #                                                       last_epoch=-1)
        train_runner = TrainRunner(self.device)
        valid_runner = ValidRunner(self.device)
        best_accuracy = 0.0
        start = time.time()
        for epoch in range(epochs):
            total_loss, all_logits, all_predicts, all_labels = train_runner.run(model, self.trainloader, self.criterion, optimizer)
            scheduler.step()
            if epoch % num_epoch_to_log == (num_epoch_to_log - 1):
                train_runner.show_metrics(total_loss, all_logits, all_predicts, all_labels, epoch)
                total_loss, all_logits, all_predicts, all_labels = valid_runner.run(model, self.validateloader, self.criterion)
                accuracy = valid_runner.show_metrics(total_loss, all_logits, all_predicts, all_labels, epoch)
                if accuracy > best_accuracy:
                    torch.save(model.state_dict(), self.best_model_file)
                end = time.time()
                print('cost {} s'.format(end - start))
                start = time.time()

    def caribrate(self, model):
        valid_runner = ValidRunner(self.device)
        state_dict = torch.load(self.best_model_file)
        model.load_state_dict(state_dict)
        new_model = valid_runner.caribrate(model, self.validateloader, self.criterion)
        torch.save(new_model.state_dict(), self.best_model_caribrated_file)
        return new_model


    def test(self, model):
        state_dict = torch.load(self.best_model_file)
        self._test(model, state_dict)

    def test_caribrate(self, model):
        state_dict = torch.load(self.best_model_caribrated_file)
        self._test(model, state_dict)

    def _test(self, model, state_dict):
        test_runner = TestRunner(self.device)
        num_gpu = torch.cuda.device_count()
        if num_gpu > 0:
            cudnn.benchmark = True
        if num_gpu > 1:
            model = torch.nn.DataParallel(model, device_ids=range(num_gpu))
        model.load_state_dict(state_dict)
        total_loss, all_logits, all_predicts, all_labels = test_runner.run(model, self.testloader, self.criterion)
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

    def run(self, model, dataloader, criterion, optimizer=None):
        self.set_mode(model)
        all_labels = []
        all_predicts = []
        all_logits = []
        total_loss = 0.0
        for inputs, labels in tqdm(dataloader):
            inputs, labels = Variable(inputs).to(self.device, non_blocking=True), Variable(labels).to(self.device, non_blocking=True)
            loss, logits, predicts, labels = self.evaluate(model, inputs, labels, criterion, optimizer)
            total_loss += loss
            all_labels.append(labels)
            all_predicts.append(predicts)
            all_logits.append(logits)
        all_logits = torch.cat(all_logits)
        all_predicts = torch.cat(all_predicts)
        all_labels = torch.cat(all_labels)
        #print(all_labels.shape, all_predicts.shape, all_logits.shape)
        #total_loss /= len(all_predicts)
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
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        softmaxes = F.softmax(logits, dim=1)
        confidences, predicts = torch.max(softmaxes, dim=1)
        accuracies = predicts.eq(labels)
        ece = torch.zeros(1, device=logits.device)
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

    @abstractmethod
    def evaluate(self, model, inputs, labels, criterion, optimizer, alpha=0.1):
        pass


class TrainRunner(AbstractRunner):
    def __init__(self, device):
        super(TrainRunner, self).__init__(device)
        self.stage = Stage.TRAIN

    def set_mode(self, model):
        model.train()

    def evaluate(self, model, inputs, labels, criterion, optimizer, alpha = 0.8):
        #logits = model(inputs)
        #loss = criterion(logits, labels)
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #_, predicts = torch.max(logits, dim=1)
        lam = np.random.beta(alpha, alpha)
        index = np.random.permutation(range(len(labels)))
        inputs = lam * inputs + (1 - lam) * inputs[index, :]
        labels_a, labels_b = labels, labels[index]
        logits = model(inputs)
        loss = lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)
        loss.to(self.device)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, predicts = torch.max(logits, dim=1)
        if lam < 0.5:
            labels = labels_b
        return loss, logits, predicts, labels


class ValidRunner(AbstractRunner):
    def __init__(self, device):
        super(ValidRunner, self).__init__(device)
        self.stage = Stage.VALID

    def set_mode(self, model):
        model.eval()

    def evaluate(self, model, inputs, labels, criterion, optimizer, alpha=0.1):
        with torch.torch.set_grad_enabled(False):
            logits = model(inputs)
            _, predicts = torch.max(logits, dim=1)
            loss = criterion(logits, labels)
        return loss, logits, predicts, labels

    def caribrate(self, model, validateloader, criterion):
        total_loss, all_logits, all_predicts, all_labels = self.run(model, validateloader, criterion)
        ece_before = self.cal_ece(all_logits, all_labels)
        print("Before scaling, ECE: ", ece_before.item())
        self.show_metrics(total_loss, all_logits, all_predicts, all_labels, epoch=0)

        new_model = TemperatureScalingModel(model, self.device).to(self.device)
        new_model.caribrate(all_logits, all_labels)
        total_loss, all_logits, all_predicts, all_labels = self.run(new_model, validateloader, criterion)
        ece_after = self.cal_ece(all_logits, all_labels)
        print("After scaling, ECE: ", ece_after.item())
        self.show_metrics(total_loss, all_logits, all_predicts, all_labels, epoch=0)

        return new_model



class TestRunner(ValidRunner):
    def __init__(self, device):
        super(TestRunner, self).__init__(device)
        self.stage = Stage.TEST
