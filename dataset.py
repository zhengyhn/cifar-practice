import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
from abc import abstractmethod, ABC
import copy
import numpy as np
import random
import os
import glob
import cv2
from PIL import Image
import time
from torch.multiprocessing import Queue, Process, Pool
import torch.multiprocessing as mp
from tqdm import tqdm
from enum import Enum
from typing import Tuple

class AbstractDataset(ABC):
    def __init__(self):
        self._init_seed()
        pass

    def _init_seed(self, seed = 0):
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def num_labels(self) -> int:
        pass

    @abstractmethod
    def num_dims(self) -> Tuple[int, int, int]:
        pass


class DatasetType(Enum):
    CIFAR10 = 0
    MNIST = 1


class DatasetFactory:
    @staticmethod
    def get_by_name(name, train_percentage):
        name = name.lower()
        if name == DatasetType.CIFAR10.name.lower():
            return CIFAR10Dataset(train_percentage)
        elif name == DatasetType.MNIST.name.lower():
            return MNISTDataset(train_percentage)

class CIFAR10Dataset(AbstractDataset):
    def __init__(self, train_percentage):
        super(AbstractDataset, self).__init__()
        self.train_percentage = train_percentage

    def num_labels(self):
        return 10

    def num_dims(self):
        return 32, 32, 3

    def get(self):
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(),
            transforms.RandomCrop(32, padding=4),
            transforms.RandomRotation((15, -15)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform)
        # Build validset
        validset = copy.deepcopy(trainset)
        random_indexes = np.random.permutation(range(0, len(trainset.data)))
        num_train = int(len(trainset.data) * self.train_percentage)
        num_val = min(len(trainset.data) - num_train, int(num_train * 0.1))
        train_indexes = random_indexes[:num_train]
        trainset.data = trainset.data[train_indexes]
        trainset.targets = list(np.array(trainset.targets)[train_indexes])

        if num_val > 0:
            valid_indexes = random_indexes[num_train:num_train + num_val]
            validset.data = validset.data[valid_indexes]
            validset.targets = list(np.array(validset.targets)[valid_indexes])
        else:
            validset = testset
        return trainset, validset, testset


class MNISTDataset(AbstractDataset):
    def __init__(self, train_percentage):
        super(AbstractDataset, self).__init__()
        self.train_percentage = train_percentage

    def num_labels(self):
        return 10

    def num_dims(self):
        return 28, 28, 1

    def get(self):
        train_transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(),
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomRotation((15, -15)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        trainset = torchvision.datasets.MNIST(root='./dataset', train=True, download=True, transform=train_transform)
        testset = torchvision.datasets.MNIST(root='./dataset', train=False, download=True, transform=transform)
        # Build validset
        validset = copy.deepcopy(trainset)
        random_indexes = np.random.permutation(range(0, len(trainset.data)))
        num_train = int(len(trainset.data) * self.train_percentage)
        num_val = min(len(trainset.data) - num_train, int(num_train * 0.1))
        train_indexes = random_indexes[:num_train]
        trainset.data = trainset.data[train_indexes]
        trainset.targets = list(np.array(trainset.targets)[train_indexes])

        if num_val > 0:
            valid_indexes = random_indexes[num_train:num_train + num_val]
            validset.data = validset.data[valid_indexes]
            validset.targets = list(np.array(validset.targets)[valid_indexes])
        else:
            validset = testset
        return trainset, validset, testset


class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, path, transforms, rows, cols):
        super(torch.utils.data.Dataset, self).__init__()
        self.path = path
        self.transforms = transforms
        self.rows = rows
        self.cols = cols
        self.labels = []
        self.label_path = os.path.join(self.path, 'labels.pt')
        if os.path.exists(self.label_path):
            self.labels = torch.load(self.label_path)
            print('labels load, ', len(self.labels))
        self.images = [filename for filename in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, filename))]
        #self.device = torch.device('cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.extractor = models.resnet18(pretrained=True).to(self.device)
        self.extractor.fc = nn.Sequential()
        self.extractor.eval()


    def __len__(self):
        return len(self.labels)
        #return 10

    def __getitem__(self, item):
        #z = torch.load(os.path.join(self.path, str(item), 'feature.pt'))
        #z = Variable(z, requires_grad = False)
        #return z.data, self.labels[item]
        #self.patches= np.chararray((len(self.images), self.rows, self.cols), itemsize=1024)
        patches = glob.glob(os.path.join(self.path, self.images[item], '*.jpeg'))
        imgs = np.zeros((self.rows, self.cols, 3, 256 // self.rows, 256 // self.cols))
        try:
            for patch in patches:
                filename = os.path.basename(patch)
                index = filename.find('_')
                row, col = int(filename[index - 1]), int(filename[index + 1])
                img = Image.open(patch)
                if self.transforms:
                    img = self.transforms(img)
                imgs[row, col] = img
            row, col, channel, height, weight = imgs.shape
            imgs = imgs.reshape(row * col, channel, height, weight).astype(np.float32)
            imgs = torch.from_numpy(imgs)
            z = self.extractor(imgs.to(self.device))
            z = z.reshape(self.rows, self.rows, -1)
            z = z.permute(2, 0, 1)
            z = Variable(z, requires_grad = False)
        except Exception as error:
            print(error)
        return z.data.cpu(), self.labels[item]

    def save(self):
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.labels = []
        images = [filename for filename in os.listdir(self.path) if os.path.isdir(os.path.join(self.path, filename))]
        #images = images[:100]
        self.patches = np.chararray((len(images), self.rows, self.cols), itemsize=1024)
        for i in range(len(images)):
            patches = glob.glob(os.path.join(self.path, images[i], '*.jpeg'))
            for patch in patches:
                filename = os.path.basename(patch)
                index = filename.find('_')
                row, col = int(filename[index - 1]), int(filename[index + 1])
                self.patches[i, row, col] = patch
            patch = patches[0]
            self.labels.append(int(patch[patch.rfind('_') + 1]))
        # Save labels
        torch.save(self.labels, os.path.join(self.path, 'labels.pt'))
        # Save feature vector
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        num_worker = 8
        size = (len(images) // num_worker) + 1
        workers = []
        for i in range(num_worker):
            worker = SaveWorker(self.path, self.patches, self.rows, self.cols, self.transforms, i * size, (i + 1) * size)
            worker.start()
            workers.append(worker)
        for worker in workers:
            worker.join()

class SaveWorker(Process):
    def __init__(self, path, patches, rows, cols, transforms, begin, end):
        Process.__init__(self, name='SplitWorker')
        self.path = path
        self.patches = patches
        self.rows = rows
        self.cols = cols
        self.transforms = transforms
        self.begin = begin
        self.end = end
        #self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = torch.device('cpu')
        self.extractor = models.resnet18(pretrained=True).to(self.device)
        self.extractor.fc = nn.Sequential()
        self.extractor.eval()

    def run(self):
        end = min(self.end, len(self.patches))
        pbar = tqdm(total = end - self.begin)
        pbar.set_description("{} - {}".format(self.begin, end))
        for item in range(self.begin, end):
            pbar.update()
            patches = self.patches[item]
            imgs = np.zeros((self.rows, self.cols, 3, 32, 32))
            filename = os.path.join(self.path, str(item), 'feature.pt')
            #if os.path.exists(filename):
            #    continue
            for row in range(self.rows):
                for col in range(self.cols):
                    img = Image.open(patches[row, col])
                    if self.transforms:
                        img = self.transforms(img)
                    imgs[row, col] = img
            row, col, channel, height, weight = imgs.shape
            imgs = imgs.reshape(row * col, channel, height, weight).astype(np.float32)
            try:
                imgs = torch.from_numpy(imgs)
            except Exception as error:
                print(error)
            z = self.extractor(imgs.to(self.device))
            z = z.reshape(self.rows, self.rows, -1)
            z = z.permute(2, 0, 1)
            torch.save(z.cpu(), os.path.join(self.path, str(item), 'feature.pt'))


class CIFAR10SplitDataset(AbstractDataset):
    def __init__(self, rows, cols):
        super(AbstractDataset, self).__init__()
        self.rows = rows
        self.cols = cols
        self.train_transform = transforms.Compose([
            #transforms.RandomHorizontalFlip(),
            #transforms.ColorJitter(),
            #transforms.RandomCrop(32, padding=4),
            #transforms.RandomRotation((15, -15)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def num_label(self):
        return 10

    def get(self):
        base_path = './split4x4'
        trainset = MergedDataset(os.path.join(base_path, 'train'), self.train_transform, self.rows, self.cols)
        testset = MergedDataset(os.path.join(base_path, 'test'), self.transform, self.rows, self.cols)
        validset = testset
        return trainset, validset, testset
