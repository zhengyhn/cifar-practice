import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import os
import glob
from PIL import Image
import time
from torch.multiprocessing import Process, Queue, Pool
import torch.multiprocessing as mp
import traceback

label_map = {
    'EBV' : 1,
    'ELSE' : 0
}

class MergedDataset(torch.utils.data.Dataset):
    def __init__(self, path, magnify, transforms):
        super(torch.utils.data.Dataset, self).__init__()
        self._num_worker = torch.get_num_threads()
        #self.queue = Queue(2 * self._num_worker)
        self.path = path
        self.magnify = magnify
        self.transforms = transforms
        self.data_path = os.path.join(path, 'data')
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        if os.path.exists(self._get_label_file()):
            self.labels = torch.load(self._get_label_file())
        self.device = torch.device('cpu')
        self.extractor = models.resnet18(pretrained=True).to(self.device)
        self.extractor.fc = nn.Sequential()
        self.extractor.eval()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        z = torch.load(self.get_feature_file(item))
        z = Variable(z, requires_grad=False)
        z = z.resize_(z.shape[0], 32, 32)
        return z.data, self.labels[item]

    def get_feature_file(self, item):
        return os.path.join(self.data_path, 'feature_' + str(item) + '.pt')

    def _get_label_file(self):
        return os.path.join(self.data_path, 'labels.pt')

    def _listdir(self, path):
        return [filename for filename in os.listdir(path) if os.path.isdir(os.path.join(path, filename))]

    def save(self):
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            pass
        pool = Pool(processes=4)
        self.labels = []
        item = 0
        for label_name in label_map.keys():
            images = self._listdir(os.path.join(self.path, label_name))
            for i in range(len(images)):
                rows, cols = [], []
                files = glob.glob(os.path.join(self.path, label_name, images[i], str(self.magnify), '*.jpeg'))
                for file in files:
                    filename = os.path.basename(file)
                    nums = filename.split('_')
                    row, col = int(nums[0]), int(nums[1])
                    rows.append(row)
                    cols.append(col)
                num_row = max(rows) - min(rows) + 1
                num_col = max(cols) - min(cols) + 1
                patches = np.chararray((num_row, num_col), itemsize=1024)
                for file in files:
                    filename = os.path.basename(file)
                    nums = filename.split('_')
                    row, col = int(nums[0]), int(nums[1])
                    patches[row - min(rows), col - min(cols)] = file
                self.labels.append(label_map[label_name])
                # Save feature vector
                pool.apply_async(self.doit, args=(item, patches, num_row, num_col), error_callback=self.print_error)
                item += 1
        # Save labels
        torch.save(self.labels, self._get_label_file())

        pool.close()
        pool.join()
        print('done')

    def print_error(self, error):
        print(error)

    def doit(self, item, patches, rows, cols):
        imgs = np.zeros((rows, cols, 3, 256, 256))
        filename = self.get_feature_file(item)
        if os.path.exists(filename):
            return
        try:
            for row in range(rows):
                for col in range(cols):
                    img = Image.open(patches[row, col])
                    if self.transforms:
                        img = self.transforms(img)
                    imgs[row, col] = img
            row, col, channel, height, weight = imgs.shape
            imgs = imgs.reshape(row * col, channel, height, weight).astype(np.float32)
            imgs = torch.from_numpy(imgs)
            z = self.extractor(imgs.to(self.device))
            z = z.reshape(rows, cols, -1)
            z = z.permute(2, 0, 1)
            #print(z.shape)
            torch.save(z.cpu(), filename)
        except Exception as error:
            traceback.print_exc()
            print('error:', error)
        print('exit', item)

class StomachDataset():
    def __init__(self):
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
        return 2

    def get(self):
        base_path = '../yuanhang256'
        trainset = MergedDataset(os.path.join(base_path, ''), transforms=self.transform, magnify=10.0)
        testset = MergedDataset(os.path.join(base_path, ''), transforms=self.transform, magnify=10.0)
        testset.labels = testset.labels[:5]
        validset = testset
        return trainset, validset, testset

if __name__ == '__main__':
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    base_path = '../yuanhang256'
    trainset = MergedDataset(os.path.join(base_path, ''), transforms=transform, magnify=10.0)
    trainset.save()
    # testset = MergedDataset(os.path.join(base_path, 'test'), self.transform, self.rows, self.cols)
    # validset = testset
    #     return trainset, validset, testset
