import os
from PIL import Image
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from multiprocessing import Queue, Process
import multiprocessing as mp
import queue

PATCH_SIZE = 64
SIZE = 256
DATA_SET_PATH = './split4x4'

class SplitWorker(Process):
    def __init__(self, dataset, save_path, begin, end):
        Process.__init__(self, name='SplitWorker')
        self.dataset = dataset
        self.save_path = save_path
        self.begin = begin
        self.end = end

    def run(self):
        end = min(self.end, len(self.dataset))
        pbar = tqdm(total=end - self.begin)
        pbar.set_description("{} - {}".format(self.begin, end))
        for i in range(self.begin, end):
            #print("{}/{}".format(i, self.end))
            img, label = self.dataset[i]
            folder = self.save_path + '/' + str(i)
            row = col = SIZE // PATCH_SIZE
            if not os.path.exists(folder):
                os.makedirs(folder)
            for j in range(row):
                for k in range(col):
                    box = (k * PATCH_SIZE, j * PATCH_SIZE, (k + 1) * PATCH_SIZE, (j + 1) * PATCH_SIZE)
                    sub = img.crop(box)
                    sub.save("{}/{}_{}_{}.jpeg".format(folder, j, k, label), 'JPEG')
            pbar.update()

def split(dataset, length, save_path):
    num_worker = 36 * 2
    size = (length // num_worker) + 1
    workers = []
    for i in range(num_worker):
        worker = SplitWorker(dataset, save_path, i * size, (i + 1) * size)
        worker.start()
        workers.append(worker)
    for worker in workers:
        worker.join()

def _split(img, label, folder):
    row = col = SIZE // PATCH_SIZE
    if not os.path.exists(folder):
        os.makedirs(folder)
    for j in range(row):
        for k in range(col):
            box = (k * PATCH_SIZE, j * PATCH_SIZE, (k + 1) * PATCH_SIZE, (j + 1) * PATCH_SIZE)
            sub = img.crop(box)
            sub.save("{}/{}_{}_{}.jpeg".format(folder, j, k, label), 'JPEG')

transform = transforms.Compose([
    transforms.Resize(SIZE)
])

trainset = torchvision.datasets.CIFAR10(root='./dataset', train=True, download=False, transform=transform)
print(len(trainset))
#split(trainset, 1, DATA_SET_PATH + '/train')
split(trainset, len(trainset), DATA_SET_PATH + '/train')
testset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=False, transform=transform)
split(testset, len(testset), DATA_SET_PATH + '/test')
#split(testset, 100, DATA_SET_PATH + '/test')

