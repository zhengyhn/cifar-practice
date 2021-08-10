import torch
import torch.distributed as dist
import argparse
from vgg import Vgg
from solver import Solver
from dataset import CIFAR10Dataset
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import torch.multiprocessing as mp


def main(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dataset = CIFAR10Dataset(train_percentage=0.95)
    solver = Solver(dataset, train_batch_size=128, local_rank=rank)
    model = Vgg(dataset.num_label())
    solver.train_model(model, epochs=100, warmup_epochs=10, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=1e-4, checkpoint='checkpoint/vgg')
    solver.test(model)

if __name__ == '__main__':
    world_size = 2
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
