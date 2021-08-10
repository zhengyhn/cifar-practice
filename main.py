import torch
import torch.distributed as dist
import argparse
from vgg import Vgg
from solver import Solver
from dataset import DatasetFactory
from models import ModelFactory
import os
import fire

def main(dataset_name, model_name, epochs=50, batch_size=128):
    dataset = DatasetFactory.get_by_name(dataset_name, train_percentage=0.95)
    checkpoint = 'checkpoint/{}_{}'.format(dataset_name, model_name)
    solver = Solver(dataset, checkpoint, train_batch_size=batch_size)
    model = ModelFactory.get_by_name(model_name, dataset)
    solver.train_model(model, epochs=epochs, warmup_epochs=10, num_epoch_to_log=5, learning_rate=1e-3, weight_decay=1e-4)
    solver.test(model)


if __name__ == '__main__':
    fire.Fire(main)
