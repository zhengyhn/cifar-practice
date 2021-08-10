import torch.nn as nn
from dataset import AbstractDataset
from logistic import LogisticRegression
from mlp import MLP
from vgg import Vgg
from enum import Enum

class ModelType(Enum):
    LOGISTIC = 0
    MLP = 1
    VGG = 2

class ModelFactory:
    @staticmethod
    def get_by_name(name: str, dataset: AbstractDataset) -> nn.Module:
        name = name.lower()
        if name == ModelType.LOGISTIC.name.lower():
            return LogisticRegression(dataset)
        elif name == ModelType.MLP.name.lower():
            return MLP(dataset)
        elif name == ModelType.VGG.name.lower():
            return Vgg(dataset)

