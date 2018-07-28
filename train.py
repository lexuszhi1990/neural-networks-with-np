# -*- coding: utf-8 -*-

from src.configuration import cfg
from src.data_loader import mnist
from src.resnet import resnet
from src.mlp import mlp
import src.optim as optim


def train():

    dataset = mnist(cfg=cfg)
    model = mlp()

    for epoch in range(cfg['max_epoch']):
        for index, (inputs, label) in enumerate(dataset):
            # inputs = inputs/2. - 127.5
            outputs = model.forward(inputs)
            loss = model.compute_loss(label)
            grads = model.backward()

if __name__ == '__main__':
    train()
