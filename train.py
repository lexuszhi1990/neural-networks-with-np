# -*- coding: utf-8 -*-

from src.configuration import cfg
from src.data_loader import mnist
from src.resnet import resnet
from src.mlp import mlp
import src.optim as optim


def train(dataset, model, optimizer, cfg):

    for epoch in range(cfg['max_epoch']):
        for index, (inputs, label) in enumerate(dataset):
            outputs = model.forward(inputs)
            loss, reg_loss = model.compute_loss(label)
            grads = model.backward()
            optimizer.step(grads)

            print("[%d/%d] training loss: %.4f, reg loss: %.4f, total: %.2f" %(epoch, index, loss, reg_loss, (loss + reg_loss)))

if __name__ == '__main__':
    dataset = mnist()
    model = mlp()
    optimizer = optim.SDG(model, cfg)
    train(dataset, model, optimizer, cfg)
