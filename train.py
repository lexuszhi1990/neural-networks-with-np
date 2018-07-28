# -*- coding: utf-8 -*-

from src.configuration import cfg
from src.data_loader import mnist
from src.resnet import resnet
from src.mlp import mlp
import src.optim as optim
from src.logger import setup_logger
from src.utils import check_dir_exists, save_weights, load_weights
import logging

def train(dataset, model, optimizer, cfg):

    check_dir_exists(cfg['workspace'])
    setup_logger("%s/training" % cfg['workspace'])

    for epoch in range(cfg['max_epoch']):
        for index, (inputs, label) in enumerate(dataset):
            outputs = model.forward(inputs)
            loss, reg_loss = model.compute_loss(label)
            grads = model.backward()
            optimizer.step(grads)

            logging.info("[%d/%d] train loss: %.4f, reg loss: %.4f, total: %.2f" %(epoch, index, loss, reg_loss, (loss + reg_loss)))

        save_weights(model.params, cfg['workspace'], model.name, epoch)
        logging.info("save model %s-%d.json" % (model.name, epoch))

if __name__ == '__main__':
    dataset = mnist()
    model = mlp()
    optimizer = optim.SDG(model, cfg)
    train(dataset, model, optimizer, cfg)
