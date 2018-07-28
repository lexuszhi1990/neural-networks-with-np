# -*- coding: utf-8 -*-

import logging
import src.optim as optim
from val import val
from src.symbol import get_symbol
from src.args import get_args
from src.logger import setup_logger
from src.data_loader import mnist
from src.configuration import cfg_list
from src.utils import check_dir_exists, save_weights, load_weights

def train(model, optimizer, dataset, cfg, val_dataset=None):

    for epoch in range(cfg['max_epoch']):
        for index, (inputs, label) in enumerate(dataset):
            outputs = model.forward(inputs)
            loss, reg_loss = model.compute_loss(label)
            grads = model.backward()
            optimizer.step(grads)

            logging.info("[%d/%d] train loss: %.4f, reg loss: %.4f, total: %.2f" %(epoch, index, loss, reg_loss, (loss + reg_loss)))

        params_path = save_weights(model.params, cfg['workspace'], model.name, epoch)
        logging.info("save model at: %s" % (params_path))
        if val_dataset is not None:
            val(model, model.name, params_path, val_dataset)

if __name__ == '__main__':

    args = get_args()
    cfg = cfg_list[args.config_id]

    check_dir_exists(cfg['workspace'])
    setup_logger("%s/training" % cfg['workspace'])

    train_dataset = mnist('train', cfg['batch_size'])
    val_dataset = mnist('test', cfg['batch_size'])
    model = get_symbol(cfg['symbol'])()
    optimizer = optim.SGD(model, cfg)
    train(model, optimizer, train_dataset, cfg, val_dataset)
