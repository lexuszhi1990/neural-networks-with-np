# -*- coding: utf-8 -*-

import logging
from val import val
from src.lr_scheduler import MultiStepLR
import src.optim as optim
from src.symbol import get_symbol
from src.args import get_args
from src.logger import setup_logger
from src.data_loader import mnist
from src.configuration import cfg_list
from src.utils import check_dir_exists, save_weights, img_preprocess, cal_precision, draw_loss_graph

def train(model, optimizer, scheduler, dataset, cfg, val_dataset=None, vis=True):

    training_loss_list = []
    test_loss_list = []

    for epoch in range(cfg['max_epoch']):
        true_nums = 0
        for index, (inputs, label) in enumerate(dataset):
            inputs = img_preprocess(inputs)
            outputs = model.forward(inputs)
            loss, reg_loss = model.compute_loss(label)
            grads = model.backward()
            optimizer.step(grads)

            true_num, precision = cal_precision(outputs, label)
            true_nums += true_num
            logging.info("[%d/%d] train loss: %.2f, reg loss: %.2f, total loss: %.4f, precision %.4f || lr: %.6f" %(epoch, index, loss, reg_loss, (loss + reg_loss), precision, optimizer.lr))
        scheduler.step()
        params_path = save_weights(model.params, cfg['workspace'], model.name, epoch)
        logging.info("save model at: %s, training precision %.4f" % (params_path, true_nums/dataset.total))
        training_loss_list.append(true_nums/dataset.total)

        if val_dataset is not None:
            loss = val(model, model.name, params_path, val_dataset)
            test_loss_list.append(loss)

    if vis:
        draw_loss_graph(cfg['workspace'] + "/loss.png", training_loss_list, test_loss_list)

if __name__ == '__main__':

    args = get_args()
    cfg = cfg_list[args.config_id]

    check_dir_exists(cfg['workspace'])
    setup_logger("%s/training" % cfg['workspace'])

    train_dataset = mnist('train', cfg['batch_size'], name=cfg['dataset_name'])
    val_dataset = mnist('test', cfg['batch_size'], name=cfg['dataset_name'])
    model = get_symbol(cfg['symbol'])(reg=cfg['reg'])
    optimizer = optim.SGD(model, cfg)
    scheduler = MultiStepLR(optimizer, cfg['milestones'], cfg['gamma'])
    train(model, optimizer, scheduler, train_dataset, cfg, val_dataset)
