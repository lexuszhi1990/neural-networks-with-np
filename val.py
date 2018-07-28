# -*- coding: utf-8 -*-

import logging
import numpy as np
from src.data_loader import mnist
from src.utils import check_dir_exists, load_weights
from src.logger import setup_logger
from src.configuration import cfg

from src.symbol.mlp import mlp

def val(model, model_name, params_path, dataset):
    if model is None:
        model = mlp()
        restore_weights(model, params_path)

    err_num = 0
    total_num = 0
    for index, (inputs, label) in enumerate(dataset):
        outputs = model.forward(inputs)
        results = outputs.argmax(axis=1)
        err_num += np.sum(label-results)
        total_num += len(label)

    logging.info("for %s, precision: %.5f" % (dataset.name(), 1-err_num/total_num))


if __name__ == '__main__':

    check_dir_exists(cfg['workspace'])
    setup_logger("%s/val" % cfg['workspace'])

    val_dataset = mnist('train', cfg['batch_size'], data_path=cfg['data_path'])
    val(None, 'mlp', 'ckpt/mlp-v1/mlp-99.json', val_dataset)
