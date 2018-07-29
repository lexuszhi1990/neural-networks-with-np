# -*- coding: utf-8 -*-

import logging
import numpy as np
from src.symbol import get_symbol
from src.logger import setup_logger
from src.configuration import cfg_list
from src.args import get_args
from src.data_loader import mnist
from src.utils import check_dir_exists, restore_weights, img_preprocess

def val(model, symbol_name, params_path, dataset):
    if model is None:
        model = get_symbol(symbol_name)()
        restore_weights(model, params_path)

    true_nums = 0
    for index, (inputs, label) in enumerate(dataset):
        inputs = img_preprocess(inputs)
        outputs = model.forward(inputs)
        results = outputs.argmax(axis=1)

        true_num = np.sum(label == results)
        true_nums += true_num
        logging.info("epoch [%d] precision: %.5f" % (index, true_num/len(label)))

    logging.info("[%s] precision: %.5f" % (dataset.name, true_nums/dataset.total))


if __name__ == '__main__':

    args = get_args()
    cfg = cfg_list[args.config_id]

    check_dir_exists(cfg['workspace'])
    setup_logger("%s/val" % cfg['workspace'])

    val_dataset = mnist('train', cfg['batch_size'])
    val(None, cfg['symbol'], args.ckpt_path, val_dataset)
