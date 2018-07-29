# -*- coding: utf-8 -*-

import argparse

def get_args():
    parser = argparse.ArgumentParser(description='neural networks based py')

    # choose config for training
    parser.add_argument('--config_id', default='mlp', type=str, help='config ID')

    # choose ckpt_path for evaluating
    parser.add_argument('--ckpt_path', default='./data/trained_model/mlp-99.json', type=str, help='params path to load base model')

    # choose ckpt_path for evaluating
    parser.add_argument('--symbol_name', default='mlp', type=str, help='symbol name used to initialize mode, [mlp, alexnet]')


    args = parser.parse_args()
    return args
