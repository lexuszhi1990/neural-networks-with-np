# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import numpy as np
from PIL import Image
from src.symbol import get_symbol
from src.args import get_args

from src.utils import restore_weights, img_preprocess

def setup_data(img):
    inputs = np.zeros((1, 1, 28, 28)).astype(np.uint8)

    for i in range(img.size[0]):
        for j in range(img.size[1]):
            inputs[0, 0, j, i] = img.getpixel((i, j))

    return inputs

def eval(model, inputs):
    inputs = img_preprocess(inputs)
    outputs = model.forward(inputs)

    return outputs.argmax(axis=1)[0]

def demo(symbol_name, params_path, img_dir):

    model = get_symbol(symbol_name)()
    restore_weights(model, params_path)

    for img_path in Path(img_dir).glob('*.png'):
        img = Image.open(img_path)
        inputs = setup_data(img)
        num = eval(model, inputs)
        print("the number in image %s is %d" % (img_path, num))


if __name__ == '__main__':
    args = get_args()

    demo(args.symbol_name, args.ckpt_path, args.test_dir)