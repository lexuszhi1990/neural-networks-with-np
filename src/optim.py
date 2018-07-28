# -*- coding: utf-8 -*-

import numpy as np


class SGD(object):
    def __init__(self, model, cfg):

        self.model = model
        self.base_lr = cfg['base_lr']
        self.momentum = cfg['momentum']

        # initialize learning rate
        self.lr = self.base_lr

    def step(self, grads):
        for key, value in self.model.params.items():
            self.model.params[key] = value - self.lr * grads[key]
