# -*- coding: utf-8 -*-

import numpy as np


class SDG(object):
    def __init__(self, model, base_lr, momentum=None):

        self.model = model
        self.base_lr = base_lr
        self.momentum = momentum

        self.lr = base_lr

    def step(self, grads):
        aa = self.model.params['l1_weight']
        for key, value in self.model.params.items():
            self.model.params[key] = value - self.lr * grads[key]
        bb = self.model.params['l1_weight']
        print(np.sum(aa-bb))
