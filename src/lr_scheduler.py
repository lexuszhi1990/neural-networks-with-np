# -*- coding: utf-8 -*-

import numpy as np


class MultiStepLR(object):
    def __init__(self, optimizer, milestones, gamma):
        super(MultiStepLR, self).__init__()
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma

        self.cur_epoch = 0

    def step(self):
        self.cur_epoch += 1

        count = 0
        for i in self.milestones:
            if self.cur_epoch > i:
                count += 1

        self.optimizer.lr = self.optimizer.base_lr * (self.gamma ** count)
