# -*- coding: utf-8 -*-

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def d_sigmoid(x):
    return ( 1 - sigmoid(x)) * sigmoid(x)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - np.tanh(x) ** 2

def relu(x):
    mask = (x > 0) * 1.0
    return mask * x

def d_relu(x):
    mask = (x > 0) * 1.0
    return mask

def arctan(x):
    return np.arctan(x)

def d_arctan(x):
    return 1 / (1 + x ** 2)

def softmax(x):
    shiftx = x - np.max(x)
    exp = np.exp(shiftx)
    return exp/exp.sum()
