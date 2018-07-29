# -*- coding: utf-8 -*-

import numpy as np

# activation operation

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


# activation layers

def relu_forward(inputs):
    return relu(inputs), inputs

def relu_backward(d_out, params):
    # outputs = np.maximum(0, inputs)
    # outputs[outputs > 0] = 1
    inputs = params
    return d_out * d_relu(inputs)


def sigmoid_forward(inputs):
    return sigmoid(inputs), inputs

def sigmoid_backword(d_out, params):
    inputs = params
    return d_out * d_sigmoid(inputs)
