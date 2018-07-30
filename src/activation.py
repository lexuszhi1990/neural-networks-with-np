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
    return x * (x > 0) * 1.0

def d_relu(x):
    return (x > 0) * 1.0

def leaky_relu(x, alpha=0.2):
    return np.where(x > 0, x, alpha * x)

def d_leaky_relu(x, alpha=0.2):
    return np.where(x > 0, 1., alpha)

def arctan(x):
    return np.arctan(x)

def d_arctan(x):
    return 1 / (1 + x ** 2)


# activation layers

def relu_forward(inputs):
    return relu(inputs), inputs

def relu_backward(d_out, params):
    return d_out * d_relu(params)

def leaky_relu_forward(inputs):
    return leaky_relu(inputs), inputs

def leaky_relu_backward(d_out, params):
    return d_out * d_leaky_relu(params)

def sigmoid_forward(inputs):
    return sigmoid(inputs), inputs

def sigmoid_backword(d_out, params):
    return d_out * d_sigmoid(params)
