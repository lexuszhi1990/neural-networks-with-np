# -*- coding: utf-8 -*-

import numpy as np
# np.random.seed(1234)

N, D_in = 64, 32
x = np.random.randn(N, D_in)
y = np.random.rand(N)

w1 = np.random.randn(D_in)
b1 = np.random.randn(N)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    return sigmoid(x)*(1-sigmoid(x))

lr = 1e-1
loss = float("inf")
while loss >= 1e-5:
    h1 = np.dot(x, w1) + b1
    y_pred = sigmoid(h1)

    loss = np.square(y_pred - y).sum()
    print(loss)

    grad_y_pred = (y_pred - y) * 2
    grad_h1 = grad_y_pred * d_sigmoid(h1)
    grad_w1 = np.dot(x.T, grad_h1)

    w1 -= lr * grad_w1
    b1 -= lr * grad_h1


print("w1: {}\n, b1: {}".format(w1, b1))
print("y: {};\ny_pred: {}".format(y, y_pred))


import pdb; pdb.set_trace()
