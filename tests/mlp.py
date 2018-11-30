# -*- coding: utf-8 -*-
# three-layer mlp

import numpy as np
# np.random.seed(1234)

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
N, D_in, H, D_out = 64, 1000, 100, 10

# Create random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# Randomly initialize weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)


lr = 1e-6

while True:
    h1 = np.dot(x, w1)
    h1_relu = np.maximum(h1, 0)
    y_pred = np.dot(h1_relu, w2)

    loss = np.square(y_pred - y).sum()
    print(loss)

    grad_y_pred = 2 * (y_pred - y)
    grad_w2 = np.dot(h1_relu.T, grad_y_pred)
    grad_h_relu = np.dot(grad_y_pred, w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[grad_h<0] = 0
    grad_w1 = np.dot(x.T, grad_h)

    w1 -= lr * grad_w1
    w2 -= lr * grad_w2

    if loss < 1e-6:
        break


print("y: {};\ny_pred: {}".format(y, y_pred))
print("mean loss: {}".format(np.mean(y - y_pred)))
import pdb; pdb.set_trace()
