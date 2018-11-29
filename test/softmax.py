# -*- coding: utf-8 -*-

import numpy as np

N, D_in, D_h1, D_out = 64, 128, 256, 10

x = np.random.randn(N, D_in)
y = [np.random.randint(D_out) for i in range(N)]
# y = np.array([np.eye(D_out)[np.random.randint(D_out)] for i in range(N)]).astype(np.float16)

w1 = np.random.randn(D_in, D_h1)
b1 = np.random.randn(N)
w2 = np.random.randn(D_h1, D_out)
b1 = np.random.randn(N)

loss = float("INF")
lr = 1e-1
base = 1e-16
while loss > 1e-5:
    h1 = np.dot(x, w1)
    h1[h1 < 0] = 0
    out = np.dot(h1, w2)
    y_pred = np.exp(out)/(np.sum(np.exp(out), axis=1, keepdims=True))
    loss = -np.log(y_pred[np.arange(N), y]).sum() / N
    print(loss)

    # y_pred[y_pred==1] = 1 - 1e-16
    # y_pred[y_pred==0] = 1e-16
    # np.sum(y * np.log(y_pred) + (1 - y) * np.log(1 - y_pred))

    grad_out = y_pred.copy()
    grad_out[np.arange(N), y] -= 1
    grad_out /= N

    grad_w2 = np.dot(h1.T, grad_out)
    grad_h1 = np.dot(grad_out, w2.T)

    grad_h1[grad_h1<0] = 0
    grad_w1 = np.dot(x.T, grad_h1)

    w1 -= lr * grad_w1
    w2 -= lr * grad_w2

print("y: {};\ny_pred: {}".format(y, np.argmax(y_pred, axis=1)))
print("mean loss: {}".format(np.mean(y - np.argmax(y_pred, axis=1))))
import pdb; pdb.set_trace()

