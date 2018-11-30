# -*- coding: utf-8 -*-

import numpy as np

def softmax(z):
    z -= np.max(z)
    sm = (np.exp(z).T / np.sum(np.exp(z), axis=0)).T
    return sm

def softmax_grad(s):
    # Take the derivative of softmax element w.r.t the each logit which is usually Wi * X
    # input s is softmax value of the original input x.
    # s.shape = (1, n)
    # i.e. s = np.array([0.3, 0.7]), x = np.array([0, 1])
    # initialize the 2-D jacobian matrix.
    jacobian_m = np.diag(s)
    for i in range(len(jacobian_m)):
        for j in range(len(jacobian_m)):
            if i == j:
                jacobian_m[i][j] = s[i] * (1-s[i])
            else:
                jacobian_m[i][j] = -s[i]*s[j]
    return jacobian_m


N, D_in, D_h1, D_out = 64, 128, 256, 10

x = np.random.randn(N, D_in)
y = [np.random.randint(D_out) for i in range(N)]
y_onehot = np.array([np.eye(D_out)[i] for i in y]).astype(int)

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

    # loss = -np.log(y_pred[np.arange(N), y]).sum() / N
    # y_pred[y_pred==1] = 1 - 1e-16
    # y_pred[y_pred==0] = 1e-16
    # loss = y_onehot * np.log(y_pred) + (1 - y_onehot) * np.log(1 - y_pred)
    loss = y_onehot * np.log(y_pred)
    loss[loss == np.inf] = 0
    loss[loss == -np.inf] = 0
    loss = -np.sum(np.nan_to_num(loss)) / N

    print(loss)

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

