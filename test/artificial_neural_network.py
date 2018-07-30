import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))

def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

x = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])


y = np.array([
    [0],
    [1],
    [1],
    [1]
])


'''
    inputs: (N, M)
    labels: (N,)
'''

w = np.random.randn(3, 1)

for e in range(10000):
    output = np.dot(x, w)
    y_hat = sigmoid(output)
    loss = 0.5 * (y_hat - y) ** 2 / len(y_hat)

    d_loss = (y_hat - y)
    d_y_hat = d_sigmoid(output)

    dw = x.T.dot(d_loss * d_y_hat)
    w -= 0.8 * dw


output = np.dot(x, w)
y_hat = sigmoid(output)
print(np.abs(y_hat-y))
import pdb
pdb.set_trace()
