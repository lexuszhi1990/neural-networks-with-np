import numpy as np

def softmax_loss(y_hat, y):
    probs = np.exp(y_hat - np.max(y_hat, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    batch_size = y_hat.shape[0]
    loss = -np.sum(np.log(probs[np.arange(batch_size), y])) / batch_size
    dx = probs.copy()
    dx[np.arange(batch_size), y] -= 1
    dx /= batch_size
    return loss, dx
