# -*- coding: utf-8 -*-

from src.configuration import cfg
from src.mnist import load_minist
from src.resnet import resnet
import src.optim as optim


def train():

    data = load_minist()
    train_samples = data['training_images']
    train_labels = data['training_labels']
    val_samples = data['test_images']
    val_lables = data['test_labels']

    model = resnet()
    # optimizer = optim.SDG(model, base_lr=1e-3)

    import numpy as np

    for i in range(100):

        samples_batch = (np.array(train_samples[i*cfg['batch_size']:(i+1)*cfg['batch_size']]).astype(np.float32))/2.0 - 127.5
        # samples_batch = np.array(train_samples[i*cfg['batch_size']:(i+1)*cfg['batch_size']]).astype(np.float32) / 255.0
        labels_batch = train_labels[i*cfg['batch_size']:(i+1)*cfg['batch_size']]

        outputs = model.forward(samples_batch)
        loss = model.compute_loss(labels_batch)
        grads = model.backward()
        # optimizer.step(grads)

if __name__ == '__main__':
    train()
