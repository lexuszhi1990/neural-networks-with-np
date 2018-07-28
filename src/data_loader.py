# -*- coding: utf-8 -*-

import numpy as np
import gzip
from pathlib import Path

default_data_path = './data/mnist'
imageset = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

class mnist(object):
    def __init__(self, imageset='train', batch_size=1000, data_path=None):

        data_path = default_data_path if data_path is None else data_path

        self.imageset = imageset
        self.batch_size = batch_size
        self.inputs, self.labels = self.load_dataset(data_path)
        self.max_iteration = len(self.labels) // batch_size
        self.anchor = 0

    def name(self):
        return "mnist-%s" % self.imageset

    def is_training(self):
        return self.imageset == 'train'

    def load_dataset(self, data_path):
        sample_path = Path(data_path, imageset["%s_images" % self.imageset])
        label_path = Path(data_path, imageset["%s_labels" % self.imageset])
        assert sample_path.exists(), "%s not exists" % sample_path
        assert label_path.exists(), "%s not exists" % label_path

        with gzip.open(sample_path.as_posix(), 'rb') as f:
            inputs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)

        with gzip.open(label_path.as_posix(), 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return inputs, labels

    def reset(self):
        self.anchor = 0
        if self.is_training():
            shuffled_ids = [i for i in range(len(self.inputs))]
            self.inputs = self.inputs[shuffled_ids]
            self.labels = self.labels[shuffled_ids]

    def __iter__(self):
        return self

    def __next__(self):
        inputs_batch = self.inputs[self.anchor*self.batch_size:(self.anchor+1)*self.batch_size].astype(np.float32)
        labels_batch = self.labels[self.anchor*self.batch_size:(self.anchor+1)*self.batch_size]
        self.anchor += 1
        if self.anchor+1 > self.max_iteration:
            self.reset()
            raise StopIteration();

        return inputs_batch, labels_batch


if __name__ == '__main__':
    dataset = mnist()
    dataset.reset()
