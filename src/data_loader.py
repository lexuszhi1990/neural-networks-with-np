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
    def __init__(self, type='train', batch_size=100, data_path=None):

        self.type = type
        self.batch_size = batch_size
        if data_path is None:
            data_path = default_data_path
        self.data_path = data_path

        self.inputs, self.labels = self.load_dataset()
        self.max_iteration = len(self.labels) / batch_size
        self.anchor = 0

    def load_dataset(self):
        sample_path = Path(self.data_path, imageset["%s_images" % self.type])
        label_path = Path(self.data_path, imageset["%s_labels" % self.type])
        assert sample_path.exists(), "%s not exists" % sample_path
        assert label_path.exists(), "%s not exists" % label_path

        with gzip.open(sample_path.as_posix(), 'rb') as f:
            inputs = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)

        with gzip.open(label_path.as_posix(), 'rb') as f:
            labels = np.frombuffer(f.read(), np.uint8, offset=8)

        return inputs, labels

    def __iter__(self):
        return self

    def __next__(self):
        # samples_batch = np.array(train_samples[i*cfg['batch_size']:(i+1)*cfg['batch_size']]).astype(np.float32) / 255.0
        # samples_batch = (np.array(train_samples[i*cfg['batch_size']:(i+1)*cfg['batch_size']]).astype(np.float32))/2.0 - 127.5

        inputs_batch = self.inputs[self.anchor*self.batch_size:(self.anchor+1)*self.batch_size].astype(np.float32)
        labels_batch = self.labels[self.anchor*self.batch_size:(self.anchor+1)*self.batch_size]

        if self.anchor+1 > self.max_iteration:
            raise StopIteration();

        return inputs_batch, labels_batch


if __name__ == '__main__':
    dataset = mnist()
    for i,j in dataset:
        import pdb
        pdb.set_trace()
