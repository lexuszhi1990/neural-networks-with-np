# -*- coding: utf-8 -*-

import numpy as np
import gzip
from pathlib import Path

data_path = './data/fashion-mnist'
imageset = {
    "training_images": "train-images-idx3-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "training_labels": "train-labels-idx1-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}

def load_fashion_minist(shuffle=False):
    dataset = {}
    for key,value in imageset.items():
        imgset_path = Path(data_path, value)
        assert imgset_path.exists(), "%s not exists" % imgset_path
        with gzip.open(imgset_path.as_posix(), 'rb') as f:
            if value.find('labels') == -1:
                dataset[key] = np.frombuffer(f.read(), np.uint8, offset=16).reshape(-1, 1, 28, 28)
            else:
                dataset[key] = np.frombuffer(f.read(), np.uint8, offset=8)

    return dataset


if __name__ == '__main__':
    ds = load_fashion_minist()
    import pdb; pdb.set_trace()
