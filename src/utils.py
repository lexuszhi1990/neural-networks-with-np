# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
import json

import matplotlib.pyplot as plt

def check_dir_exists(path):
    if not Path(path).exists():
        Path(path).mkdir(parents=True)

def save_weights(params, path, prefix, epoch):
    class np_encoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(np_encoder, self).default(obj)

    params_path = Path(path, '%s-%d.json'%(prefix, epoch))
    with params_path.open('w') as f:
        json.dump(params, f, cls=np_encoder)

    return params_path

def load_weights(path):
    assert Path(path).exists(), "%s not exists" % path
    with Path(path).open('r') as f:
        weights = json.load(f)

    return weights

def restore_weights(model, params_path):
    weights = load_weights(params_path)
    for key in model.params.keys():
        model.params[key] = np.array(weights[key]).astype(np.float32)

def img_preprocess(img):
    return img / 255.

def transfer_samples(path):
    import cv2
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (28, 28))
    cv2.imwrite(img_path.split('/')[-1], img)

def cal_precision(inputs, label):
    assert len(inputs) == len(label), "predicts and lable dont match"
    results = inputs.argmax(axis=1)
    true_num = np.sum(results == label)
    return true_num, true_num/len(label)

def draw_loss_graph(path, training_loss, test_loss=[]):
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.plot(np.arange(len(training_loss)), training_loss, color='#2A6EA6', label="Loss on the training data")
    if len(test_loss) > 0:
        ax.plot(np.arange(len(test_loss)), test_loss, color='#FFA933', label="Loss on the testing data")

    ax.grid(True)
    ax.set_xlim([0, len(training_loss)])
    ax.set_xlabel('Epoch')
    ax.set_ylim([0, 1])
    plt.legend(loc="lower right")
    plt.savefig(path)
