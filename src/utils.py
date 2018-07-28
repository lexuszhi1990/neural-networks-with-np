# -*- coding: utf-8 -*-

import numpy as np
from pathlib import Path
import json

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
        model.params[key] = weights[key]

