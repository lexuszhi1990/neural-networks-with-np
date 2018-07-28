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

def load_weights():
    pass
