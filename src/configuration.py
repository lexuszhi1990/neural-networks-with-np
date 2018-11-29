from collections import defaultdict

cfg_list = defaultdict(set)

cfg_list['mlp'] = {
    'workspace': './ckpt/mlp-v6',
    'dataset_name': 'mnist',
    'symbol': 'mlp',
    'batch_size': 10000,
    'max_epoch': 100,
    'milestones': [ i*25 for i in range(1, 100//25) ],
    'base_lr': 5e-1,
    'gamma': 0.1,
    'momentum': 0.9,
    'reg': 1e-4,
}


cfg_list['alexnet'] = {
    'workspace': './ckpt/alexnet-v3',
    'dataset_name': 'mnist',
    'symbol': 'alexnet',
    'batch_size': 1000,
    'max_epoch': 3,
    'milestones': [ i for i in range(1, 3) ],
    'base_lr': 1e-1,
    'gamma': 0.1,
    'momentum': 0.9,
    'reg': 1e-3,
}


cfg_list['alexnet-fashion'] = {
    'workspace': './ckpt/alexnet-fashion-v1',
    'dataset_name': 'fashion-mnist',
    'symbol': 'alexnet',
    'batch_size': 1000,
    'max_epoch': 9,
    'milestones': [ i*3 for i in range(1, 9//3) ],
    'base_lr': 1e-1,
    'gamma': 0.1,
    'momentum': 0.9,
    'reg': 1e-3,
}
