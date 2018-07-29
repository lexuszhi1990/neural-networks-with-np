from collections import defaultdict

cfg_list = defaultdict(set)

cfg_list['mlp'] = {
    'workspace': './ckpt/mlp-v3',
    'dataset_name': 'mnist',
    'symbol': 'mlp',
    'batch_size': 10000,
    'max_epoch': 100,
    'milestones': [ i*20 for i in range(1, 100//20) ],
    'base_lr': 5e-1,
    'gamma': 0.5,
    'momentum': 0.9,
    'reg': 1e-3,
}


cfg_list['alexnet'] = {
    'workspace': './ckpt/alexnet-v1',
    'dataset_name': 'mnist',
    'symbol': 'alexnet',
    'batch_size': 10000,
    'max_epoch': 100,
    'milestones': [ i*10 for i in range(1, 100//10) ],
    'base_lr': 5e-1,
    'gamma': 0.5,
    'momentum': 0.9,
    'reg': 1e-3,
}
