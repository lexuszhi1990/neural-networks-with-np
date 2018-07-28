from collections import defaultdict

cfg_list = defaultdict(set)

cfg_list['mlp'] = {
    'workspace': './ckpt/mlp-v3',
    'dataset_name': 'mnist',
    'symbol': 'mlp',
    'batch_size': 10000,
    'max_epoch': 100,
    'milestones': [ i*10 for i in range(1, 100//10) ],
    'gamma': 0.6,
    'base_lr': 5e-2,
    'momentum': 0.9,
    'reg': 5e-3,
}
