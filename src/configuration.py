from collections import defaultdict

cfg_list = defaultdict(set)

cfg_list['mlp'] = {
    'workspace': './ckpt/mlp-v2',
    'symbol': 'mlp',
    'batch_size': 5000,
    'max_epoch': 100,
    'milestones': [ i*10 for i in range(1, 100//10) ],
    'gamma': 0.75,
    'base_lr': 1e-2,
    'momentum': 0.9,
    'reg': 1e-3,
}
