from collections import defaultdict

cfg = defaultdict(set)

cfg['data_path'] = './data'
cfg['batch_size'] = 5000
cfg['max_epoch'] = 100

cfg['base_lr'] = 1e-3
cfg['momentum'] = 0.9
