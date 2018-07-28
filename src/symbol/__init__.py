from src.symbol.mlp import mlp
from src.symbol.resnet import resnet

symbol_list = {
    'mlp': mlp,
    'resnet': resnet,
}

def get_symbol(name):
    if name not in symbol_list.keys():
        raise RuntimeError()

    return symbol_list[name]
