from src.symbol.mlp import mlp
from src.symbol.alexnet import alexnet
from src.symbol.resnet import resnet

symbol_list = {
    'mlp': mlp,
    'alexnet': alexnet,
    'resnet': resnet,
}

def get_symbol(name):
    if name not in symbol_list.keys():
        raise RuntimeError()

    return symbol_list[name]
