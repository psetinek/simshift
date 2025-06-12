from .registry import register_da_algorithm, get_da_algorithm
from .base_da_algorithm import DAAlgorithm
from .deep_coral import DeepCORAL
from .cmd import CMD
from .dann import DANN

__all__ = [
    "register_da_algorithm",
    "get_da_algorithm",
    "DAAlgorithm",
    "DeepCORAL",
    "CMD",
    "DANN",
]
