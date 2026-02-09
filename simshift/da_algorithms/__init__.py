from .registry import register_da_algorithm, get_da_algorithm
from .base_da_algorithm import DAAlgorithm, AccumulateDAAlgorithm
from .deep_coral import DeepCORAL
from .cmd import CMD
from .dann import DANN
from .dare_gram import DareGram
from .deep_coral_accumulating import DeepCORALAcc
from .cmd_accumulating import CMDAcc
from .dann_accumulating import DANNAcc
from .dare_gram_accumulating import DareGramAcc

__all__ = [
    "register_da_algorithm",
    "get_da_algorithm",
    "AccumulateDAAlgorithm",
    "DAAlgorithm",
    "DeepCORAL",
    "CMD",
    "DANN",
    "DareGram",
    "DeepCORALAcc",
    "CMDAcc",
    "DANNAcc",
    "DareGramAcc",
]
