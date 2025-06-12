from typing import Optional

MODEL_REGISTRY = {}


def register_model(name: Optional[str] = None):
    def decorator(fn):
        nonlocal name
        if name is None:
            name = fn.__name__
        MODEL_REGISTRY[name] = fn
        return fn

    return decorator


def get_model_class(name: str):
    reg = MODEL_REGISTRY
    if name not in reg:
        raise ValueError(
            f"Baseline '{name}' is not registered."
            f"Available baselines: {list(reg.keys())}"
        )
    return reg[name]
