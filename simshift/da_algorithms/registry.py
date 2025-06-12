from typing import Optional

DA_ALGORITHM_REGISTRY = {}


def register_da_algorithm(name: Optional[str] = None):
    """Decorator to register a loss function under a given name."""

    def decorator(fn):
        nonlocal name
        if name is None:
            name = fn.__name__
        DA_ALGORITHM_REGISTRY[name] = fn
        return fn

    return decorator


def get_da_algorithm(name: str):
    """Retrieve a loss function by name from the registry."""
    if name not in DA_ALGORITHM_REGISTRY:
        raise ValueError(
            f"Domain Adaptation algorithm '{name}' is not registered. "
            f"Available algorithms: {list(DA_ALGORITHM_REGISTRY.keys())}"
        )
    return DA_ALGORITHM_REGISTRY[name]
