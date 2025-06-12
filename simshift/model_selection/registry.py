MODEL_SELECTION_ALGORITHM_REGISTRY = {}


def register_model_selection_algorithm(name):
    """Decorator to register model selection algorithm."""

    def decorator(fn):
        MODEL_SELECTION_ALGORITHM_REGISTRY[name] = fn
        return fn

    return decorator


def get_model_selection_algorithm(name):
    """Retrieve a model selection algorithm by name from the registry."""
    if name not in MODEL_SELECTION_ALGORITHM_REGISTRY:
        raise ValueError(
            f"Model selection algorithm '{name}' is not registered. "
            f"Available algorithms: {list(MODEL_SELECTION_ALGORITHM_REGISTRY.keys())}"
        )
    return MODEL_SELECTION_ALGORITHM_REGISTRY[name]
