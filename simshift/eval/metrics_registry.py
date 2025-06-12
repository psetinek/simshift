METRIC_REGISTRY = {}


def register_metric(name):
    """Decorator to register a metric function under a given name."""

    def decorator(fn):
        METRIC_REGISTRY[name] = fn
        return fn

    return decorator


def get_metric(name):
    """Retrieve a metric function by name."""
    if name not in METRIC_REGISTRY:
        raise ValueError(
            f"Metric '{name}' not registered. Available metrics: \
                {list(METRIC_REGISTRY.keys())}"
        )
    return METRIC_REGISTRY[name]
