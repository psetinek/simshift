import torch

from simshift.model_selection import register_model_selection_algorithm


@register_model_selection_algorithm("SB")
def source_best(source_val_loss, **kwargs):
    """Naive model selection strategy that selects the model based on validation loss
    in the source domain.

    Parameters:
        source_val_loss (torch.Tensor): A tensor of shape [n_models, n_samples]
        representing
            validation losses of each model on source domain validation data.
        **kwargs: Additional keyword arguments (ignored in this method).

    Returns:
        torch.Tensor: A one-hot tensor of shape [n_models], where the index
                      corresponding to the model with the lowest mean validation
                      loss in the source domain is set to 1, and all others are 0.
    """
    _ = kwargs
    # weights shape: [n_models, n_samples]
    # source_val_loss shape: [n_models, n_samples]
    mean_loss = torch.mean(source_val_loss, dim=1)  # [n_models]

    # select model with minimum dev risk
    min_index = torch.argmin(mean_loss)
    model_weights = torch.zeros_like(mean_loss)
    model_weights[min_index] = 1

    return model_weights
