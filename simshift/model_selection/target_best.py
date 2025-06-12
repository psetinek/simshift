import torch

from simshift.model_selection import register_model_selection_algorithm


@register_model_selection_algorithm("TB")
def target_best(target_test_loss, **kwargs):
    """Selects the best model for each sample based on target test loss. Should only
    be used as a theoretical lower bound for other model selection strategies since
    it uses the test set.

    Parameters:
        target_test_loss (torch.Tensor): A tensor of shape [n_models, n_samples]
                                         representing the loss of each model on each
                                         sample from the target domain.
        **kwargs: Additional keyword arguments (ignored in this method).

    Returns:
        torch.Tensor: A one-hot tensor of shape [n_models], where the index
                      corresponding to the model with the lowest mean validation
                      loss on the testset is set to 1, and all others are 0.
    """
    _ = kwargs

    # selectm minimum model
    min_index = torch.min(target_test_loss, dim=0).indices
    model_weights = torch.zeros_like(target_test_loss)
    model_weights[min_index] = 1

    return model_weights
