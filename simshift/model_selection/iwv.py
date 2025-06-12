import torch

from simshift.model_selection import register_model_selection_algorithm


@register_model_selection_algorithm("IWV")
def iwv(weights: torch.Tensor, source_val_loss: torch.Tensor, **kwargs) -> torch.Tensor:
    r"""
    Importance-Weighted Validation (IWV) algorithm for unsupervised model selection.

    Estimates model risk through a weighted average of validation losses:

    .. math::

        R_{\mathrm{IWV}}^{(i)} = \frac{1}{N} \sum_{j=1}^{N} w_{ij}\,\ell_{ij}

    Where for :math:`M` candidate models and :math:`N` validation samples:

    - :math:`w_{ij}`: Learned weight for model :math:`i` on sample :math:`j`  
    (typically :math:`\sum_j w_{ij} = 1`)
    - :math:`\ell_{ij}`: Validation loss for model :math:`i` on sample :math:`j`

    Selects the model with minimal :math:`R_{\mathrm{IWV}}`.

    **Reference**:  
    Sugiyama et al., *"Importance-Weighted Validation for Robust Model Selection"*  
    https://jmlr.org/papers/v8/sugiyama07a.html

    Args:
        weights (Tensor):  
            Tensor of shape :math:`[M, N]`. Learned relative weights for :math:`M`
            candidate models across :math:`N` validation samples.

        source_val_loss (Tensor):  
            Tensor of shape :math:`[M, N]`. Validation losses where lower values
            indicate better model performance.

        **kwargs:  
            Ignored arguments (maintained for API compatibility)

    Returns:
        Tensor:  
            Tensor of shape :math:`[M]`. One-hot encoded selection vector where 1
            indicates the chosen model.  
            Example: For 3 models with minimum at index 1, returns
            :code:`tensor([0., 1., 0.])`
    """

    _ = kwargs
    # weighted validation loss
    weighted_loss = weights * source_val_loss
    iwv_risk = torch.mean(weighted_loss, axis=-1)

    # only take the model that with minimum iwv
    min_index = torch.min(iwv_risk, dim=0).indices
    model_weights = torch.zeros_like(iwv_risk)
    model_weights[min_index] = 1

    return model_weights
