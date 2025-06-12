import torch

from simshift.model_selection import register_model_selection_algorithm


@register_model_selection_algorithm("DEV")
def dev(weights: torch.Tensor, source_val_loss: torch.Tensor, **kwargs) -> torch.Tensor:
    r"""
    Deep Embedded Validation (DEV) algorithm for unsupervised model selection.

    Computes a corrected risk estimate by modeling the dependency between
    learned sample weights (estimated density ratios) and validation losses:

    .. math::

        \eta = -\frac{\mathrm{Cov}(w, \ell)}{\mathrm{Var}(w)},
        \quad
        R_{\mathrm{DEV}} = \mathbb{E}[w \ell] + \eta \mathbb{E}[w] - \eta

    Where:

    - :math:`w` (weights): shape :math:`[M, N]` for :math:`M` models, :math:`N`
    validation samples
    - :math:`\ell` (source_val_loss): shape :math:`[M, N]`, validation losses
    - Expectations and variances are computed over the sample dimension

    Selects the model with minimal :math:`R_{\mathrm{DEV}}`.

    **Reference**:
    You et al. *"Towards Accurate Model Selection in Deep Unsupervised Domain
    Adaptation"*
    https://proceedings.mlr.press/v97/you19a.html

    Args:
        weights (Tensor):
            Tensor of shape :math:`[M, N]`. Learned relative weights for :math:`M`
            candidate models across :math:`N` validation samples.

        source_val_loss (Tensor):
            Tensor of shape :math:`[M, N]`. Validation losses where lower values
            indicate better model performance.

        **kwargs:
            Additional keyword arguments (ignored, maintained for API compatibility)

    Returns:
        Tensor:
        Tensor of shape :math:`[M]`. One-hot encoded selection vector where 1 indicates
        the chosen model.
        Example: For 3 models with minimum at index 1,
        returns :code:`tensor([0., 1., 0.])`
    """

    _ = kwargs
    # weights: [n_models, n_samples], source_val_loss: [n_models, n_samples]
    weights_mean = torch.mean(weights, dim=-1)  # [n_models]
    weighted_loss = weights * source_val_loss  # [n_models, n_samples]
    weighted_loss_mean = torch.mean(weighted_loss, dim=1)  # [n_models]

    weighted_loss_centered = weighted_loss - weighted_loss_mean.unsqueeze(
        1
    )  # [n_models, n_samples]
    weights_centered = weights - weights_mean.unsqueeze(1)  # [n_models, n_samples]

    cov_lw = torch.mean(weighted_loss_centered * weights_centered, dim=1)  # [n_models]
    var_w = torch.mean(weights_centered**2, dim=1)  # [n_models]

    eta = -cov_lw / var_w
    R_dev = weighted_loss_mean + eta * weights_mean - eta  # [n_models]

    # select model with minimum dev risk
    min_index = torch.argmin(R_dev)
    model_weights = torch.zeros_like(R_dev)
    model_weights[min_index] = 1

    return model_weights
