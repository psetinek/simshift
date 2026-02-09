"""
Central Moment Discrepancy (CMD) algorithm, with gradient accumulation.
"""

import torch
from simshift.da_algorithms import AccumulateDAAlgorithm, register_da_algorithm


@register_da_algorithm("cmd_accumulating")
class CMDAcc(AccumulateDAAlgorithm):
    r"""
    Central Moment Discrepancy (CMD) for domain adaptation, with gradient accumulation.

    Matches higher-order central moments between source and target distributions.

    .. math::

        \mathcal{L}_{\text{CMD}} = \frac{1}{|b - a|} \left\| \mathbf{E}(X_S) -
        \mathbf{E}(X_T) \right\|_2 +
        \sum_{k=2}^K \frac{1}{|b - a|^k} \left\| c_k(X_S) - c_k(X_T) \right\|_2

    Where:

    - :math:`\mathbf{E}(\cdot)` is the expectation (1st moment)
    - :math:`c_k(\cdot)` are the :math:`k`-th order central moments
    - :math:`[a, b]` is the feature value range
    - :math:`K` is the number of moments (specified by the `n_moments` parameter)

    **Reference**:
    Zellinger et al., *"Central Moment Discrepancy (CMD) for Domain-Invariant
    Representation Learning"*
    https://arxiv.org/abs/1702.08811

    Args:
        n_moments (int):
            Number of moments to match. Defaults to 5.

        scaling_factor (float):
            Scaling factor for moment differences (corresponding to :math:`|b - a|`).

        **base_class_kwargs:
            Additional arguments for the base `DAAlgorithm` class.
    """

    def __init__(
        self,
        n_moments: int = 5,
        scaling_factor=None,
        **base_class_kwargs,
    ):
        super().__init__(**base_class_kwargs)
        assert n_moments >= 1
        self.n_moments = n_moments
        self.scaling_factor = scaling_factor
        self.src_latent_buffer = []
        self.trgt_latent_buffer = []

    def _update(self, src_sample, trgt_sample, **kwargs):
        _ = kwargs
        # predictions
        src_pred, src_latents = self.model(**src_sample.as_dict())
        src_pred, pred_coords = src_pred
        _, trgt_latents = self.model(**trgt_sample.as_dict())
        self.src_latent_buffer.append(src_latents)
        self.trgt_latent_buffer.append(trgt_latents)

        # positions loss
        pos_loss = self.mse_loss(pred_coords, src_sample.y_mesh_coords)

        # prediction loss
        mse_loss = self.mse_loss(src_pred, src_sample.y)

        # set total loss
        self.loss = pos_loss + mse_loss

        # loss dictionary
        self.loss_dict["mse_loss"] = mse_loss.item()
        self.loss_dict["summed_loss"] = self.loss.item()
    
    def _update_da_loss(self):
        all_src_latents = torch.cat(self.src_latent_buffer, dim=0)
        all_trgt_latents = torch.cat(self.trgt_latent_buffer, dim=0)
        self.src_latent_buffer.clear()
        self.trgt_latent_buffer.clear()

        cmd_loss = self._cmd_loss(all_src_latents, all_trgt_latents)
        self.da_loss = self.da_loss_weight * cmd_loss
        self.loss_dict["da_loss"] = cmd_loss.item() * self.logical_batch_size  # so it gets logged correctly in the train loss dict

    def _cmd_loss(self, source_features, target_features):
        # calculate means
        sm = torch.mean(source_features, dim=0)
        tm = torch.mean(target_features, dim=0)

        # centralize
        sc = source_features - sm
        tc = target_features - tm

        # calculate moments
        central_moments_source = []
        central_moments_target = []
        for i in range(self.n_moments - 1):
            central_moments_source.append(self._calculate_moments(sc, i + 2))
            central_moments_target.append(self._calculate_moments(tc, i + 2))

        # calc loss
        if self.scaling_factor is None:
            scaling_factor = (
                torch.max(source_features.max(), target_features.max()) - 0.28
            )
        else:
            scaling_factor = self.scaling_factor
        res = torch.linalg.norm(sm - tm, ord=2) / scaling_factor
        for i in range(len(central_moments_source)):
            res += torch.linalg.norm(
                central_moments_source[i] - central_moments_target[i], ord=2
            ) / scaling_factor ** (i + 2)
        return res

    def _calculate_moments(self, cetralized_features, moment):
        n = cetralized_features.shape[0]
        return torch.sum(cetralized_features**moment, dim=1) / (n - 1)
