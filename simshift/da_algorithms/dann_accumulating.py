"""
Domain Adversarial Neural Networks (DANN) algorithm, with gradient accumulation.
"""

from typing import List, Optional

import torch
import torch.nn as nn
from torch.autograd import Function

from simshift.da_algorithms import AccumulateDAAlgorithm, register_da_algorithm
from simshift.models.utils import MLP


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

@register_da_algorithm("DANN_accumulating")
class DANNAcc(AccumulateDAAlgorithm):
    r"""
    Domain-Adversarial Neural Network (DANN) for domain adaptation, with gradient accumulation.

    Adversarial training with a domain classifier to learn domain-invariant features.

    .. math::

        \mathcal{L}_{\text{DANN}} = \mathcal{L}_{\text{task}} - \lambda
        \mathcal{D}_{\text{domain}}

    Where:

    - :math:`\mathcal{L}_{\text{task}}` is the task loss (e.g., MSE)
    - :math:`\mathcal{D}_{\text{domain}}` is the domain classification loss (from the
    discriminator)
    - :math:`\lambda` is the adaptation weight (corresponds to the `da_loss_weight`
    parameter)

    **Reference**:
    Ganin et al., *"Domain-Adversarial Training of Neural Networks"*
    https://arxiv.org/abs/1505.07818

    Args:
        discriminator_hidden_sizes (List[int]):
            Layer sizes for the domain discriminator.

        **base_class_kwargs:
            Additional arguments for the base `DAAlgorithm` class.
    """

    def __init__(
        self,
        discriminator_hidden_sizes: Optional[List[int]] = None,
        **base_class_kwargs,
    ):
        if discriminator_hidden_sizes is None:
            discriminator_hidden_sizes = [16, 32, 16]
        super().__init__(**base_class_kwargs)
        hidden_sizes = [8] + discriminator_hidden_sizes + [1]
        self.discriminator = MLP(latents=hidden_sizes, act_fn=nn.SiLU).to(self.device)
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.opt.add_param_group({"params": self.discriminator.parameters()})
        self.src_latent_buffer = []
        self.trgt_latent_buffer = []

    def _update(self, src_sample, trgt_sample, p=None, alpha=None, **kwargs):
        assert p is not None and alpha is not None
        _ = kwargs
        self.alpha = alpha * self.da_loss_weight

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

        # gradient reversal and domain classification
        src_latents_reversed = ReverseLayerF.apply(all_src_latents, self.alpha)
        trgt_latents_reversed = ReverseLayerF.apply(all_trgt_latents, self.alpha)

        src_discr_logits = self.discriminator(src_latents_reversed)
        trgt_discr_logits = self.discriminator(trgt_latents_reversed)

        # dicriminator loss
        src_discr_loss = self.bce_loss(
            src_discr_logits, torch.zeros_like(src_discr_logits)
        )
        trgt_discr_loss = self.bce_loss(
            trgt_discr_logits, torch.ones_like(trgt_discr_logits)
        )
        discr_loss = src_discr_loss + trgt_discr_loss

        self.da_loss = discr_loss
        self.loss_dict["da_loss"] = discr_loss.item() * self.logical_batch_size  # so it gets logged correctly in the train loss dict
