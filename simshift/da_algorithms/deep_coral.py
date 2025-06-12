"""
CORrelation ALignment (CORAL) for Deep Domain Adaptation algorithm.
"""

import torch

from simshift.da_algorithms import DAAlgorithm, register_da_algorithm


@register_da_algorithm("deep_coral")
class DeepCORAL(DAAlgorithm):
    r"""
    Deep CORrelation ALignment (CORAL) for Domain Adaptation.

    DeepCORAL aligns covariances of the latent spaces for source and target inputs.

    .. math::

        \mathcal{L}_{\text{CORAL}} = \frac{1}{4d^2} \left\| \mathbf{C}_S - \mathbf{C}_T
        \right\|_F^2

    Where:

    - :math:`\mathbf{C}_S` is the source feature covariance matrix
    - :math:`\mathbf{C}_T` is the target feature covariance matrix
    - :math:`d` is the feature dimension
    - :math:`\|\cdot\|_F` denotes the Frobenius norm

    **Reference**:
    Sun et al., *"Deep CORAL: Correlation Alignment for Deep Domain Adaptation"*
    https://arxiv.org/abs/1607.01719
    """

    def __init__(self, **base_class_kwargs):
        super().__init__(**base_class_kwargs)

    def _update(self, src_sample, trgt_sample, **kwargs):
        _ = kwargs
        # predictions
        src_pred, src_latents = self.model(**src_sample.as_dict())
        # src_pred, src_latents, supernode_index = self.model(**src_sample.as_dict())
        src_pred, pred_coords = src_pred
        _, trgt_latents = self.model(**trgt_sample.as_dict())
        # _, trgt_latents, _ = self.model(**trgt_sample.as_dict())

        # positions loss
        pos_loss = self.mse_loss(pred_coords, src_sample.y_mesh_coords)

        # prediction loss
        mse_loss = self.mse_loss(src_pred, src_sample.y)
        # mse_loss = self.mse_loss(src_pred, src_sample.y[supernode_index])

        # coral loss
        coral_loss = self._coral_loss(src_latents, trgt_latents)

        # set total loss
        self.loss = pos_loss + mse_loss + self.da_loss_weight * coral_loss
        # self.loss = pos_loss + mse_loss

        # loss dictionary
        self.loss_dict["mse_loss"] = mse_loss.item()
        self.loss_dict["da_loss"] = coral_loss.item()
        self.loss_dict["summed_loss"] = self.loss.item()

    def _coral_loss(self, source_features, target_features):
        """Implementation adapted from https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/coral.py."""
        d = source_features.size(1)  # dim vector

        source_c = self._compute_covariance(source_features)
        target_c = self._compute_covariance(target_features)

        loss = torch.sum(torch.mul((source_c - target_c), (source_c - target_c)))

        loss = loss / (4 * d * d)
        return loss

    def _compute_covariance(self, input_data):
        """Implementation adapted from https://github.com/DenisDsh/PyTorch-Deep-CORAL/blob/master/coral.py."""
        n = input_data.size(0)  # batch_size

        id_row = torch.ones((1, n), device=input_data.device)

        sum_column = torch.mm(id_row, input_data)
        term_mul_2 = torch.mm(sum_column.t(), sum_column) / n
        d_t_d = torch.mm(input_data.t(), input_data)

        return (d_t_d - term_mul_2) / (n - 1)
