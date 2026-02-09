"""
DareGram: Domain Adaptation via Regularized Gram matrices, with gradient accumulation.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast

from simshift.da_algorithms import AccumulateDAAlgorithm, register_da_algorithm


@register_da_algorithm("dare_gram_accumulating")
class DareGramAcc(AccumulateDAAlgorithm):
    r"""
    DARE-GRAM Domain Adaptation, with gradient accumulation.

    DARE-GRAM aligns the source and target latent spaces by decomposing the alignment 
    into two components: **Spectral Scaling** and **Angular Orientation**. 
    
    It operates on the augmented latent representations (appending a bias term) and computes 
    the Gram matrices (uncentered covariances). The method uses Singular Value Decomposition (SVD) 
    to dynamically select the significant components based on an energy threshold, and then aligns:
    1. The magnitude of the top-k singular values (Scale).
    2. The orientation of the pseudo-inverse covariance matrices using Cosine Similarity (Angle).

    The objective function is defined as:

    .. math::

        \mathcal{L}_{\text{DareGram}} = \lambda_{\text{angle}} \mathcal{L}_{\text{angle}} + 
        \lambda_{\text{scale}} \mathcal{L}_{\text{scale}}

    **Augmentation & Gram Calculation:**

    .. math::
        \tilde{\mathbf{H}} = [\mathbf{1}, \mathbf{H}] \in \mathbb{R}^{b \times (d+1)} \\
        \mathbf{G} = \tilde{\mathbf{H}}^\top \tilde{\mathbf{H}}

    **Spectral Decomposition & Rank Selection:**
    
    Let $\sigma$ be the singular values of $\mathbf{G}$. The effective rank $k$ is determined 
    such that the cumulative energy exceeds the threshold $\tau$:
    
    .. math::
        k = \min \{ i \mid \frac{\sum_{j=1}^i \sigma_j}{\sum \sigma} > \tau \}

    **Loss Components:**

    .. math::
        \mathcal{L}_{\text{scale}} = \frac{1}{k} \| \sigma_S[:k] - \sigma_T[:k] \|_2

    .. math::
        \mathcal{L}_{\text{angle}} = \frac{1}{d+1} \sum_{j=1}^{d+1} | 1 - \text{cos\_sim}(\mathbf{G}_S^\dagger[:j], \mathbf{G}_T^\dagger[:j]) |

    Where:

    - :math:`\mathbf{H}` are the latent feature vectors
    - :math:`\mathbf{G}^\dagger` is the pseudo-inverse of the Gram matrix computed with rank :math:`k`
    - :math:`\lambda_{\text{angle}}` (tradeoff_angle) weights the orientation alignment
    - :math:`\lambda_{\text{scale}}` (tradeoff_scale) weights the spectral alignment
    """
    def __init__(self, **base_class_kwargs):
        super().__init__(**base_class_kwargs)
        self.threshold = 0.95
        self.tradeoff_scale = 0.001
        self.tradeoff_angle = 0.02
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

        with autocast(enabled=False):
            # dare gram loss
            b,p = all_src_latents.shape

            H1 = all_src_latents
            H2 = all_trgt_latents

            A = torch.cat((torch.ones(b,1).to(all_src_latents.device), H1), 1)
            B = torch.cat((torch.ones(b,1).to(all_src_latents.device), H2), 1)

            cov_A = (A.t()@A)
            cov_B = (B.t()@B) 

            _,L_A,_ = torch.linalg.svd(cov_A)
            _,L_B,_ = torch.linalg.svd(cov_B)
            
            eigen_A = torch.cumsum(L_A.detach(), dim=0)/L_A.sum()
            eigen_B = torch.cumsum(L_B.detach(), dim=0)/L_B.sum()

            if(eigen_A[1]>self.threshold):
                T = eigen_A[1].detach()
            else:
                T = self.threshold
                
            index_A = torch.argwhere(eigen_A.detach()<=T)[-1]

            if(eigen_B[1]>self.threshold):
                T = eigen_B[1].detach()
            else:
                T = self.threshold

            index_B = torch.argwhere(eigen_B.detach()<=T)[-1]
            
            k = max(index_A, index_B)[0]

            A = torch.linalg.pinv(cov_A ,rtol = (L_A[k]/L_A[0]).detach())
            B = torch.linalg.pinv(cov_B ,rtol = (L_B[k]/L_B[0]).detach())
            
            cos_sim = nn.CosineSimilarity(dim=0,eps=1e-6)
            cos = torch.dist(torch.ones((p+1)).to(all_src_latents.device),(cos_sim(A,B)),p=1)/(p+1)
            
            dare_gram_loss = self.tradeoff_angle*(cos) + self.tradeoff_scale*torch.dist((L_A[:k]),(L_B[:k]))/k

        self.da_loss = self.da_loss_weight * dare_gram_loss
        self.loss_dict["da_loss"] = dare_gram_loss.item() * self.logical_batch_size  # so it gets logged correctly in the train loss dict