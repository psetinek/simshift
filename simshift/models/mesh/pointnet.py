"""
PointNet conditional model for mesh modeling.

https://arxiv.org/abs/1612.00593
"""


from typing import Optional

import torch
import torch.nn as nn

from simshift.models.condition import ContinuousSincosEmbed
from simshift.models.registry import register_model
from simshift.models.utils import MLP


@register_model()
class PointNet(nn.Module):
    """
    Conditional PointNet architecture.

    Args:
        n_conds (int): Dimension of conditioning inputs.
        latent_channels (int): Latent embedding size.
        output_channels (int): Number of output channels.
        act_fn (nn.Module): Activation function.
        dropout_prob (float): Dropout rate.
        space (int): Spatial dimension.
        pointnet_base (int): Base feature size for PointNet layers.
        out_deformation (bool): Predict coordinate deformations if True.
        n_materials (Optional[int]): Number of material types (if used).
    """

    def __init__(
        self,
        n_conds: int,
        latent_channels: int = 128,
        output_channels: int = 17,
        act_fn: nn.Module = nn.SiLU,
        dropout_prob: float = 0.1,
        space: int = 2,
        pointnet_base: int = 8,
        out_deformation: bool = True,
        n_materials: Optional[int] = None,
    ):
        super().__init__()

        self.space = space
        self.output_channels = output_channels
        self.out_deformation = out_deformation

        self.conditioning = nn.Sequential(
            ContinuousSincosEmbed(dim=256, ndim=n_conds),
            MLP(
                [256, 256 // 2, 256 // 4, latent_channels],
                act_fn=act_fn,
                last_act_fn=act_fn,
                dropout_prob=dropout_prob,
            ),
        )

        # encode positions to latent
        self.coord_embed = ContinuousSincosEmbed(dim=latent_channels, ndim=space)
        self.encoder = MLP(
            [latent_channels, latent_channels], act_fn=act_fn, dropout_prob=dropout_prob
        )

        # material embedding
        if n_materials is not None:
            self.material_embedding = nn.Embedding(
                num_embeddings=n_materials, embedding_dim=latent_channels
            )

        # pointnet processor
        self.in_block = MLP(
            [latent_channels, pointnet_base, pointnet_base * 2],
            act_fn=act_fn,
            dropout_prob=dropout_prob,
        )
        self.max_block = MLP(
            [
                pointnet_base * 2,
                pointnet_base * 4,
                pointnet_base * 8,
                pointnet_base * 32,
            ],
            act_fn=act_fn,
            dropout_prob=dropout_prob,
        )
        self.out_block = MLP(
            [
                pointnet_base * (32 + 2) + latent_channels,  # (globals + locals + cond)
                pointnet_base * 16,
                pointnet_base * 8,
                pointnet_base * 4,
                latent_channels,
            ],
            act_fn=act_fn,
            dropout_prob=dropout_prob,
        )

        self.decoder = MLP(
            [latent_channels, output_channels + (space if out_deformation else 0)],
            act_fn,
            dropout_prob=dropout_prob,
        )

    def forward(
        self,
        cond: torch.Tensor,
        mesh_coords: torch.Tensor,
        mesh_edges: torch.Tensor,
        mesh_material: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.Tensor] = None,
    ):
        _ = mesh_edges
        latent_vector = self.conditioning(cond)
        # encoder
        coords = mesh_coords.clone()
        x = self.encoder(self.coord_embed(mesh_coords))  # (BxN, C)

        if mesh_material is not None:
            # add material embedding if we have it
            mesh_material_embedding = self.material_embedding(mesh_material.squeeze())
            x += mesh_material_embedding

        x = self.in_block(x)

        x_m = self.max_block(x)
        bs = batch_index.max() + 1
        global_x = torch.zeros((bs, x_m.shape[-1]), device=x.device, dtype=x.dtype)
        global_x.scatter_reduce_(
            0, batch_index.unsqueeze(1).repeat(1, x_m.shape[-1]), x_m, reduce="amax"
        )  # (B, C)
        # count points per mesh
        points = torch.zeros(bs, device=x.device, dtype=x.dtype)
        ones = torch.ones_like(batch_index, device=x.device, dtype=x.dtype)
        points.scatter_reduce_(0, batch_index, ones, reduce="sum")
        points = points.long()
        # concatenate conditioning to globals
        global_x = torch.cat([global_x, latent_vector], dim=-1)  # (B, gC+C)
        global_x = torch.repeat_interleave(global_x, points, dim=0)
        x = torch.cat([x, global_x], dim=1)  # (BxN, lC+gC+C)
        x = self.out_block(x)  # (BxN, C)

        # decoder
        x = self.decoder(x)

        if self.out_deformation:
            x, dpos = x.split([self.output_channels, self.space], -1)
            coords = coords + dpos

        return (x, coords), latent_vector
