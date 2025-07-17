"""
GraphSAGE-based, Film-conditioned GNN model.

https://arxiv.org/abs/1706.02216
"""


from functools import partial
from typing import Optional

import torch
import torch.nn as nn
import torch_geometric.nn as pygnn

from simshift.models.condition import ContinuousSincosEmbed, Film
from simshift.models.registry import register_model
from simshift.models.utils import MLP


class ModulatedSAGEConv(pygnn.SAGEConv):
    """
    SAGEConv layer with modulation: extends `pygnn.SAGEConv` with FILM-conditioning.

    Args:
        cond_dim (int): Dimension of the conditioning vector.
    """
    def __init__(self, cond_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.modulation = Film(cond_dim, self.in_channels)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, cond: torch.Tensor, size=None
    ):
        x = self.modulation(x, cond)
        return super().forward(x, edge_index, size)


@register_model()
class GraphSAGE(nn.Module):
    """
    Graph-based mesh model with conditioned GraphSAGE layers.

    Args:
        n_conds (int): Conditioning input dimensionality.
        latent_channels (int): Dimension of latent conditioning.
        output_channels (int): Number of output prediction channels.
        act_fn (nn.Module): Activation function.
        dropout_prob (float): Dropout rate in MLPs.
        space (int): Spatial dimension of the mesh.
        gnn_base (int): Hidden size for GNN layers.
        num_layers (int): Number of GNN message-passing layers.
        conditioning_mode (str): 'film' or 'cat' mode.
        out_deformation (bool): If True, predicts mesh coordinate shifts.
        n_materials (Optional[int]): Number of material types (if used).
    """

    def __init__(
        self,
        n_conds: int,
        latent_channels: int = 256,
        output_channels: int = 17,
        act_fn: nn.Module = nn.SiLU,
        dropout_prob: float = 0.1,
        space: int = 2,
        gnn_base: int = 64,
        num_layers: int = 5,
        conditioning_mode: str = "film",
        out_deformation: bool = True,
        n_materials: Optional[int] = None,
        conditioning_bn: bool = False,
    ):
        super().__init__()

        self.space = space
        self.output_channels = output_channels
        assert conditioning_mode in ["cat", "film"]
        self.conditioning_mode = conditioning_mode
        self.out_deformation = out_deformation
        self.activation = act_fn()

        self.conditioning = nn.Sequential(
            ContinuousSincosEmbed(dim=256, ndim=n_conds),
            MLP(
                [256, 256 // 2, 256 // 4, latent_channels],
                act_fn=act_fn,
                last_act_fn=act_fn,
                dropout_prob=dropout_prob,
                batchnorm=conditioning_bn,
            ),
        )

        # encode positions to latent
        self.coord_embed = ContinuousSincosEmbed(dim=gnn_base, ndim=space)
        self.encoder = MLP([gnn_base, gnn_base], act_fn=act_fn)

        # material embedding
        if n_materials is not None:
            self.material_embedding = nn.Embedding(
                num_embeddings=n_materials, embedding_dim=gnn_base
            )

        # message passing processor
        MPBlockType = pygnn.SAGEConv
        if conditioning_mode == "cat":
            self.proj_cond = nn.Linear(latent_channels + gnn_base, gnn_base, bias=False)
        if conditioning_mode == "film":
            # node modulation layer before message passing
            MPBlockType = partial(ModulatedSAGEConv, latent_channels)
        gnn_layers = []
        for _ in range(num_layers):
            gconv = MPBlockType(gnn_base, gnn_base, aggr="mean")
            gnn_layers.append(gconv)
        self.processor = nn.ModuleList(gnn_layers)
        # decode latent to fields + positions
        self.decoder = MLP(
            [gnn_base, output_channels + (space if out_deformation else 0)],
            act_fn,
        )

    def forward(
        self,
        cond: torch.Tensor,
        mesh_coords: torch.Tensor,
        mesh_edges: torch.Tensor,
        mesh_material: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.Tensor] = None,
    ):
        latent_vector = self.conditioning(cond)
        z = latent_vector[batch_index]  # (BxN, C)
        # encoder
        coords = mesh_coords.clone()
        x = self.encoder(self.coord_embed(mesh_coords))  # (BxN, C)

        if mesh_material is not None:
            # add material embedding if we have it
            mesh_material_embedding = self.material_embedding(mesh_material.squeeze())
            x += mesh_material_embedding

        if self.conditioning_mode == "cat":
            x = self.proj_cond(torch.cat([x, z], dim=-1))
            cond = {}
        if self.conditioning_mode == "film":
            cond = {"cond": z}

        # message passing layers
        for layer in self.processor:
            x = layer(x, edge_index=mesh_edges, **cond)
            x = self.activation(x)

        # decoder
        x = self.decoder(x)
        if self.out_deformation:
            x, dpos = x.split([self.output_channels, self.space], -1)
            coords = coords + dpos
        return (x, coords), latent_vector
