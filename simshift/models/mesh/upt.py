# Emmi AI Code License Agreement (Non-Commercial, Non-Transferable)

# This License Agreement is made between Emmi AI GmbH (“Licensor”) and RISC (“Licensee”).
# It governs Licensee’s use of the software code provided by Licensor (the “Software”).

# The parties agree to the following terms:

# 1. License Grant and Permitted Use

# Licensor grants Licensee a perpetual, worldwide, non-exclusive, non-transferable license
# to use, copy, and modify the
# Software solely for Licensee’s internal, non-commercial purposes. This license does not
# transfer ownership of the
# Software; all intellectual property rights in the Software remain with the Licensor. All
# rights not expressly granted
# to Licensee are reserved by Licensor.

# 2. Restrictions

# Licensee must not engage in any of the following activities:

# - No Commercial Use: The Software may not be used in any product or service that is sold
#                      or that generates revenue.
#   Licensee shall not use the Software for any commercial purpose or for monetary gain.

# - No Distribution or Sharing: Licensee shall not publish, distribute, share, or make the
#                               Software (or any modified
#   version or derivative work of the Software) available to any third party. The Software
#   is for Licensee’s use only and
#   may not be sublicensed, assigned, or transferred to others in any form.

# - Preservation of Notices: Licensee shall not remove or alter any copyright, trademark,
#                            or attribution notices present in the Software. Any copies
#                            or modifications of the Software that Licensee creates must
#                            include all original notices.


# 3. Attribution Requirement

# Licensee must give appropriate credit to Emmi AI GmbH in any use of the Software or in 
# any derivative works or materials based on the Software. This attribution should be
# clear and reasonably prominent, for example by citing Emmi AI GmbH as the source of the
# code in documentation, research papers, or other materials where the Software or its
# derivatives are used.


# 4. Term and Termination

# This Agreement is effective from the date Licensee receives the Software and remains in
# force perpetually (with no expiration date), unless terminated as described here.
# Licensor may terminate this Agreement immediately upon written notice if Licensee
# breaches any of the terms and conditions. Upon termination, Licensee must immediately
# cease all use of the Software and destroy any copies of the Software and any derivative
# works in Licensee’s possession. The provisions of this Agreement relating to
# intellectual property ownership, attribution, disclaimer of warranties,
# limitation of liability, and governing law shall survive any termination of the
# Agreement.


# 5. Disclaimer of Warranties

# The Software is provided on an “as is” basis, without any warranties of any kind.
# Licensor makes no guarantees or conditions, express or implied, regarding the Software.
# This includes, but is not limited to, implied warranties of merchantability, fitness
# for a particular purpose, and non-infringement. Licensee assumes all risks associated
# with the use or performance of the Software.


# 6. Limitation of Liability

# To the maximum extent permitted by law, Licensor shall not be liable for any damages
# arising out of or relating to this Agreement or Licensee’s use of the Software. This
# exclusion applies to all types of damages or losses, including direct, indirect,
# incidental, consequential, special, exemplary, or punitive damages (such as loss of
# profits, loss of data, business interruptions, or personal injuries), even if Licensor
# has been advised of the possibility of such damages. Licensee agrees that Licensor’s
# total cumulative liability, if any, for any claims relating to the Software or
# this Agreement will not exceed the amount of zero Euros (EUR 0), since the Software is
# provided at no charge.


# 7. Governing Law

# This License Agreement is governed by and construed in accordance with the laws of
# Austria. All disputes arising from or related to this Agreement shall be subject to the
# jurisdiction of the competent courts of Austria. Both Licensor and Licensee agree to
# submit to the personal jurisdiction of such courts, if necessary, in order to resolve
# any disputes.


"""
Universal Physics Transformers: A Framework For Efficiently Scaling Neural Operators

https://arxiv.org/abs/2402.12365
"""


from typing import Any, Optional

import einops
import torch
import torch_geometric
import torch_scatter
from kappamodules.transformer import DitBlock, DitPerceiverBlock
from torch import nn
from torch_geometric.utils import to_dense_batch

from simshift.models.condition import ContinuousSincosEmbed
from simshift.models.functional import init_truncnormal_zero_bias
from simshift.models.registry import register_model
from simshift.models.utils import MLP


class LinearProjection(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        ndim: None | int = None,
        bias: bool = True,
        optional: bool = False,
        init_weights: str = "torch",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.ndim = ndim
        self.bias = bias
        self.init_weights = init_weights

        self.proj: nn.Module
        if optional and input_dim == output_dim:
            self.proj = nn.Identity()
        elif ndim is None:
            self.proj = nn.Linear(input_dim, output_dim, bias=bias)
        elif ndim == 1:
            self.proj = nn.Conv1d(input_dim, output_dim, kernel_size=1, bias=bias)
        elif ndim == 2:
            self.proj = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)
        elif ndim == 3:
            self.proj = nn.Conv3d(input_dim, output_dim, kernel_size=1, bias=bias)
        else:
            raise NotImplementedError

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_weights == "torch":
            pass
        elif self.init_weights in ["truncnormal", "truncnormal002"]:
            init_truncnormal_zero_bias(self.proj)
        else:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class SupernodePooling(nn.Module):
    """
    Supernode pooling layer with the following default settings:

    - Messages from incoming nodes are averaged (:code:`aggregation="mean`)
    - After message passing, the original positional embedding of the supernode is
      concatenated to the supernode vector, followed by a downprojection to the original
      hidden dimension.
    - The permutation of the supernodes is preserved through message passing
      (unlike in the UPT code).
    - A radius-based graph is used instead of `radius_graph`, which is more efficient.

    Args:
        radius (float):  
            Radius around each supernode. Messages are passed from points within radius.

        hidden_dim (int):  
            Hidden dimension for positional embeddings, messages, and output vectors.

        input_dim (int):  
            Number of input features (set to `None` if only positions are used).

        ndim (int):  
            Number of positional dimensions (e.g., `ndim=2` for 2D, `ndim=3` for 3D).

        max_degree (int):  
            Maximum degree for the radius graph.

        init_weights (str):  
            Weight initialization method for linear layers.

        readd_supernode_pos (bool):  
            If `True`, the absolute positional encoding of the supernode is concatenated
            to the supernode vector after message passing and projected to `hidden_dim`.

        aggregation (str):  
            Aggregation method for message passing. One of `"mean"` or `"sum"`.
    """


    def __init__(
        self,
        hidden_dim: int,
        ndim: int,
        input_dim: int | None = None,
        radius: float | None = None,
        max_degree: int = 32,
        init_weights: str = "truncnormal002",
        readd_supernode_pos: bool = True,
        aggregation: str = "mean",
    ):
        super().__init__()
        self.radius = radius
        self.max_degree = max_degree
        self.hidden_dim = hidden_dim
        self.ndim = ndim
        self.init_weights = init_weights
        self.readd_supernode_pos = readd_supernode_pos
        self.aggregation = aggregation
        self.input_dim = input_dim

        self.pos_embed = ContinuousSincosEmbed(dim=hidden_dim, ndim=ndim)
        if input_dim is None:
            self.input_proj = None
        else:
            self.input_proj = LinearProjection(
                input_dim, hidden_dim, init_weights=init_weights
            )
        self.message = nn.Sequential(
            LinearProjection(hidden_dim * 2, hidden_dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(hidden_dim, hidden_dim, init_weights=init_weights),
        )
        if readd_supernode_pos:
            self.proj = LinearProjection(
                2 * hidden_dim, hidden_dim, init_weights=init_weights
            )
        else:
            self.proj = None
        self.output_dim = hidden_dim

    def forward(
        self,
        input_pos: torch.Tensor,
        supernode_idx: torch.Tensor,
        input_feat: Optional[torch.Tensor] = None,
        batch_idx: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        assert input_pos.ndim == 2
        assert input_feat is None or input_feat.ndim == 2
        assert supernode_idx.ndim == 1

        # radius graph
        if batch_idx is None:
            batch_y = None
        else:
            batch_y = batch_idx[supernode_idx]
        edges = torch_geometric.nn.pool.radius(
            x=input_pos,
            y=input_pos[supernode_idx],
            r=self.radius,
            max_num_neighbors=self.max_degree,
            batch_x=batch_idx,
            batch_y=batch_y,
        )
        # remap dst indices
        dst_idx, src_idx = edges.unbind()
        dst_idx = supernode_idx[dst_idx]

        # create message
        x = self.pos_embed(input_pos)
        if self.input_dim is not None:
            assert input_feat is not None
            x = x + self.input_proj(input_feat)
        supernode_pos_embed = x[supernode_idx]
        x = torch.concat([x[src_idx], x[dst_idx]], dim=1)
        x = self.message(x)

        # accumulate messages
        # indptr is a tensor of indices betweeen which to aggregate
        # i.e. a tensor of [0, 2, 5] would result in
        # [src[0] + src[1], src[2] + src[3] + src[4]]
        dst_indices, counts = dst_idx.unique_consecutive(return_counts=True)
        assert torch.all(supernode_idx == dst_indices)
        # first index has to be 0
        # NOTE: padding for target indices that don't occour is not needed as self-loop
        # is always present
        padded_counts = torch.zeros(
            len(counts) + 1, device=counts.device, dtype=counts.dtype
        )
        padded_counts[1:] = counts
        indptr = padded_counts.cumsum(dim=0)
        x = torch_scatter.segment_csr(src=x, indptr=indptr, reduce=self.aggregation)

        # sanity check: dst_indices has len of batch_size * num_supernodes and has to be
        # divisible by batch_size
        # if num_supernodes is not set in dataset this assertion fails
        if batch_idx is None:
            batch_size = 1
        else:
            batch_size = batch_idx.max() + 1
            assert dst_indices.numel() % batch_size == 0

        # convert to dense tensor (dim last)
        x = einops.rearrange(
            x,
            "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
            batch_size=batch_size,
        )

        if self.readd_supernode_pos:
            supernode_pos_embed = einops.rearrange(
                supernode_pos_embed,
                "(batch_size num_supernodes) dim -> batch_size num_supernodes dim",
                batch_size=len(x),
            )
            # concatenate input and supernode embeddings
            x = torch.concat([x, supernode_pos_embed], dim=-1)
            x = self.proj(x)

        return x


class DeepPerceiverDecoderConditioned(nn.Module):
    """A perceiver decoder model.

    Args:
        dim: Dimension of the decoder.
        num_heads: Number of heads in the decoder.
        ndim: Number of dimensions for the position input.
        input_dim: Dimension of the input.
        output_dim: Dimension of the output.
        depth: Depth of the decoder.
        block_ctor: Block constructor.
        init_weights: Initialization method for the weights.
        mlp_expansion_factor: Ratio of the hidden dimension of the MLPs.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ndim: int,
        input_dim: int,
        output_dim: int,
        depth: int = 1,
        block_ctor: type = DitPerceiverBlock,
        init_weights: str = "truncnormal002",
        eps: float = 1e-6,
        mlp_expansion_factor: int = 4,
    ):
        super().__init__()
        # create query
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.query = nn.Sequential(
            LinearProjection(dim, dim, init_weights=init_weights),
            nn.GELU(),
            LinearProjection(dim, dim, init_weights=init_weights),
        )
        # perceiver
        self.proj = LinearProjection(
            input_dim, dim, init_weights=init_weights, optional=True
        )
        self.blocks = nn.ModuleList(
            [
                block_ctor(
                    dim=dim,
                    num_heads=num_heads,
                    # init_weights=init_weights,
                    # eps=eps,
                    # mlp_hidden_dim=dim * mlp_expansion_factor,
                    cond_dim=8,
                    drop_path=0,
                )
                for _ in range(depth)
            ],
        )
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.pred = LinearProjection(dim, output_dim, init_weights=init_weights)

    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        cond: torch.Tensor,
        block_kwargs: dict[str, Any] | None = None,
        unbatch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x: Latent tokens as dense tensor (batch_size, num_latent_tokens, dim).
            pos: Query positions (batch_size, num_output_pos, pos_dim).
            block_kwargs: Additional arguments for the block.
            unbatch_mask: Unbatch mask.
        
        Returns:
            The predictions as sparse tensor
            (batch_size * num_output_pos, num_out_values).
        """
        assert x.ndim == 3
        assert pos.ndim == 3

        # create query
        query = self.query(self.pos_embed(pos))

        # project to perceiver dim
        x = self.proj(x)

        # perceiver
        for block in self.blocks:
            query = block(q=query, kv=x, cond=cond, **(block_kwargs or {}))

        # predict value
        query = self.pred(self.norm(query))

        # dense tensor (batch_size, max_num_points, dim) -> sparse tensor
        # (batch_size * num_points, dim)
        query = einops.rearrange(
            query, "batch_size max_num_points dim -> (batch_size max_num_points) dim"
        )
        if len(pos) == 1:
            # batch_size=1 -> no padding is needed
            pass
        else:
            if unbatch_mask is not None:
                query = query[unbatch_mask]
        assert isinstance(query, torch.Tensor)
        return query


@register_model()
class UPT(nn.Module):
    """UPT model with two decoders (field-based and point-based).

    Parameters:
        n_conds: Number of conditioning parameters.
        latent_channels: Latent (conditioning) vector dimension.
        supernodes_radius: Radius for supernode pooling.
        output_channels: Number of output features to predict
                         (e.g., 1 for surface pressure prediction).
        upt_base: Hidden dimension to use for the whole model.
                  Typical values are 192, 384 or 768. Default: 192
        app_depth: How many approximator blocks (i.e, transformer blocks to use).
                   Default: 12
        space: Number of dimension in the domain. Typically set to 3 for 3D coordinates.
               Default: 3
        input_dim: If input positions have additional features
                   (e.g., pressure or velocity in CFD), input_dim defines how many
                   additional features are used. If defined, requires input_features
                   in the `forward` method.
        dec_depth: How many decoder blocks to use for the field-based perceiver decoder.
                   If set to 0, only the point-based decoder is used. Default: 0
        mlp_expansion_factor: Expansion factor for the MLP layers of
                              transformer/perciever blocks. Default: 4
        num_supernotes: Number of supernodes to sample.
    """

    def __init__(
        self,
        n_conds: int,
        latent_channels: int = 8,
        supernodes_radius: float = 0.05,
        output_channels: int = 5,
        upt_base: int = 192,
        app_depth: int = 12,
        num_heads: int = 3,
        space: int = 3,
        input_dim: int | None = None,
        dec_depth: int = 0,
        mlp_expansion_factor: int = 4,
        num_supernodes: int = 8000,
        supernodes_max_neighbours: int = 32,
        out_deformation: bool = False,
        n_materials: int = 1,
    ):
        super().__init__()
        self.num_supernodes = num_supernodes

        self.conditioning = nn.Sequential(
            ContinuousSincosEmbed(dim=256, ndim=n_conds),
            MLP(
                [256, 256 // 2, 256 // 4, latent_channels],
                act_fn=nn.SiLU,
                last_act_fn=nn.SiLU,
                dropout_prob=0.1,
            ),
        )

        # supernode pooling
        self.encoder = SupernodePooling(
            input_dim=input_dim,
            hidden_dim=upt_base,
            ndim=space,
            radius=supernodes_radius,
            init_weights="truncnormal002",
            max_degree=supernodes_max_neighbours,
        )

        # blocks
        self.blocks = nn.ModuleList(
            [
                DitBlock(
                    dim=upt_base,
                    num_heads=num_heads,
                    cond_dim=latent_channels,
                    drop_path=0,
                )
                for _ in range(app_depth)
            ],
        )

        # decoders
        if dec_depth > 0:
            self.field_decoder = DeepPerceiverDecoderConditioned(
                dim=upt_base,
                num_heads=num_heads,
                input_dim=upt_base,
                output_dim=output_channels,
                ndim=space,
                depth=dec_depth,
                mlp_expansion_factor=mlp_expansion_factor,
                init_weights="truncnormal002",
            )
        else:
            self.field_decoder = None
        self.point_decoder = LinearProjection(
            upt_base, output_channels, init_weights="truncnormal002"
        )

    def _sample_supernodes(self, batch_index: torch.Tensor):
        # nodes per graph
        num_graphs = int(batch_index.max().item()) + 1
        node_counts = torch.bincount(batch_index, minlength=num_graphs)
        cum_counts = torch.cat(
            [torch.tensor([0], device=batch_index.device), node_counts.cumsum(0)]
        )

        supernode_index = []
        for i in range(num_graphs):
            count = node_counts[i].item()
            if count < self.num_supernodes:
                raise ValueError(
                    f"Graph {i} has only {count} nodes, but {self.num_supernodes} \
                        supernodes requested."
                )
            idx = torch.randperm(count, device=batch_index.device)[
                : self.num_supernodes
            ]
            supernode_index.append(idx + cum_counts[i])
        supernode_index = torch.cat(supernode_index)
        super_node_batch_index = torch.arange(
            num_graphs, device=batch_index.device
        ).repeat_interleave(self.num_supernodes)
        return supernode_index, super_node_batch_index

    def forward(
        self,
        cond: torch.Tensor,
        mesh_coords: torch.Tensor,
        mesh_edges: torch.Tensor,
        batch_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for the UPT model with dual decoders.

        Parameters:
            input_position: Sparse tensor (batch_size * num_points, ndim) contianing the
                            input point clouds.
            supernode_idx: Index tensor (batch_size * num_supernodes,) containing the
                           indices of the supernodes.
            output_position: Positions for the field-based decoder
                             (batch_size, num_output_positions, ndim).
                             Not needed if field-based decoder is not used.
            batch_idx: Index tensor that assigns the input_position tensor to its
                       respective sample in the batch.
                       Not needed if batch_size=1.
            unbatch_mask_output_position: Mask to remove padding values from the output
                                          if variable output sizes are produced by the
                                          field-based decoder. Not needed if
                                          batch_size=1 or if the field-based decoder is
                                          not used (dec_depth=0).

        Returns:
              dictionary with the key "point" for the point_based prediction
              (the prediction at the supernode locations) and optionally a second key
              "field" for the field-based prediction if a field-based decoder is used
              (dec_depth > 0).
        """
        coords = mesh_coords.clone()

        # parameter conditioning mlp
        latent_vector = self.conditioning(cond)

        # super node pooling
        supernode_idx, supernode_batch_index = self._sample_supernodes(
            batch_index=batch_index
        )

        # encoder
        x = self.encoder(
            input_pos=mesh_coords,
            supernode_idx=supernode_idx,
            batch_idx=batch_index,
        )

        # blocks
        for block in self.blocks:
            x = block(x, latent_vector)

        # must pad coords for perceiver query
        if mesh_coords.ndim == 2:
            # pad to max nodes
            mesh_coords, pad_mask = to_dense_batch(mesh_coords, batch_index)

        # decoders
        outputs = {}
        # field-based decoder
        if self.field_decoder is not None:
            outputs["field"] = self.field_decoder(
                x=x,
                pos=mesh_coords,
                cond=latent_vector,
                unbatch_mask=pad_mask.flatten(),
            )

        return (outputs["field"], coords), latent_vector
