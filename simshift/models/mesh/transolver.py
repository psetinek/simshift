"""
Transolver: A Fast Transformer Solver for PDEs on General Geometries

Adapted from https://github.com/thuml/Transolver
"""

from functools import partial
from typing import Optional

import torch
import torch.nn as nn
from einops import einsum, rearrange
from torch.nn import functional as F
from torch_geometric.utils import to_dense_batch

from simshift.models.condition import ContinuousSincosEmbed, DiT
from simshift.models.registry import register_model
from simshift.models.utils import MLP


class TransolverAttention(nn.Module):
    """
    Multi-head self-attention with physics-aware slicing for PDE solvers.

    :param dim: Input feature dimension.
    :type dim: int
    :param num_heads: Number of attention heads.
    :type num_heads: int
    :param dropout_prob: Dropout probability after attention.
    :type dropout_prob: float
    :param attn_dropout_prob: Dropout within attention weights.
    :type attn_dropout_prob: float
    :param slice_base: Base number of slices for token grouping.
    :type slice_base: int
    """


    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.0,
        slice_base: int = 64,
    ):
        super().__init__()

        assert (dim % num_heads) == 0
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.num_heads = num_heads
        self.slice_base = slice_base
        self.attn_dropout_prob = attn_dropout_prob

        self.temperature = nn.Parameter(torch.ones([1, num_heads, 1, 1]) * 0.5)
        # input projection
        self.x_proj = nn.Linear(dim, dim)
        self.fx_proj = nn.Linear(dim, dim)
        self.slice_proj = nn.Linear(self.head_dim, slice_base)
        nn.init.orthogonal_(self.slice_proj.weight)
        # qkv projection
        self.qkv = nn.Linear(self.head_dim, self.head_dim * 3, bias=False)
        self.readout = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout_prob))

    def forward(self, x: torch.Tensor):
        # slices
        x_mid = rearrange(self.x_proj(x), "b n (h c) -> b h n c", c=self.head_dim)
        fx_mid = rearrange(self.fx_proj(x), "b n (h c) -> b h n c", c=self.head_dim)
        slice_weights = F.softmax(
            self.slice_proj(x_mid) / self.temperature, -1
        )  # b h n g
        # in-slice attention
        scale = (slice_weights.sum(2) + 1e-5)[..., None].repeat(1, 1, 1, self.head_dim)
        slice_att = einsum(fx_mid, slice_weights, "b h n c, b h n g -> b h g c") / scale
        # global (across slices) attention
        qkv = rearrange(self.qkv(slice_att), "b h g (thr c) -> thr b h g c", thr=3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_dropout_prob)
        # merge (cross attention)
        x = einsum(att, slice_weights, "b h g c, b h n g -> b h n c")
        x = rearrange(x, "b h n d -> b n (h d)")
        return self.readout(x)


class TransolverBlock(nn.Module):
    def __init__(
        self,
        dim: int = 128,
        num_heads: int = 4,
        dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.0,
        act_fn: nn.Module = nn.SiLU,
        mlp_ratio: float = 4.0,
        slice_base: int = 64,
        norm_layer: nn.Module = nn.LayerNorm,
    ):
        super().__init__()

        self.dim = dim

        self.norm1 = norm_layer(dim)
        self.attn = TransolverAttention(
            dim=dim,
            num_heads=num_heads,
            dropout_prob=dropout_prob,
            attn_dropout_prob=attn_dropout_prob,
            slice_base=slice_base,
        )

        # TODO drop path like transformers?
        self.norm2 = norm_layer(dim)
        self.mlp = MLP([dim, int(dim * mlp_ratio), dim], act_fn=act_fn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ViT-like structure
        # attention
        x = x + self.attn(self.norm1(x))
        # mlp
        x = x + self.mlp(self.norm2(x))
        return x


class DiTransolverBlock(TransolverBlock):
    def __init__(self, cond_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dit = DiT(self.dim, cond_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x = super().forward(x)
        scale1, shift1, gate1, scale2, shift2, gate2 = self.dit(cond)
        # modulated attention
        x1 = self.dit.modulate_scale_shift(self.norm1(x), scale1, shift1)
        x2 = self.dit.modulate_gate(self.attn(x1), gate1) + x
        # modulated mlp
        x3 = self.dit.modulate_scale_shift(self.norm2(x2), scale2, shift2)
        x4 = self.dit.modulate_gate(self.mlp(x3), gate2) + x2
        return x4


@register_model()
class Transolver(nn.Module):
    """
    Transolver: A Transformer-based PDE solver, with Physics-Attention to model physical
    states and efficiently capture complex geometries.

    Args:
        n_conds (int): Number of conditioning inputs.
        latent_channels (int): Channels in the latent representation.
        output_channels (int): Number of output channels.
        act_fn (nn.Module): Activation function to use.
        dropout_prob (float): Dropout probability.
        attn_dropout_prob (float): Dropout for attention layers.
        space (int): Spatial dimensionality (e.g., 2D or 3D).
        transolver_base (int): Base feature size for the transformer.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of transformer layers.
        slice_base (int): Base number of tokens for slicing.
        mlp_ratio (float): Ratio for MLP hidden dimension.
        conditioning_mode (str): Type of conditioning ('dit', etc.).
        out_deformation (bool): Whether to output deformation fields.
        n_materials (Optional[int]): Number of materials (if applicable).

    Paper: https://arxiv.org/abs/2402.02366
    """

    def __init__(
        self,
        n_conds: int,
        latent_channels: int = 8,
        output_channels: int = 17,
        act_fn: nn.Module = nn.SiLU,
        dropout_prob: float = 0.1,
        attn_dropout_prob: float = 0.1,
        space: int = 2,
        transolver_base: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        slice_base: int = 64,
        mlp_ratio: float = 2.0,
        conditioning_mode: str = "dit",
        out_deformation: bool = True,
        n_materials: Optional[int] = None,
    ):
        super().__init__()

        self.space = space
        self.output_channels = output_channels

        assert (transolver_base % num_heads) == 0
        assert conditioning_mode in ["cat", "dit"]
        self.conditioning_mode = conditioning_mode
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
        self.coord_embed = ContinuousSincosEmbed(dim=transolver_base, ndim=space)
        self.encoder = MLP([transolver_base, transolver_base], act_fn=act_fn)

        # material embedding
        if n_materials is not None:
            self.material_embedding = nn.Embedding(
                num_embeddings=n_materials, embedding_dim=transolver_base
            )

        # processor ("physics attention")
        BlockType = TransolverBlock
        if conditioning_mode == "cat":
            # TODO project down instead
            self.proj_cond = nn.Linear(
                transolver_base + latent_channels, transolver_base, bias=False
            )
        if conditioning_mode == "dit":
            BlockType = partial(DiTransolverBlock, latent_channels)

        blocks = []
        for _ in range(num_layers):
            block = BlockType(
                dim=transolver_base,
                num_heads=num_heads,
                dropout_prob=dropout_prob,
                attn_dropout_prob=attn_dropout_prob,
                act_fn=act_fn,
                mlp_ratio=mlp_ratio,
                slice_base=slice_base,
            )
            blocks.append(block)
        self.blocks = nn.ModuleList(blocks)

        # TODO: why did we have this here???
        # self.bias = nn.Parameter(
        #     (1 / (transolver_base)) * torch.rand(transolver_base, dtype=torch.float)
        # )

        # decode latent to fields (+ positions)
        self.decoder = MLP(
            [transolver_base, output_channels + (space if out_deformation else 0)],
            act_fn,
            dropout_prob=dropout_prob,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        # NOTE reproduce original initialization?
        if isinstance(m, nn.Linear):
            # from timm.models.layers import trunc_normal_
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(
        self,
        cond: torch.Tensor,
        mesh_coords: torch.Tensor,
        mesh_edges: torch.Tensor,
        mesh_material: Optional[torch.Tensor] = None,
        batch_index: Optional[torch.Tensor] = None,
    ):
        _ = mesh_edges

        # NOTE: transolver expects inputs of shape (B, N, d)
        pad_mask = None
        coords = mesh_coords.clone()

        # pad to max nodes if in sparse format
        if mesh_coords.ndim == 2:
            mesh_coords, pad_mask = to_dense_batch(mesh_coords, batch_index)

        # conditioning
        latent_vector = self.conditioning(cond)

        # encoder
        x = self.encoder(self.coord_embed(mesh_coords))  # (B, N, C)

        if mesh_material is not None:
            # add material embedding if we have it
            mesh_material, _ = to_dense_batch(mesh_material.squeeze(), batch_index)
            mesh_material_embedding = self.material_embedding(mesh_material.squeeze())
            x += mesh_material_embedding

        if self.conditioning_mode == "cat":
            z = latent_vector[batch_index]
            if pad_mask is not None:
                z, _ = to_dense_batch(z, batch_index)
            x = self.proj_cond(torch.cat([x, z], dim=-1))
            cond = {}
        if self.conditioning_mode == "dit":
            cond = {"cond": latent_vector}

        # transolver
        for block in self.blocks:
            x = block(x, **cond)

        # decoder
        x = self.decoder(x)

        # unpad
        if pad_mask is not None:
            x = x[pad_mask]

        # dynamic mesh
        if self.out_deformation:
            x, dpos = x.split([self.output_channels, self.space], -1)
            coords = coords + dpos

        return (x, coords), latent_vector
