from typing import Optional, Sequence, Type, Union

import torch
from einops import rearrange
from torch import nn
from torch.nn import functional as F


class NonLearnableLayerNorm(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        return (x - mean) / (std + self.eps)


class MLP(nn.Module):
    def __init__(
        self,
        latents: Sequence[int],
        act_fn: nn.Module = nn.GELU,
        last_act_fn: Optional[nn.Module] = None,
        bias: Union[bool, Sequence[bool]] = True,
        dropout_prob: float = 0.0,
    ):
        super().__init__()
        if isinstance(bias, bool):
            bias = [bias] * (len(latents) - 1)
        dropout = nn.Dropout(dropout_prob)
        mlp = []
        for i, (lat_i, lat_i2) in enumerate(zip(latents, latents[1:], strict=False)):
            mlp.append(nn.Linear(lat_i, lat_i2, bias=bias[i]))
            if i != len(latents) - 2:
                mlp.append(act_fn())
                mlp.append(dropout)
        if last_act_fn is not None:
            mlp.append(last_act_fn())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)


class AttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 2.0,
        qkv_bias: bool = True,
        dropout_prob: float = 0.0,
        attn_drop: float = 0.0,
        norm_layer: Type[nn.Module] = nn.LayerNorm,
        use_checkpoint: bool = False,
        act_fn: nn.Module = nn.GELU,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.attn_drop = attn_drop
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_prob)

        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim), act_fn(), nn.Linear(mlp_hidden_dim, dim)
        )
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def attn(self, x):
        b, h, w, c = x.shape
        x = x.flatten(1, -2)
        qkv = rearrange(
            self.qkv(x),
            "b n (three heads c) -> three b heads n c",
            three=3,
            heads=self.num_heads,
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        x = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop)

        # attention readout
        x = rearrange(x, "b k n c -> b n (k c)")
        x = self.proj(x)
        x = self.proj_drop(x)
        # back to original shape
        x = x.view(b, h, w, c)
        return x

    def forward(self, x):
        skip = x
        x = self.norm1(x)
        x = self.attn(x)
        x = self.norm2(self.mlp(x))
        return skip + x
