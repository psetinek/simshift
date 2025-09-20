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
        batchnorm: bool = False,
        layernorm: bool = False,
    ):
        super().__init__()
        if isinstance(bias, bool):
            bias = [bias] * (len(latents) - 1)
        dropout = nn.Dropout(dropout_prob)
        mlp = []
        for i, (lat_i, lat_i2) in enumerate(zip(latents, latents[1:], strict=False)):
            mlp.append(nn.Linear(lat_i, lat_i2, bias=bias[i]))
            if i != len(latents) - 2:
                if batchnorm:
                    mlp.append(nn.BatchNorm1d(lat_i2))
                if layernorm:
                    mlp.append(nn.LayerNorm(lat_i2))
                mlp.append(act_fn())
                mlp.append(dropout)
        if last_act_fn is not None:
            mlp.append(last_act_fn())
        self.mlp = nn.Sequential(*mlp)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
