"""
Conditioning modules: continuous sin-cos embeddings and feature-wise modulation layers.
"""


import torch
from einops import rearrange
from torch import nn


class ContinuousSincosEmbed(nn.Module):
    """
    Continuous sin-cos positional embedding for continuous coordinates.

    Args:
        dim (int): Embedding dimension.
        ndim (int): Number of coordinate dimensions.
        max_wavelength (int): Max wavelength for frequency scaling.
    """

    def __init__(
        self, dim: int, ndim: int, max_wavelength: int = 10000, dtype=torch.float32
    ):
        super().__init__()
        self.dim = dim
        self.ndim = ndim
        self.ndim_padding = dim % ndim
        dim_per_ndim = (dim - self.ndim_padding) // ndim
        self.sincos_padding = dim_per_ndim % 2
        self.max_wavelength = max_wavelength
        self.padding = self.ndim_padding + self.sincos_padding * ndim
        effective_dim_per_wave = (self.dim - self.padding) // ndim
        assert effective_dim_per_wave > 0
        self.register_buffer(
            "omega",
            1.0
            / max_wavelength
            ** (
                torch.arange(0, effective_dim_per_wave, 2, dtype=dtype)
                / effective_dim_per_wave
            ),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        out_dtype = coords.dtype
        ndim = coords.shape[-1]
        assert self.ndim == ndim
        out = coords.unsqueeze(-1).to(self.omega.dtype) @ self.omega.unsqueeze(0)
        emb = torch.concat([torch.sin(out), torch.cos(out)], dim=-1)
        emb = rearrange(emb, "... ndim dim -> ... (ndim dim)")
        emb = emb.to(out_dtype)
        if self.padding > 0:
            padding = torch.zeros(
                *emb.shape[:-1], self.padding, device=emb.device, dtype=emb.dtype
            )
            emb = torch.concat([emb, padding], dim=-1)
        return emb


class Film(nn.Module):
    """
    Feature-wise linear modulation (FiLM) layer: scale and shift from conditioning.

    https://arxiv.org/abs/1709.07871

    Args:
        cond_dim (int): Dimension of conditioning vector.
        dim (int): Dimension of input features.
    """

    def __init__(self, cond_dim: int, dim: int):
        super().__init__()

        self.dim_cond = cond_dim
        self.dim = dim
        self.modulation = nn.Linear(cond_dim, dim * 2)  # scale + shift

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        mod = self.modulation(cond)
        # broadcast to x
        scale, shift = mod.reshape(
            mod.shape[0], *(1,) * (x.ndim - cond.ndim), *mod.shape[1:]
        ).chunk(2, dim=-1)
        return x * (scale + 1) + shift


class DiT(nn.Module):
    """
    Gated modulation layer producing multiple scale, shift, and gate parameters.
    
    https://arxiv.org/abs/2212.09748

    Args:
        dim (int): Feature dimension.
        cond_dim (int): Conditioning vector dimension.
        gate_indices (optional): Indices of gates to initialize to zero.
        init_weights (str): Weight initialization method.
        init_gate_zero (bool): Whether to zero-initialize specified gates.
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int,
        gate_indices=None,
        init_weights="xavier_uniform",
        init_gate_zero=False,
    ):
        super().__init__()
        self.dim = dim
        self.cond_dim = cond_dim
        # NOTE: 6 for (scale1, shift1, gate1, scale2, shift2, gate2)
        self.modulation = nn.Linear(cond_dim, 6 * dim)
        self.init_gate_zero = init_gate_zero
        self.gate_indices = gate_indices
        if init_weights is not None:
            self.reset_parameters(init_weights)

    def reset_parameters(self, init_weights):
        if init_weights == "torch":
            pass
        elif init_weights == "xavier_uniform":
            nn.init.xavier_uniform_(self.modulation.weight)
        elif init_weights in ["truncnormal", "truncnormal002"]:
            self.modulation.apply(nn.init.trunc_normal_)
        else:
            raise NotImplementedError

        if self.init_gate_zero:
            assert self.gate_indices is not None
            for gate_index in self.gate_indices:
                start = self.dim * gate_index
                end = self.dim * (gate_index + 1)
                with torch.no_grad():
                    self.modulation.weight[start:end] = 0
                    self.modulation.bias[start:end] = 0

    def forward(self, cond):
        return self.modulation(cond).chunk(6, dim=1)

    @staticmethod
    def modulate_scale_shift(x: torch.Tensor, scale: torch.Tensor, shift: torch.Tensor):
        scale = scale.reshape(
            scale.shape[0], *(1,) * (x.ndim - scale.ndim), *scale.shape[1:]
        )
        shift = shift.reshape(
            shift.shape[0], *(1,) * (x.ndim - shift.ndim), *shift.shape[1:]
        )
        return x * (1 + scale) + shift

    @staticmethod
    def modulate_gate(x: torch.Tensor, gate: torch.Tensor):
        gate = gate.reshape(
            gate.shape[0], *(1,) * (x.ndim - gate.ndim), *gate.shape[1:]
        )
        return gate * x
