"""
Geometry-Informed Neural Operator for Large-Scale 3D PDEs.

https://arxiv.org/abs/2309.00583
"""
from time import perf_counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuralop.models import GINO

from simshift.models.registry import register_model
from simshift.models.condition import ContinuousSincosEmbed
from simshift.models.utils import MLP


@register_model()
class ConditionedGINO(nn.Module):
    def __init__(
        self,
        n_conds: int,
        latent_channels: int = 8,
        output_channels: int = 17,
        act_fn: nn.Module = nn.SiLU,
        dropout_prob: float = 0.1,
        space: int = 2,
        num_layers: int = 5,  # this will be fno_n_layers
        out_deformation: bool = True,
        n_materials: int = None,
        conditioning_bn: bool = False,
        conditioning_ln: bool = False,
        # --- GINO-specific parameters ---
        fno_n_modes: tuple = (16, 16, 16),
        fno_hidden_channels: int = 16,
        fno_lifting_channel_ratio: int = 2,
        in_gno_radius: float = 0.1,
        out_gno_radius: float = 0.1,
        gno_use_open3d: bool = True, # Open3D often fails with 2D
    ):
        super().__init__()
        self.space = space
        self.output_channels = output_channels
        self.out_deformation = out_deformation
        self.latent_channels = latent_channels

        # --- Sanity checks and param mapping ---
        if len(fno_n_modes) != space:
            print(f"Warning: fno_n_modes length ({len(fno_n_modes)}) does not match space ({space}). Adjusting...")
            fno_n_modes = fno_n_modes[:space]
        
        # GINO's output layer must predict deformation if enabled
        gino_out_channels = output_channels
        if out_deformation:
            gino_out_channels += space

        # gno_coord_dim must match space
        gno_coord_dim = space

        # --- 1. Conditioning Module ---
        self.conditioning = nn.Sequential(
            ContinuousSincosEmbed(dim=256, ndim=n_conds),
            MLP(
                [256, 256 // 2, 256 // 4, latent_channels],
                act_fn=act_fn,
                last_act_fn=act_fn, # Ensure last layer has activation
                dropout_prob=dropout_prob,
                batchnorm=conditioning_bn,
                layernorm=conditioning_ln,
            ),
        )

        # --- 2. GINO Model ---
        self.gino = GINO(
            in_channels=1,  # We will feed a (B=1, N, 1) tensor of ones
            out_channels=gino_out_channels,
            latent_feature_channels=latent_channels, # From conditioning
            projection_channel_ratio=4,
            gno_coord_dim=gno_coord_dim,
            in_gno_radius=in_gno_radius,
            out_gno_radius=out_gno_radius,
            in_gno_transform_type="linear",
            out_gno_transform_type="linear",
            fno_in_channels=3, # Can be tuned
            fno_n_modes=fno_n_modes,
            fno_hidden_channels=fno_hidden_channels,
            fno_lifting_channel_ratio=fno_lifting_channel_ratio,
            fno_n_layers=num_layers, # Use num_layers from signature
            gno_use_open3d=False,  # HERE
            # gno_use_torch_scatter=True,
            fno_non_linearity=F.gelu,
            fno_skip="linear",
            fno_channel_mlp_skip="soft-gating",
            # Add other GINO params as needed
            gno_embed_channels=16,
            gno_embed_max_positions=10000,
            in_gno_channel_mlp_hidden_layers=[40, 40, 40],
            out_gno_channel_mlp_hidden_layers=[256, 128],
            gno_channel_mlp_non_linearity=F.gelu,
        )

        # --- 3. Latent Grid (for FNO) ---
        # Create the structured grid that GINO expects for 'latent_queries'
        grid_nodes = []
        for n_modes in fno_n_modes:
            # Use n_modes to define the grid resolution
            grid_nodes.append(torch.linspace(0, 1, n_modes))
        
        grid = torch.stack(torch.meshgrid(*grid_nodes, indexing="ij"), dim=-1)
        # Register as buffer, add the B=1 dimension
        self.register_buffer('latent_queries', grid.unsqueeze(0)) # Shape (1, G1, G2, D)

    def forward(
        self,
        cond: torch.Tensor,
        mesh_coords: torch.Tensor,
        mesh_edges: torch.Tensor,
        mesh_material=None,
        batch_index=None,
    ):
        # 0. Store original coords for deformation
        coords = mesh_coords.clone()

        # Your dataloader (B=1) provides:
        # cond: (1, n_conds)
        # mesh_coords: (N, D)
        # batch_index: (N,) -- all zeros
        
        # 1. Get the conditioning vector
        # latent_vector shape: (1, C_latent)
        latent_vector = self.conditioning(cond)

        # 2. Prepare GINO inputs (add B=1 dimension)
        
        # input_geom: (1, N, D)
        input_geom = mesh_coords.unsqueeze(0)
        
        # We query the output at the same points
        # output_queries: (1, N, D)
        output_queries = input_geom
        
        # Artificial input features: (1, N, 1)
        x_features = torch.ones(
            1, mesh_coords.shape[0], 1, device=mesh_coords.device
        )
        
        # Broadcast conditioning to the latent grid
        # latent_queries shape: (1, G1, G2, D)
        # We need latent_features: (1, G1, G2, C_latent)
        grid_shape = self.latent_queries.shape[1:-1] # (G1, G2)
        latent_features = latent_vector.view(1, *([1] * len(grid_shape)), -1)
        latent_features = latent_features.expand(
            1, *grid_shape, self.latent_channels
        )
    
        # start = perf_counter()
        # 3. Call GINO (B=1)
        # out shape: (1, N, gino_out_channels)
        try:
            out = self.gino(
                input_geom=input_geom,
                latent_queries=self.latent_queries,
                output_queries=output_queries,
                x=x_features,
                latent_features=latent_features,
                ada_in=None,
            )
        except RuntimeError as err:
            shapes = {
                "cond": tuple(cond.shape),
                "x": tuple(x_features.shape),
                "input_geom": tuple(input_geom.shape),
                "output_queries": tuple(output_queries.shape),
                "latent_features": tuple(latent_features.shape),
                "latent_queries": tuple(self.latent_queries.shape),
                "mesh_coords": tuple(mesh_coords.shape),
                "mesh_edges": None if mesh_edges is None else tuple(mesh_edges.shape),
                "batch_index": (
                    None if batch_index is None else tuple(batch_index.shape)
                ),
            }
            shapes_msg = "GINO input mismatch " + " | ".join(
                f"{name}={value}" for name, value in shapes.items()
            )
            raise RuntimeError(f"{shapes_msg} | cause={err}") from err        
        # t = perf_counter() - start
        # print(f"Time taken for gino pass search: {t:.5f}")
        
        # 4. Squeeze the batch dimension for the loss function
        # x shape: (N, gino_out_channels)
        x = out.squeeze(0)

        # 5. Handle deformation and return
        if self.out_deformation:
            x, dpos = x.split([self.output_channels, self.space], -1)
            coords = coords + dpos

        # Return in the format your DAAlgorithm expects:
        # 1. (predictions, coords)
        # 2. latent_vector (B, C_latent) for CORAL loss
        return (x, coords), latent_vector
