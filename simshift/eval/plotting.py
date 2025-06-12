from copy import deepcopy
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter


def plot_error_distribution(
    losses_source: torch.Tensor, losses_target: torch.Tensor, bins: Optional[int] = None
) -> matplotlib.figure.Figure:
    """Plot a histogram comparing per-sample NRMSE losses for source and target domains,
    using relative frequencies for each group.

    Parameters:
        losses_source (torch.Tensor): 1D tensor of source domain per-sample losses.
        losses_target (torch.Tensor): 1D tensor of target domain per-sample losses.
        bins (int, optional): Number of bins for the histograms. If None, a default is
                              chosen.

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the plot.
    """
    losses_source_np = losses_source.cpu().numpy()
    losses_target_np = losses_target.cpu().numpy()

    if bins is None:
        bins = max(len(losses_source_np), len(losses_target_np)) // 20

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(
        losses_source_np,
        bins=15,
        color="#219ebc",
        alpha=0.7,
        label='Source',
        edgecolor='black',
        weights=np.ones_like(losses_source_np) / len(losses_source_np),
    )
    ax.hist(
        losses_target_np,
        bins=bins,
        color="#e9c46a",
        alpha=0.7,
        label='Target',
        edgecolor='black',
        weights=np.ones_like(losses_target_np) / len(losses_target_np),
    )

    ax.set_xlabel("NRMSE per sample", fontsize=14)
    ax.set_ylabel("Relative frequency", fontsize=14)
    ax.set_title("Error distribution", fontsize=16)
    ax.grid(True, axis="y", linestyle='--', alpha=0.5)
    ax.legend()
    ax.tick_params(axis='both', labelsize=12)
    fig.tight_layout()

    return fig


def plot_rolling(
    final_mesh_coords: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    error: np.ndarray,
    channels: dict,
    aspect_ratio: float = 5.0,
    cmap: str = "plasma",
) -> matplotlib.figure.Figure:
    """Plot ground truth, prediction, and absolute error fields over a 2D mesh for each
       channel.

    Parameters:
        final_mesh_coords (np.ndarray): Array of shape [num_nodes, 2] with 2D mesh node
                                        coordinates.
        gt (np.ndarray): Array of shape [num_nodes, num_channels] with ground truth
                         values.
        pred (np.ndarray): Array of shape [num_nodes, num_channels] with predicted
                           values.
        error (np.ndarray): Array of shape [num_nodes, num_channels] with absolute
                            errors.
        channels (dict): Mapping of channel names to column indices or slices in the
                         data arrays.
        aspect_ratio (float, optional): Aspect ratio for each subplot. Default is 5.0.
        cmap (str, optional): Matplotlib colormap name for contour plots. Default is
                              "plasma".

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the
                                  multi-channel plots.
    """
    num_channels = len(channels)
    fig, axes = plt.subplots(num_channels, 3, figsize=(15, 5 * num_channels))
    plt.subplots_adjust(hspace=0.4, wspace=0.3, right=0.85)

    # Create base triangulation
    triang = tri.Triangulation(final_mesh_coords[:, 0], final_mesh_coords[:, 1])

    titles = ["Ground truth", "Prediction", "Absolute error"]

    for row_idx, (channel_name, channel_slice) in enumerate(channels.items()):
        # Compute mean values per node
        gt_data = gt[:, channel_slice].mean(axis=1)
        pred_data = pred[:, channel_slice].mean(axis=1)
        error_data = error[:, channel_slice].mean(axis=1)

        # Shared levels for GT and prediction
        vmin = min(gt_data.min(), pred_data.min())
        vmax = max(gt_data.max(), pred_data.max())
        levels_main = np.linspace(vmin, vmax, 10)

        levels_error = np.linspace(error_data.min(), error_data.max(), 10)

        for col_idx, (data, levels) in enumerate(
            zip(
                [gt_data, pred_data, error_data],
                [levels_main, levels_main, levels_error],
                strict=False,
            )
        ):
            ax = axes[row_idx, col_idx] if num_channels > 1 else axes[col_idx]

            contour = ax.tricontourf(triang, data, levels=levels, cmap=cmap, alpha=1.0)

            ax.set_axis_off()
            ax.set_title(titles[col_idx], fontsize=16)
            ax.set_aspect(aspect_ratio)

            # Colorbar with label
            cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label(f"{channel_name}", fontsize=14)
            cbar.ax.tick_params(labelsize=12)
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    fig.tight_layout()
    return fig


def plot_forming(
    triangles_final: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    error: np.ndarray,
    channels: dict,
    cmap: str = "plasma",
) -> matplotlib.figure.Figure:
    """Plot ground truth, prediction, and absolute error fields over a triangulated 2D
       mesh for each channel.

    Parameters:
        triangles_final (np.ndarray): Triangulation object or array for the 2D mesh
                                      (e.g., from matplotlib.tri).
        gt (np.ndarray): Array of shape [num_nodes, num_channels] with ground truth
                         values.
        pred (np.ndarray): Array of shape [num_nodes, num_channels] with predicted
                           values.
        error (np.ndarray): Array of shape [num_nodes, num_channels] with absolute
                            errors.
        channels (dict): Mapping of channel names to column indices or slices in the
                         data arrays.
        aspect_ratio (float, optional): Aspect ratio for each subplot. Default is 2.0.
        cmap (str, optional): Matplotlib colormap name for contour plots. Default is
                              "plasma".

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the
        multi-channel plots.
    """
    num_channels = len(channels)
    fig, axes = plt.subplots(
        3, num_channels, figsize=(10, 8)
    )  # Note: switched dimensions
    plt.subplots_adjust(hspace=0.4, wspace=0.3, right=0.85)

    titles = ["Ground truth", "Prediction", "Absolute error"]

    for col_idx, (channel_name, channel_slice) in enumerate(channels.items()):
        # Compute mean values per node
        gt_data = gt[:, channel_slice].mean(axis=1)
        pred_data = pred[:, channel_slice].mean(axis=1)
        error_data = error[:, channel_slice].mean(axis=1)

        # Shared levels for GT and prediction
        vmin = min(gt_data.min(), pred_data.min())
        vmax = max(gt_data.max(), pred_data.max())
        levels_main = np.linspace(vmin, vmax, 10)
        levels_error = np.linspace(error_data.min(), error_data.max(), 10)

        for row_idx, (data, levels, title) in enumerate(
            zip(
                [gt_data, pred_data, error_data],
                [levels_main, levels_main, levels_error],
                titles,
                strict=False,
            )
        ):
            ax = axes[row_idx, col_idx] if num_channels > 1 else axes[row_idx]

            contour = ax.tricontourf(
                triangles_final, data, levels=levels, cmap=cmap, alpha=1.0
            )

            ax.set_axis_off()
            ax.set_title(f"{title} - {channel_name}", fontsize=14)

            # Set consistent limits and aspect ratio
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            ax.set_xlim(xmin, xmax)
            aspect = (xmax - xmin) / (ymax - ymin) / 6
            ax.set_ylim(-1, ymax + 5)
            ax.set_aspect(aspect)

            # Colorbar with label
            cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label(f"{channel_name}", fontsize=14)
            cbar.ax.tick_params(labelsize=12)
            cbar.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

    fig.tight_layout()
    return fig


def plot_motor(
    final_mesh_coords: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    error: np.ndarray,
    material_field: np.ndarray,
    channels: dict,
    cmap: str = "plasma",
) -> matplotlib.figure.Figure:
    """Plot ground truth, prediction, and absolute error fields over a 2D mesh for each
    channel, with material- and density-based masking applied for visualization of
    motor regions.

    Parameters:
        final_mesh_coords (np.ndarray): Array of shape [num_nodes, 2] with 2D mesh node
                                        coordinates.
        gt (np.ndarray): Array of shape [num_nodes, num_channels] with ground truth
                         values.
        pred (np.ndarray): Array of shape [num_nodes, num_channels] with predicted
                           values.
        error (np.ndarray): Array of shape [num_nodes, num_channels] with absolute
                            errors.
        material_field (np.ndarray): Array indicating material presence per node, used
                                     for masking.
        channels (dict): Mapping of channel names to column indices or slices in the
                         data arrays.
        cmap (str, optional): Matplotlib colormap name for contour plots. Default is
                              "plasma".

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the
                                  multi-channel plots.
    """
    num_channels = len(channels)
    fig, axes = plt.subplots(num_channels, 3, figsize=(15, 5 * num_channels))
    plt.subplots_adjust(hspace=0.4, wspace=0.3, right=0.85)

    # Create base triangulation
    triang = tri.Triangulation(final_mesh_coords[:, 0], final_mesh_coords[:, 1])
    tri_indices = triang.triangles

    # Precompute masks (same for all subplots)
    material_mask = np.any(material_field[tri_indices] == 0, axis=1).squeeze()

    tri_coords = final_mesh_coords[tri_indices]
    vec1 = tri_coords[:, 1] - tri_coords[:, 0]
    vec2 = tri_coords[:, 2] - tri_coords[:, 0]
    area = 0.5 * np.abs(vec1[:, 0] * vec2[:, 1] - vec1[:, 1] * vec2[:, 0])
    density_mask = (area > 2e-7) | (~material_mask)

    titles = ["Ground truth", "Prediction", "Absolute error"]

    for row_idx, (channel_name, channel_slice) in enumerate(channels.items()):
        # Compute mean values per node
        gt_data = gt[:, channel_slice].mean(axis=1)
        pred_data = pred[:, channel_slice].mean(axis=1)
        error_data = error[:, channel_slice].mean(axis=1)

        # Shared levels for GT and prediction
        vmin = min(gt_data.min(), pred_data.min())
        vmax = max(gt_data.max(), pred_data.max())
        levels_main = np.linspace(vmin, vmax, 10)

        levels_error = np.linspace(error_data.min(), error_data.max(), 10)

        for col_idx, (data, levels) in enumerate(
            zip(
                [gt_data, pred_data, error_data],
                [levels_main, levels_main, levels_error],
                strict=False,
            )
        ):
            ax = axes[row_idx, col_idx] if num_channels > 1 else axes[col_idx]

            # Copy and mask triangulations
            rotor_triang = deepcopy(triang)
            magnet_triang = deepcopy(triang)

            rotor_triang.set_mask(density_mask)
            magnet_triang.set_mask(material_mask)

            contour = ax.tricontourf(
                rotor_triang, data, levels=levels, cmap=cmap, alpha=1.0
            )
            ax.tricontourf(magnet_triang, data, levels=levels, cmap=cmap, alpha=0.2)

            ax.set_axis_off()
            ax.set_title(titles[col_idx], fontsize=16)

            # Colorbar with label
            cbar = fig.colorbar(contour, ax=ax, orientation='vertical', pad=0.05)
            cbar.set_label(f"{channel_name}", fontsize=14)
            cbar.ax.tick_params(labelsize=12)

            ax.set_aspect("equal")

    fig.tight_layout()
    return fig


def plot_heatsink(
    final_mesh_coords: np.ndarray,
    gt: np.ndarray,
    pred: np.ndarray,
    error: np.ndarray,
    channels: dict,
    cmap: str = "plasma",
) -> matplotlib.figure.Figure:
    """Plot ground truth, prediction, and absolute error fields for a 3D heatsink mesh
    for each channel, visualized as scatter plots for each data type.

    Parameters:
        final_mesh_coords (np.ndarray): Array of shape [num_nodes, 3] with 3D mesh node
                                        coordinates.
        gt (np.ndarray): Array of shape [num_nodes, num_channels] with ground truth
                         values.
        pred (np.ndarray): Array of shape [num_nodes, num_channels] with predicted
                           values.
        error (np.ndarray): Array of shape [num_nodes, num_channels] with absolute
                            errors.
        channels (dict): Mapping of channel names to column indices or slices in the
                         data arrays.
        cmap (str, optional): Matplotlib colormap name for scatter plots. Default is
                              "plasma".

    Returns:
        matplotlib.figure.Figure: The matplotlib Figure object containing the
                                  multi-channel 3D scatter plots.
    """
    num_channels = len(channels)
    fig = plt.figure(figsize=(16, 5 * num_channels))
    titles = ["Ground truth", "Prediction", "Absolute error"]

    mask = final_mesh_coords[:, 0] > 0.07
    final_mesh_coords = final_mesh_coords[mask]
    gt = gt[mask]
    pred = pred[mask]
    error = error[mask]

    for row_idx, (channel_name, channel_slice) in enumerate(channels.items()):
        # Compute scalar field per node
        gt_data = gt[:, channel_slice].mean(axis=1)
        pred_data = pred[:, channel_slice].mean(axis=1)
        error_data = error[:, channel_slice].mean(axis=1)

        # Shared levels for GT and prediction
        vmin = min(gt_data.min(), pred_data.min())
        vmax = max(gt_data.max(), pred_data.max())
        vmin_err = error_data.min()
        vmax_err = error_data.max()

        # Coordinates
        x, y, z = (
            final_mesh_coords[:, 0],
            final_mesh_coords[:, 1],
            final_mesh_coords[:, 2],
        )

        for col_idx, (data, title, vmin_i, vmax_i) in enumerate(
            zip(
                [gt_data, pred_data, error_data],
                titles,
                [vmin, vmin, vmin_err],
                [vmax, vmax, vmax_err],
                strict=False,
            )
        ):
            # if col_idx > 0:
            #     break
            ax = fig.add_subplot(
                num_channels, 3, row_idx * 3 + col_idx + 1, projection='3d'
            )
            p = ax.scatter(
                x, y, z, c=data, cmap=cmap, vmin=vmin_i, vmax=vmax_i, s=1, marker="o"
            )
            ax.view_init(elev=20, azim=-165)  # Key modification
            range_x = final_mesh_coords[:, 0].max() - final_mesh_coords[:, 0].min()
            range_y = final_mesh_coords[:, 1].max() - final_mesh_coords[:, 1].min()
            range_z = final_mesh_coords[:, 2].max() - final_mesh_coords[:, 2].min()

            # Set box aspect using actual data proportions
            ax.set_box_aspect([range_x, range_y, range_z])
            # ax.set_box_aspect([1, 1, 1])  # Equal scaling in x, y, z

            ax.set_title(f"{title}", fontsize=16)
            ax.grid(False)
            ax.axis("off")
            cbar = fig.colorbar(p, ax=ax, shrink=0.6, pad=0.1, format="%.2e")
            cbar.set_label(f"{channel_name}", fontsize=14)
            cbar.ax.tick_params(labelsize=14)

    fig.tight_layout()
    return fig
