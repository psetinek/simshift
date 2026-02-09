from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np
from matplotlib.ticker import FormatStrFormatter


def plot_error_distribution_comparison(
    losses_source, losses_target, bins=30, figsize=None
):
    """
    Plot a histogram comparing source and target per-sample NRMSE losses with relative
    frequencies. Styled for ICML submission.

    Args:
        losses_source (torch.Tensor): Tensor of shape [N] with source domain losses.
        losses_target (torch.Tensor): Tensor of shape [M] with target domain losses.
        bins (int, optional): Number of histogram bins. If None, it will be
            automatically chosen.
        figsize (tuple, optional): Figure size. Defaults to ICML single-column width.

    Returns:
        matplotlib.figure.Figure: The resulting figure object.
    """
    losses_source_np = losses_source.cpu().numpy()
    losses_target_np = losses_target.cpu().numpy()

    cutoff = np.percentile(np.concatenate([losses_source_np, losses_target_np]), 98)

    # force the bins to stop at this cutoff
    bins = np.linspace(0, cutoff, bins)  # fixed number of bins within the useful range

    if bins is None:
        bins = max(len(losses_source_np), len(losses_target_np)) // 20

    if figsize is None:
        figsize = (6, 4)

    fig, ax = plt.subplots(figsize=figsize)

    # colors matching the style from ranges_plots
    c_source = (77 / 255, 156 / 255, 185 / 255)  # blue-ish
    c_target = "#e3c477"  # gold/yellow

    ax.hist(
        losses_source_np,
        bins=bins,
        color=c_source,
        alpha=0.75,
        label='Source',
        edgecolor='black',
        linewidth=0.3,
        weights=np.ones_like(losses_source_np) / len(losses_source_np),
    )
    ax.hist(
        losses_target_np,
        bins=bins,
        color=c_target,
        alpha=0.75,
        label='Target',
        edgecolor='black',
        linewidth=0.3,
        weights=np.ones_like(losses_target_np) / len(losses_target_np),
    )

    ax.set_xlabel("nRMSE per sample")
    ax.set_ylabel("Relative frequency")
    # explicitly set limit just in case
    ax.set_xlim(0, cutoff)

    # clean up spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['bottom'].set_linewidth(0.5)

    # subtle grid
    ax.grid(True, axis="y", linestyle='--', alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # legend
    ax.legend(frameon=False, loc='upper right')

    ax.tick_params(axis='both', which='both', direction='out', length=2, width=0.5)

    fig.tight_layout(pad=0.3)

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
