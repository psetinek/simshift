from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # outputs a probability in [0, 1]
        )

    def forward(self, x):
        return self.layers(x)


@torch.no_grad()
def get_latents(
    model: nn.Module,
    trainloader_source: Dataset,
    valloader_source: Dataset,
    trainloader_target: Dataset,
    device: Optional[str] = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    def predict(model, dataloader):
        model.to(device)
        model.eval()
        latents = []
        for sample in dataloader:
            sample = sample.to(device)
            _, latent = model(**sample.as_dict())
            latents.append(latent)
        return torch.concat(latents, dim=0)

    trainset_source_latents = predict(model, trainloader_source)
    valset_source_latents = predict(model, valloader_source)
    trainset_target_latents = predict(model, trainloader_target)

    return trainset_source_latents, valset_source_latents, trainset_target_latents


def get_mse_loss(
    model: nn.Module, dataloader: DataLoader, device: Optional[str] = "cpu"
) -> torch.Tensor:
    """Compute per sample (per graph) mean squared error (MSE) loss for a model on a
    dataset.

    Args:
        model (nn.Module): Neural network to compute the predictions.
        dataloader (DataLoader): Dataloader for the dataset for which to compute the
                                 MSE on.
        device (str, optional): Torch device name to use.

    Returns:
        Tensor: A 1D tensor of shape `(N,)` containing the MSE per sample,
                where `N == len(dataset)`.
    """
    losses = []
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for sample in dataloader:
            sample = sample.to(device)
            pred, _ = model(**sample.as_dict())
            pred_fields, pred_pos = pred
            pred = torch.cat([pred_fields, pred_pos], dim=-1)
            gt = torch.cat([sample.y, sample.y_mesh_coords], dim=-1)
            loss = F.mse_loss(gt, pred, reduction="none").mean(-1)  # avg mse per node
            loss_per_sample = torch.zeros([sample.batch_index.max().item() + 1]).to(
                device
            )
            loss_per_sample.scatter_reduce_(
                dim=0, index=sample.batch_index, src=loss, reduce="mean"
            )
            losses.append(loss_per_sample.to("cpu"))

    return torch.cat(losses)


def _rmse(pred_fields, gt_fields, pred_pos, gt_pos, batch_index, device):
    _mse_fields = (gt_fields - pred_fields) ** 2
    _mse_per_graph = torch.zeros(
        [batch_index.max().item() + 1, _mse_fields.shape[-1]], device=device
    )
    sample_batch_index_expanded = batch_index.unsqueeze(-1).expand(_mse_fields.shape)
    _mse_per_graph.scatter_reduce_(
        dim=0, index=sample_batch_index_expanded, src=_mse_fields, reduce="mean"
    )
    _rmse_per_field = _mse_per_graph.sqrt()  # [n_batch_samples, n_fields]

    _l2_pos = ((gt_pos - pred_pos) ** 2).sum(dim=-1).sqrt()
    _mean_l2_pos_per_graph = torch.zeros([batch_index.max().item() + 1], device=device)
    _mean_l2_pos_per_graph.scatter_reduce_(
        dim=0, index=batch_index, src=_l2_pos, reduce="mean"
    )

    return _rmse_per_field, _mean_l2_pos_per_graph


def compute_rmse(
    model: nn.Module,
    dataloader: DataLoader,
    compute_denormalized: bool = False,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    model = model.to(device)
    model.eval()
    rmse_fields = []
    l2_coordinates = []
    if compute_denormalized:
        rmse_fields_denormalized = []
        l2_coordinates_denormalized = []
    with torch.no_grad():
        for sample in dataloader:
            sample = sample.to(device)
            gt_fields = sample.y
            gt_pos = sample.y_mesh_coords
            pred, _ = model(**sample.as_dict())
            pred_fields, pred_pos = pred

            # normalized errors
            _rmse_per_field, _mean_l2_pos_per_graph = _rmse(
                pred_fields, gt_fields, pred_pos, gt_pos, sample.batch_index, device
            )
            rmse_fields.append(_rmse_per_field.to("cpu"))
            l2_coordinates.append(_mean_l2_pos_per_graph.to("cpu"))

            if compute_denormalized:
                # denormalize predictions
                pred_fields = dataloader.dataset.denormalize(
                    None, pred_fields.to("cpu")
                )
                pred_fields = pred_fields.to(device)
                pred_pos = dataloader.dataset.denormalize_coords(pred_pos.to("cpu"))
                pred_pos = pred_pos.to(device)
                # denormalize gt
                gt_fields = dataloader.dataset.denormalize(None, gt_fields.to("cpu"))
                gt_fields = gt_fields.to(device)
                gt_pos = dataloader.dataset.denormalize_coords(gt_pos.to("cpu"))
                gt_pos = gt_pos.to(device)

                # denormalized errors
                _rmse_per_field, _mean_l2_pos_per_graph = _rmse(
                    pred_fields, gt_fields, pred_pos, gt_pos, sample.batch_index, device
                )
                rmse_fields_denormalized.append(_rmse_per_field.to("cpu"))
                l2_coordinates_denormalized.append(_mean_l2_pos_per_graph.to("cpu"))

    rmse_fields = torch.cat(rmse_fields, dim=0)
    l2_coordinates = torch.cat(l2_coordinates, dim=0)

    if compute_denormalized:
        rmse_fields_denormalized = torch.cat(rmse_fields_denormalized, dim=0)
        l2_coordinates_denormalized = torch.cat(l2_coordinates_denormalized, dim=0)
        return (
            rmse_fields,
            l2_coordinates,
            rmse_fields_denormalized,
            l2_coordinates_denormalized,
        )
    else:
        return rmse_fields, l2_coordinates


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute model predictions for the dataset in the dataloder.

    Args:
        model (nn.Module): Neural network to compute predictions with.
        dataloader (DataLoader): Dataloader for the dataset for which to compute
                                 predictions.
        device (str, optional): Torch device name to use.

    Returns:
        tuple[Tensor, Tensor]:
            - `predictions` (torch.Tensor): concatenated `[pred_fields, pred_pos]`
              of shape `(total_nodes, n_fields + coord_dim)`.
            - `batch_index` (torch.Tensor): tensor of shape `(total_nodes,)`, indicating
                                            which sample each node belongs to.
    """
    model = model.to(device)
    model.eval()
    preds = []
    indices = []
    with torch.no_grad():
        curr_max_sample = 0
        for sample in dataloader:
            sample = sample.to(device)
            pred, _ = model(**sample.as_dict())
            pred_fields, pred_pos = pred
            pred = torch.cat([pred_fields, pred_pos], dim=-1)
            preds.append(pred)
            indices.append(sample.batch_index + curr_max_sample)
            curr_max_sample += sample.batch_index.max() + 1
    predictions = torch.cat(preds, dim=0)  # (total_nodes, feat+coord)
    batch_index = torch.cat(indices, dim=0)  # (total_nodes,)
    return predictions, batch_index


def train_domain_classifier(
    trainset_source_latents,
    trainset_target_latents,
    num_epochs=100,
    lr=1e-3,
    device="cuda",
):
    combined_latents = torch.cat(
        [trainset_source_latents, trainset_target_latents], dim=0
    ).to(device)
    ys = torch.cat(
        [
            torch.ones(trainset_source_latents.size(0), 1),
            torch.zeros(trainset_target_latents.size(0), 1),
        ],
        dim=0,
    ).to(device)

    combined_np = combined_latents.detach().cpu().numpy()
    ys_np = ys.detach().cpu().numpy()

    feat_train, feat_val, label_train, label_val = train_test_split(
        combined_np, ys_np, train_size=0.8, stratify=ys_np, random_state=42
    )

    decays = [1e-2, 1e-3, 1e-4, 1e-5]

    best_val_acc = -1
    best_model = None

    feat_train = torch.tensor(feat_train, dtype=combined_latents.dtype).to(device)
    label_train = torch.tensor(label_train, dtype=ys.dtype).to(device)
    feat_val = torch.tensor(feat_val, dtype=combined_latents.dtype).to(device)
    label_val = torch.tensor(label_val, dtype=ys.dtype).to(device)

    input_dim = combined_latents.shape[-1]

    for decay in decays:
        model = LogisticRegressionModel(input_dim=input_dim)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        criterion = nn.BCELoss()

        model = model.to(device)
        model.train()
        for _ in range(num_epochs):
            opt.zero_grad()
            preds = model(feat_train)
            loss = criterion(preds, label_train)
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_preds = model(feat_val)
            pred_labels = (val_preds >= 0.5).float()
        val_acc = (pred_labels.squeeze() == label_val.squeeze()).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model = model

    return best_model


def rolling_evaluate_middle_line(
    preds, gts, coords, batch_indices, x_rel_tol, channel, eps=0.001
):
    errors = []
    for i in range(batch_indices.max().item() + 1):
        sample_mask = batch_indices == i
        pred = preds[:, sample_mask, :][:, :, channel]
        gt = gts[:, sample_mask, :][:, :, channel]
        coord = coords[:, sample_mask, :]

        # get center line
        x = coord[0, :, 0]
        x_min, x_max = x.min(), x.max()
        x_center = 0.5 * (x_min + x_max)

        # points mask
        x_tol = (x_max - x_min) * x_rel_tol
        mask = (x - x_center).abs() < x_tol

        f_pred = pred[:, mask, :]
        f_gt = gt[:, mask, :]

        # compute average error
        avg_ae = torch.abs((f_pred - f_gt) / (f_gt + eps)).mean(dim=(1, 2))
        errors.append(avg_ae)

    return torch.stack(errors).mean(dim=0)


def forming_evaluate_middle_line(
    preds, gts, coords, conds, batch_indices, x_rel_tol, dataset, eps=1
):
    errors = []
    for i in range(batch_indices.max().item() + 1):
        sample_mask = batch_indices == i
        pred = preds[:, sample_mask, :][:, :, dataset.channels["s"]][:, :, 0].unsqueeze(
            -1
        )
        gt = gts[:, sample_mask, :][:, :, dataset.channels["s"]][:, :, 0].unsqueeze(-1)
        coord = coords[:, sample_mask, :]
        cond = conds[i]

        # l/2 coordinate
        x = coord[0, :, 0]
        # y = coord[:, 1]
        l_half = cond[dataset.conds["l"]] / 2
        x_tol = l_half * x_rel_tol
        mask = (x - l_half).abs() < x_tol

        f_pred = pred[:, mask, :]
        f_gt = gt[:, mask, :]

        # compute average error
        avg_ae = torch.abs((f_pred - f_gt) / (f_gt + eps)).mean(dim=(1, 2))
        errors.append(avg_ae)

    return torch.stack(errors).mean(dim=0)


def motor_evaluate_chord(
    preds, gts, coords, conds, batch_indices, x_rel_tol, channel, dataset, eps=1
):
    errors = []
    for i in range(batch_indices.max().item() + 1):
        sample_mask = batch_indices == i
        pred = preds[:, sample_mask, :][:, :, channel].unsqueeze(-1)
        gt = gts[:, sample_mask, :][:, :, channel].unsqueeze(-1)
        coord = coords[:, sample_mask, :]
        cond = conds[i]

        x = coord[0, :, 0]
        y = coord[0, :, 1]

        # find top right point
        y_tol = 1e-1 * y.max()  # adjustable tolerance
        top_mask = coord[0, :, 1] >= y.max() - y_tol

        # Among those, find the one with largest x
        x_top = coord[0, top_mask, 0]
        idx_top_right = torch.argmax(x_top)
        top_right_point = coord[:, top_mask, :][:, idx_top_right, :]

        # shift to the left
        trsb1 = cond[dataset.conds["Geometry.Rotor.trsb1"]]
        rr2 = cond[dataset.conds["Geometry.Rotor.rr2"]]
        shift = (trsb1 / 2 + 1.1 * rr2) * 1e-3
        x_cord = top_right_point[:, 0] - shift
        mask = (x - x_cord).abs() < shift * x_rel_tol

        y_line = y[mask]
        f_pred_line = pred[:, mask, :]
        f_gt_line = gt[:, mask, :]
        y_sorted, indices = torch.sort(y_line)
        f_pred_sorted = f_pred_line[:, indices, :]
        f_gt_sorted = f_gt_line[:, indices, :]

        # start from right, stop at first decrease
        cut_idx = len(f_gt_sorted)  # default: keep all
        for idx in range(len(f_gt_sorted[0, :, 0]) - 1, 0, -1):
            if (
                f_gt_sorted[0, idx - 1, 0] < 0.7 * f_gt_sorted[0, idx, 0]
                or y_sorted[idx] < 0.04
            ):
                cut_idx = idx
                break

        # Trim the arrays
        y_sorted = y_sorted[cut_idx:]
        f_pred_sorted = f_pred_sorted[:, cut_idx:, :]
        f_gt_sorted = f_gt_sorted[:, cut_idx:, :]

        avg_ae = torch.abs((f_pred_sorted - f_gt_sorted) / (f_gt_sorted + eps)).mean(
            dim=(1, 2)
        )
        errors.append(avg_ae)

    return torch.stack(errors).mean(dim=0)


def motor_constitutive_error(
    logarithmic_strain, cauchy_stress, mesh_material, batch_indices, eps=1e-6
):
    """
    Compute constitutive error for motor dataset using linear elastic Hooke's law.

    Args:
        logarithmic_strain: (n_algorithms, n_nodes, 4) with components [xx, xy, yy, zz]
        cauchy_stress: (n_algorithms, n_nodes, 4) with components [xx, xy, yy, zz]
        mesh_material: (n_nodes,) material IDs (1 = rotor core, 2 = magnet)
        batch_indices: (n_nodes,) sample indices for graph batching
        eps: small value to avoid division by zero

    Returns:
        (n_algorithms,) tensor of relative constitutive errors averaged over samples
    """
    # component indices
    XX, XY, YY, ZZ = 0, 1, 2, 3

    # material properties: rotor core 1 and magnet 2
    E_1 = 185.0e9
    nu_1 = 0.29
    C1 = E_1 / (1 - nu_1**2)
    G_1 = E_1 / (2 * (1 + nu_1))

    E_2 = 175.0e9
    nu_2 = 0.24
    factor_2 = E_2 / ((1 + nu_2) * (1 - 2 * nu_2))
    G_2 = E_2 / (2 * (1 + nu_2))

    num_samples = int(batch_indices.max().item()) + 1

    sample_errors = []

    for i in range(num_samples):
        sample_mask = batch_indices == i
        if not sample_mask.any():
            continue

        # extract strain and stress for this sample: (n_algorithms, n_nodes_i, 4)
        strain = logarithmic_strain[:, sample_mask, :]
        sigma_gt = cauchy_stress[:, sample_mask, :]
        mat_ids = mesh_material[sample_mask].squeeze()  # (n_nodes_i,)

        # initialize predicted stress
        sigma_pred = torch.zeros_like(sigma_gt)

        mask1 = mat_ids == 0  # rotor core (plane stress)
        mask2 = mat_ids == 1  # magnet (plane strain)

        # rotor core (plane stress)
        if mask1.any():
            eps_xx = strain[:, mask1, XX]
            eps_xy = strain[:, mask1, XY]
            eps_yy = strain[:, mask1, YY]

            # sigma_xx = C * (eps_xx + nu * eps_yy)
            sigma_pred[:, mask1, XX] = C1 * (eps_xx + nu_1 * eps_yy)
            # sigma_yy = C * (eps_yy + nu * eps_xx)
            sigma_pred[:, mask1, YY] = C1 * (eps_yy + nu_1 * eps_xx)
            # sigma_xy = 2 * G * eps_xy
            sigma_pred[:, mask1, XY] = 2 * G_1 * eps_xy
            # sigma_zz = 0 for plane stress
            sigma_pred[:, mask1, ZZ] = 0.0

        # magnet (plane strain)
        if mask2.any():
            eps_xx = strain[:, mask2, XX]
            eps_xy = strain[:, mask2, XY]
            eps_yy = strain[:, mask2, YY]

            # sigma_xx = factor * ((1-nu)*eps_xx + nu*eps_yy)
            sigma_pred[:, mask2, XX] = factor_2 * ((1 - nu_2) * eps_xx + nu_2 * eps_yy)
            # sigma_yy = factor * ((1-nu)*eps_yy + nu*eps_xx)
            sigma_pred[:, mask2, YY] = factor_2 * ((1 - nu_2) * eps_yy + nu_2 * eps_xx)
            # sigma_xy = 2 * G * eps_xy
            sigma_pred[:, mask2, XY] = 2 * G_2 * eps_xy
            # sigma_zz = nu * (sigma_xx + sigma_yy) for plane strain
            sigma_pred[:, mask2, ZZ] = nu_2 * (
                sigma_pred[:, mask2, XX] + sigma_pred[:, mask2, YY]
            )

        # Compute squared difference
        diff_sq = (sigma_pred - sigma_gt) ** 2
        # Sum over all nodes and all components for the sample
        # dimensions: (n_algorithms, n_nodes_i, 4) -> sum over (1, 2)
        numerator = torch.sqrt(diff_sq.sum(dim=(1, 2)))

        # Compute squared magnitude of ground truth
        gt_sq = sigma_gt**2
        denominator = torch.sqrt(gt_sq.sum(dim=(1, 2))) + eps

        rel_error = numerator / denominator

        sample_errors.append(rel_error)

    # Average over samples
    return torch.stack(sample_errors, dim=0).mean(dim=0)  # (n_algorithms,)


def heatsink_evaluate_middle_line(
    preds, gts, coords, batch_indices, x_rel_tol, z_rel_tol, z_fixed, channel, eps=1
):
    errors = []

    for i in range(batch_indices.max().item() + 1):
        sample_mask = batch_indices == i
        pred = preds[:, sample_mask, :][:, :, channel]
        gt = gts[:, sample_mask, :][:, :, channel]
        coord = coords[:, sample_mask, :]

        x = coord[0, :, 0]
        z = coord[0, :, 2]

        # Get center x value
        x_min, x_max = x.min(), x.max()
        x_center = 0.5 * (x_min + x_max)
        x_tol = (x_max - x_min) * x_rel_tol
        z_tol = (z.max() - z.min()) * z_rel_tol

        # Mask for middle line: x ~ center and z ~ fixed
        mask = ((x - x_center).abs() < x_tol) & ((z - z_fixed).abs() < z_tol)

        if mask.sum() == 0:
            print(f"Warning: no points for sample {i} on center line at z={z_fixed}")
            continue

        f_pred = pred[:, mask, :]
        f_gt = gt[:, mask, :]

        # Compute average error
        avg_ae = torch.abs((f_pred - f_gt) / (f_gt + eps)).mean(dim=(1, 2))
        errors.append(avg_ae)

    return torch.stack(errors).mean(dim=0)


def _r2_score(y_pred, y_true):
    y_true_mean = y_true.mean()
    ss_tot = ((y_true - y_true_mean) ** 2).sum()
    ss_res = ((y_true - y_pred) ** 2).sum()
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def calculate_tta_metrics(y_pred, y, batch_index=None):
    # check for nan
    if torch.isnan(y_pred).any():
        print("y_pred is nan")
    if torch.isnan(y).any():
        print("y is nan")

    y_true = y.float().flatten()
    y_pred_flat = y_pred.float().flatten()
    mae = F.l1_loss(y_pred_flat, y_true).item()
    r2 = _r2_score(y_pred_flat, y_true)

    metrics = {
        "mae": mae,
        "r2": r2.item(),
    }
    return metrics


def mises_stress_consistency(stresses_pred, mises_pred, batch_indices, eps=1e-6):
    metrics = []
    for i in range(batch_indices.max().item() + 1):
        sample_mask = batch_indices == i
        # Assume [S11, S22, S33, S12]
        S11 = stresses_pred[:, sample_mask, 0]
        S22 = stresses_pred[:, sample_mask, 1]
        S33 = stresses_pred[:, sample_mask, 2]
        S12 = stresses_pred[:, sample_mask, 3]

        # Calculate von Mises from components
        # (S13 and S23 are 0)
        term1 = (S11 - S22) ** 2
        term2 = (S22 - S33) ** 2
        term3 = (S33 - S11) ** 2
        term4 = 6 * S12**2  # shape will be [n_models, n_nodes]

        # Note: .squeeze() to match shapes
        mises_recalc = torch.sqrt(0.5 * (term1 + term2 + term3 + term4))
        mises_pred_squeezed = mises_pred[:, sample_mask, 0]

        # Check the difference
        difference = (torch.abs(mises_pred_squeezed - mises_recalc)).mean(dim=-1)

        mean_true_stress = (torch.abs(mises_pred_squeezed)).mean(dim=-1) + eps
        metric = difference / mean_true_stress
        metrics.append(metric)
    return torch.stack(metrics).mean(dim=0)


def mises_stress_consistency_motor(stresses_pred, mises_pred, batch_indices, eps=1e-6):
    metrics = []
    for i in range(batch_indices.max().item() + 1):
        sample_mask = batch_indices == i
        # Assume [Sxx, Sxy, Syx, Syy, Szz]
        S11 = stresses_pred[:, sample_mask, 0]
        S12 = stresses_pred[:, sample_mask, 1]
        S21 = stresses_pred[:, sample_mask, 2]
        S22 = stresses_pred[:, sample_mask, 3]
        S33 = stresses_pred[:, sample_mask, 4]

        # assert torch.allclose(S12, S21)
        S12 = (S12 + S21) * 0.5

        # Calculate von Mises from components
        # (S13 and S23 are 0)
        term1 = (S11 - S22) ** 2
        term2 = (S22 - S33) ** 2
        term3 = (S33 - S11) ** 2
        term4 = 6 * S12**2

        # Note: .squeeze() to match shapes
        mises_recalc = torch.sqrt(0.5 * (term1 + term2 + term3 + term4)).squeeze()
        mises_pred_squeezed = mises_pred[:, sample_mask, 0]

        # Check the difference
        difference = (torch.abs(mises_pred_squeezed - mises_recalc)).mean(dim=-1)

        mean_true_stress = (torch.abs(mises_pred_squeezed)).mean(dim=-1) + eps
        metric = difference / mean_true_stress
        metrics.append(metric)
    return torch.stack(metrics).mean(dim=0)


def yield_metrics(
    vm: torch.Tensor,
    peeq: torch.Tensor,
    sigma_y0: float,
    H: float,
    batch_indices: torch.Tensor,
    peeq_tol: float = 1e-6,
):
    """
    vm, peeq: shape [M, N_total] or [M, N_total, 1]
        M = number of models
        N_total = total nodes across all samples
    batch_indices: shape [N_total], int tensor with sample IDs in [0, S-1]
    Returns:
        elastic_frac_over_0_mean : shape [M]
        plastic_residual_mean_rel_mean : shape [M]
        (each averaged over all samples in the batch)
    """

    # Make sure vm and peeq are [M, N_total]
    vm = vm.squeeze(-1)
    peeq = peeq.squeeze(-1)

    M, N_total = vm.shape
    num_samples = int(batch_indices.max().item()) + 1

    elastic_frac_over_0_list = []
    plastic_residual_mean_rel_list = []

    for i in range(num_samples):
        sample_mask = batch_indices == i  # [N_total]
        if not sample_mask.any():
            # no nodes for this sample id
            continue

        # Select nodes belonging to this sample -> [M, N_i]
        vm_masked = vm[:, sample_mask]
        peeq_masked = peeq[:, sample_mask]

        # -----------------------------
        # 1) Elastic region: PEEQ ~ 0
        # -----------------------------
        elastic_mask = peeq_masked < peeq_tol  # [M, N_i]
        plastic_mask = ~elastic_mask  # [M, N_i]

        # elastic: fraction of points with σ_vm > σ_y0
        elastic_counts = elastic_mask.sum(dim=1)  # [M]
        overstress = vm_masked - sigma_y0  # [M, N_i]
        viol_elastic = (overstress > 0) & elastic_mask  # [M, N_i]
        num_viol = viol_elastic.sum(dim=1)  # [M]

        # avoid division by zero
        elastic_frac_over_0 = torch.where(
            elastic_counts > 0,
            num_viol.float() / elastic_counts.clamp(min=1).float(),
            torch.zeros_like(elastic_counts, dtype=torch.float),
        )  # [M]

        elastic_frac_over_0_list.append(elastic_frac_over_0)

        # -----------------------------------------
        # 2) Plastic region: σ_vm ≈ σ_y0 + H * PEEQ
        # -----------------------------------------
        plastic_counts = plastic_mask.sum(dim=1)  # [M]

        sigma_yield = sigma_y0 + H * peeq_masked  # [M, N_i]
        resid = vm_masked - sigma_yield  # [M, N_i]
        rel_resid = resid.abs() / sigma_y0  # [M, N_i]

        # Masked average over plastic nodes
        sum_rel_resid = (rel_resid * plastic_mask).sum(dim=1)  # [M]
        plastic_residual_mean_rel = torch.where(
            plastic_counts > 0,
            sum_rel_resid / plastic_counts.clamp(min=1).float(),
            torch.zeros_like(plastic_counts, dtype=torch.float),
        )  # [M]

        plastic_residual_mean_rel_list.append(plastic_residual_mean_rel)

    if len(elastic_frac_over_0_list) == 0:
        # no samples / nodes – return zeros
        M = vm.shape[0]
        return (
            torch.zeros(M, device=vm.device),
            torch.zeros(M, device=vm.device),
        )

    # Stack over samples and average across them: shape [num_samples, M] -> [M]
    elastic_frac_over_0_mean = torch.stack(elastic_frac_over_0_list, dim=0).mean(dim=0)
    plastic_residual_mean_rel_mean = torch.stack(
        plastic_residual_mean_rel_list, dim=0
    ).mean(dim=0)

    return elastic_frac_over_0_mean, plastic_residual_mean_rel_mean


def fin_BC_consistency(gt, pred, cond, dataset, batch_indices, temp_tol=1e-6):
    channels = dataset.channels
    cond_channels = dataset.conds
    gt = gt.squeeze(0)

    temp_errs = []
    vel_errs = []
    for i in range(batch_indices.max().item() + 1):
        sample_mask = batch_indices == i

        # scalar temperatures from conditioning
        t_solid = cond[i, cond_channels["solidTemp"]].item()
        t_env = 300
        dT = t_solid - t_env

        # absolute tolerance around T_solid
        tol_abs = abs(dT) * temp_tol

        # extract temperature fields
        T_gt = gt[sample_mask, channels["T"]]
        T_pred = pred[:, sample_mask, channels["T"]]

        # identify fin nodes from GT
        fin_mask = (T_gt - t_solid).abs() < tol_abs
        fin_mask = fin_mask.squeeze()
        n_fin = int(fin_mask.sum().item())
        if n_fin == 0:
            continue

        # temperature BC violation (Dirichlet T = T_solid) ---
        temp_err_pred = (
            ((T_pred[:, fin_mask] - t_solid).abs() / dT).squeeze(-1).mean(dim=-1)
        )

        # velocity BC violation (no-slip U = 0) ---
        U_pred = pred[:, sample_mask, channels["U"]]  # (N, 3)

        vel_norm_pred = U_pred[:, fin_mask, :].norm(dim=-1).mean(dim=-1)

        temp_errs.append(temp_err_pred)
        vel_errs.append(vel_norm_pred)

    return torch.stack(temp_errs).mean(dim=0), torch.stack(vel_errs).mean(dim=0)
