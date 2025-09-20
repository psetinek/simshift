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


def rolling_evaluate_middle_line(preds, gts, coords, batch_indices, x_rel_tol, channel, eps=0.001):
    errors = []
    for i in range(batch_indices.max().item() + 1):
        sample_mask = batch_indices == i
        pred = preds[:, sample_mask, :][:, :, channel]
        gt = gts[:, sample_mask, :][:, :, channel]
        coord = coords[:, sample_mask, :]

        # get center line
        x = coord[0, :, 0]
        # y = coord[0, :, 1]
        x_min, x_max = x.min(), x.max()
        x_center = 0.5 * (x_min + x_max)

        # points mask
        x_tol = (x_max - x_min) * x_rel_tol
        mask = (x - x_center).abs() < x_tol

        # y_line = y[mask]
        f_pred = pred[:, mask, :]
        f_gt = gt[:, mask, :]
        # y_sorted, indices = torch.sort(y_line)
        # f_pred_sorted = f_pred_line[indices]
        # f_gt_sorted = f_gt_line[indices]

        # compute average error
        avg_ae = torch.abs((f_pred - f_gt)/(f_gt + eps)).mean(dim=(1,2))
        errors.append(avg_ae)

    return torch.stack(errors).mean(dim=0)


def forming_evaluate_middle_line(preds, gts, coords, conds, batch_indices, x_rel_tol, dataset, eps=1):
    errors = []
    for i in range(batch_indices.max().item() + 1):
        sample_mask = batch_indices == i
        pred = preds[:, sample_mask, :][:, :, dataset.channels["nodes_stresses"]][:, :, 0].unsqueeze(-1)
        gt = gts[:, sample_mask, :][:, :, dataset.channels["nodes_stresses"]][:, :, 0].unsqueeze(-1)
        coord = coords[:, sample_mask, :]
        cond = conds[i]

        # l/2 coordinate
        x = coord[0, :, 0]
        # y = coord[:, 1]
        l_half = cond[dataset.conds["l"]] / 2
        x_tol = l_half * x_rel_tol
        mask = (x - l_half).abs() < x_tol

        # y_line = y[mask]
        f_pred= pred[:, mask, :]
        f_gt = gt[:, mask, :]
        # y_sorted, indices = torch.sort(y_line)
        # f_pred_sorted = f_pred_line[indices]
        # f_gt_sorted = f_gt_line[indices]

        # compute average error
        avg_ae = torch.abs((f_pred - f_gt)/(f_gt + eps)).mean(dim=(1,2))
        errors.append(avg_ae)

    return torch.stack(errors).mean(dim=0)


def motor_evaluate_chord(preds, gts, coords, conds, batch_indices, x_rel_tol, channel, dataset, eps=1):
    errors = []
    for i in range(batch_indices.max().item() + 1):
        sample_mask = batch_indices == i
        pred = preds[:, sample_mask, :][:, :, channel]
        gt = gts[:, sample_mask, :][:, :, channel]
        coord = coords[:, sample_mask, :]
        cond = conds[i]

        x = coord[0, :, 0]
        y = coord[0, :, 1]

        # find top right point
        y_tol = 1e-1 * y.max()  # adjustable tolerance
        top_mask = (coord[0, :, 1] >= y.max() - y_tol)

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
        # mask = (x - x_cord).abs() < 1e-4


        y_line = y[mask]
        f_pred_line = pred[:, mask, :]
        f_gt_line = gt[:, mask, :]
        y_sorted, indices = torch.sort(y_line)
        f_pred_sorted = f_pred_line[:, indices, :]
        f_gt_sorted = f_gt_line[:, indices, :]

        # start from right, stop at first decrease
        cut_idx = len(f_gt_sorted)  # default: keep all
        for idx in range(len(f_gt_sorted[0, : ,0]) - 1, 0, -1):
            if f_gt_sorted[0, idx - 1, 0] < .7 * f_gt_sorted[0, idx, 0] or y_sorted[idx] < 0.04:
                cut_idx = idx
                break

        # Trim the arrays
        y_sorted = y_sorted[cut_idx:]
        f_pred_sorted = f_pred_sorted[:, cut_idx:, :]
        f_gt_sorted = f_gt_sorted[:, cut_idx:, :]

        avg_ae = torch.abs((f_pred_sorted - f_gt_sorted)/(f_gt_sorted + eps)).mean(dim=(1,2))
        errors.append(avg_ae)

    return torch.stack(errors).mean(dim=0)


def heatsink_evaluate_middle_line(
    preds, gts, coords, batch_indices,
    x_rel_tol, z_rel_tol, z_fixed,
    channel, eps=1
):
    errors = []

    for i in range(batch_indices.max().item() + 1):
        sample_mask = batch_indices == i
        pred = preds[:, sample_mask, :][:, :, channel]
        gt = gts[:, sample_mask, :][:, :, channel]
        coord = coords[:, sample_mask, :]

        x = coord[0, :, 0]
        y = coord[0, :, 1]
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
