import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def evaluate_model(model: nn.Module, dataloader: DataLoader, device: torch.device):
    """Evaluates a model on a given dataloader and computes the per-sample RMSE.

    Parameters:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): DataLoader providing batches of samples.
        device (torch.device): Device to perform evaluation on.

    Returns:
        torch.Tensor: Per-sample RMSE losses across the entire dataset.
    """
    model.to(device)
    model.eval()
    losses = []

    with torch.no_grad():
        for sample in tqdm(dataloader, total=len(dataloader), desc="Computing losses"):
            sample.to(device)
            pred, _ = model(**sample.as_dict())
            pred, pred_coords = pred
            mse = (pred - sample.y) ** 2
            loss_per_sample = torch.zeros(
                [sample.batch_index.max().item() + 1, mse.shape[-1]]
            ).to(device)
            loss_per_sample = torch.zeros(
                [sample.batch_index.max().item() + 1, mse.shape[-1]]
            ).to("cpu")
            batch_index_expanded = (
                sample.batch_index.unsqueeze(-1).expand(mse.shape).to("cpu")
            )
            loss_per_sample.scatter_reduce_(
                dim=0, index=batch_index_expanded, src=mse.to("cpu"), reduce="mean"
            )
            loss = loss_per_sample.sqrt().mean(dim=1)
            losses.append(loss.to("cpu"))

    return torch.concat(losses)
