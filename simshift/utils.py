import logging
import os
import random
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch import optim
from torch.utils.data import Dataset

from simshift.data import get_data
from simshift.models import get_model


def set_seed(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set as {seed}")


class Logger(logging.Logger):
    def __init__(
        self, name: str, n_epochs: int = 0, wandb_writer: Optional[Callable] = None
    ):
        super().__init__(name, level=logging.INFO)

        self.n_epochs = n_epochs
        self.wandb_writer = wandb_writer

        if not self.hasHandlers():
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s - %(message)s")
            console_handler.setFormatter(formatter)
            self.addHandler(console_handler)

    def log(
        self,
        epoch: int,
        log_dict: Dict[str, Any],
        plot_dict: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs,
    ):
        del plot_dict
        if self.wandb_writer is not None:
            # log to wandb
            self.wandb_writer.log(log_dict, step=epoch)

        e_s = str(epoch).zfill(len(str(self.n_epochs)))
        msg = f"[{e_s}] "
        for i, (k, v) in enumerate(log_dict.items()):
            if v is not None and v != float("-inf"):
                msg += f"{k}: {v:.5f}"
                if i != len(log_dict):
                    msg += ", "

        super().info(msg, *args, **kwargs)


def save_model_and_cfg(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    cfg: DictConfig,
    trainset: Dataset,
    epoch: int,
    val_loss: float,
    loss_val_min: float,
):
    """Save the current model, optimizer state, config, and training statistics to disk.

    This function creates (if necessary) the output directory based on the config, and
    saves the model checkpoint
    ("ckp.pth") at every call. If the current validation loss is lower than the previous
    minimum, it also updates
    the best checkpoint ("best.pth").

    Parameters:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        cfg (DictConfig): Configuration object containing output paths and metadata.
        trainset (Dataset): The training dataset, used to save normalization statistics.
        epoch (int): The current epoch number.
        val_loss (float): The current validation loss.
        loss_val_min (float): The lowest validation loss achieved so far.
    """
    # create directory if it s not there
    output_path = os.path.join(cfg.output_path, cfg.logging.run_id)
    os.makedirs(output_path, exist_ok=True)
    cfg_dict = OmegaConf.to_object(cfg)
    torch.save(
        {
            "cfg": cfg_dict,
            "dataset_stats": trainset.normalization_stats,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": val_loss,
        },
        f"{output_path}/ckp.pth",
    )

    if val_loss < loss_val_min:
        torch.save(
            {
                "cfg": cfg_dict,
                "dataset_stats": trainset.normalization_stats,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": val_loss,
            },
            f"{output_path}/best.pth",
        )


def load_model(ckp_path, load_opt, valset=None, load_trainset=False):
    """Load a model and optionally related datasets, dataloaders, and optimizer state
    from a checkpoint.

    Parameters:
        ckp_path (str): Path to the checkpoint file (.pth) to load.
        load_opt (bool): If True, also load the optimizer state into the returned
                         dictionary.
        valset (optional): Validation dataset to use for initializing the model.
                           If None, datasets and dataloaders are loaded from config.
        load_trainset (bool, optional): If True, also load and return train datasets and
                                        dataloaders; otherwise only validation datasets
                                        and loaders.

    Returns:
        dict: Dictionary containing at least the loaded model and config, and optionally
              datasets, dataloaders, and optimizer depending on arguments:
            - "cfg": Loaded config object.
            - "model": Loaded model with weights restored.
            - "opt": Loaded optimizer (if load_opt is True).
            - "trainset_source", "valset_source", "trainset_target", "valset_target":
              Loaded datasets (as available).
            - "trainloader_source", "valloader_source", "trainloader_target",
              "valloader_target": Corresponding dataloaders (as available).
    """
    dict_out = {}
    ckp = torch.load(ckp_path, weights_only=True)
    cfg = OmegaConf.create(ckp["cfg"])
    dict_out["cfg"] = cfg

    if valset is None:
        datasets, dataloaders = get_data(
            cfg, val_only=not load_trainset, normalization_stats=ckp["dataset_stats"]
        )
        if load_trainset:
            (
                trainset_source,
                valset_source,
                trainset_target,
                valset_target,
            ) = datasets
            (
                trainloader_source,
                valloader_source,
                trainloader_target,
                valloader_target,
            ) = dataloaders
            # datasets
            dict_out["trainset_source"] = trainset_source
            dict_out["valset_source"] = valset_source
            dict_out["trainset_target"] = trainset_target
            dict_out["valset_target"] = valset_target
            # dataloaders
            dict_out["trainloader_source"] = trainloader_source
            dict_out["valloader_source"] = valloader_source
            dict_out["trainloader_target"] = trainloader_target
            dict_out["valloader_target"] = valloader_target
        else:
            (
                valset_source,
                valset_target,
            ) = datasets
            (
                valloader_source,
                valloader_target,
            ) = dataloaders
            # datasets
            dict_out["valset_source"] = valset_source
            dict_out["valset_target"] = valset_target
            # dataloaders
            dict_out["valloader_source"] = valloader_source
            dict_out["valloader_target"] = valloader_target

    # model loading
    if valset is None:
        valset = datasets[1]
    model = get_model(cfg, dataset=valset)
    model.load_state_dict(ckp["model_state_dict"])
    dict_out["model"] = model

    # optimizer loading
    if load_opt:
        opt = optim.Adam(
            model.parameters(),
            lr=cfg.training.lr,
            weight_decay=cfg.training.weight_decay,
        )
        opt.load_state_dict(ckp["optimizer_state_dict"])
        dict_out["opt"] = opt

    return dict_out
