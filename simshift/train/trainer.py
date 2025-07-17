from collections import defaultdict
from itertools import cycle
from typing import Any, List

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch import optim
from torch.utils.data import DataLoader, Dataset

from simshift.eval import Metrics
from simshift.utils import Logger, save_model_and_cfg


class Trainer:
    def __init__(
        self,
        datasets: List[Dataset],
        dataloaders: List[DataLoader],
        da_algorithm: nn.Module,
        device: torch.device,
        scheduler: Any = None,
        n_epochs: int = 5000,
        early_stopping_patience: int = 10,
        metrics: Metrics = None,
        logger: Logger = None,
        cfg: DictConfig = None,
    ):
        if metrics is None:
            metrics = Metrics()
        self.datasets = datasets
        (
            trainloader_source,
            valloader_source,
            trainloader_target,
            valloader_target,
        ) = dataloaders
        # recreate iterator
        if len(trainloader_target) < len(trainloader_source):
            self.trainloader = lambda: zip(
                trainloader_source, cycle(trainloader_target)
            )
        else:
            self.trainloader = lambda: zip(
                trainloader_source, trainloader_target, strict=False
            )
        self.len_dataloader = len(trainloader_source)
        self.valloader_source = valloader_source
        self.valloader_target = valloader_target

        self.da_algorithm = da_algorithm
        if scheduler == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.da_algorithm.opt, T_max=n_epochs
            )
        else:
            self.scheduler = None

        self.device = device
        self.n_epochs = n_epochs
        self.early_stopping_patience = early_stopping_patience
        self.metrics = metrics
        assert logger is not None
        self.logger = logger
        self.cfg = cfg

    def run(self):
        val_loss_min = torch.inf
        early_stop_counter = 0
        for epoch in range(1, self.n_epochs + 1):
            train_loss_dict = self.train_epoch(epoch)
            if (epoch % 10) == 0:
                val_loss_dict = self.eval_epoch()

                # construct loss dicts to log
                train_loss_dict_log = {}
                for k, v in train_loss_dict.items():
                    train_loss_dict_log["train/" + k] = v

                val_loss_dict_source = val_loss_dict["source"]
                val_loss_dict_log = {}
                for metric_name in self.metrics.metric_names:
                    val_loss_dict_log[
                        "val_source/" + metric_name
                    ] = val_loss_dict_source[metric_name]

                # log metrics
                loss_dict = train_loss_dict_log | val_loss_dict_log
                self.logger.log(epoch, log_dict=loss_dict)

                # save model and log summary statistics
                val_loss = val_loss_dict_log["val_source/mse_avg"]
                if val_loss < val_loss_min:
                    early_stop_counter = 0
                    # logging summary stats
                    if self.logger.wandb_writer is not None:
                        self.logger.wandb_writer.summary[
                            "model_saving/val_source/mse_avg"
                        ] = val_loss_dict_log["val_source/mse_avg"]
                        self.logger.wandb_writer.summary[
                            "model_saving/val_source/mse_max"
                        ] = val_loss_dict_log["val_source/mse_max"]
                        self.logger.wandb_writer.summary["model_saving/epoch"] = epoch
                    # save model checkpoint and config (only if config is available,
                    # e.g. dont save in tutorial notebook)
                    if self.cfg:
                        save_model_and_cfg(
                            self.da_algorithm.ema_model
                            if self.da_algorithm.use_ema
                            else self.da_algorithm.model,
                            optimizer=self.da_algorithm.opt,
                            cfg=self.cfg,
                            trainset=self.datasets[0],
                            epoch=epoch,
                            val_loss=val_loss,
                            loss_val_min=val_loss_min,
                        )
                    val_loss_min = val_loss
                else:
                    early_stop_counter += 10
                    if early_stop_counter >= self.early_stopping_patience:
                        print(
                            f"Early stopping triggered at epoch {epoch} after \
                                {self.early_stopping_patience} evaluations without \
                                    improvement."
                        )
                        break

    def train_epoch(self, epoch):
        self.da_algorithm.train()
        train_loss_dict = defaultdict(int)
        sum_bs = 0
        for step, sample in enumerate(self.trainloader()):
            sample = tuple(s.to(self.device) for s in sample)
            # p and alpha for grl weight scheduling (for DANN)
            p = ((epoch - 1) * self.len_dataloader + step + 1) / (
                self.len_dataloader * self.n_epochs
            )
            alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
            self.da_algorithm.update(*sample, p=p, alpha=alpha)
            bs = sample[0].cond.shape[0]
            sum_bs += bs
            for k, v in self.da_algorithm.loss_dict.items():
                train_loss_dict[k] += v * bs
        for k, val in train_loss_dict.items():
            train_loss_dict[k] = val / sum_bs

        if self.scheduler is not None:
            self.scheduler.step()
            lr = self.scheduler.get_last_lr()[0]
            train_loss_dict["lr"] = lr

        return train_loss_dict

    @torch.no_grad()
    def eval_epoch(self):
        self.da_algorithm.eval()
        self.metrics.reset_epoch()
        for sample in self.valloader_source:
            sample = sample.to(self.device)
            pred = self.da_algorithm.predict(sample)
            pred, coords = pred
            self.metrics.update_domain_metrics(pred, sample.y, domain="source")
        for sample in self.valloader_target:
            sample = sample.to(self.device)
            pred = self.da_algorithm.predict(sample)
            pred, coords = pred
            self.metrics.update_domain_metrics(pred, sample.y, domain="target")
        epoch_stats = self.metrics.get_epoch_stats()
        return epoch_stats
