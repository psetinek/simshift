import torch
from omegaconf import DictConfig, OmegaConf
from torch import optim

import wandb
from simshift.da_algorithms import get_da_algorithm
from simshift.data import get_data
from simshift.eval import get_metrics
from simshift.models import get_model
from simshift.train import Trainer
from simshift.utils import Logger


def run(cfg: DictConfig):
    n_epochs = cfg.training.n_epochs

    # set up logging
    wandb_run = None
    if cfg.logging.writer == "wandb":
        wandb_run = wandb.init(
            project=cfg.logging.wandb_project,
            entity=cfg.logging.wandb_entity,
            name=cfg.logging.run_id,
            save_code=True,
            config=OmegaConf.to_container(cfg),
        )
    logger = Logger(__name__, n_epochs=n_epochs, wandb_writer=wandb_run)

    datasets, dataloaders = get_data(cfg)

    model = get_model(cfg, dataset=datasets[0])
    print(
        f"Model parameters: {(sum(p.numel() for p in model.parameters()) / 1e6):.2f}M"
    )

    DAAlgorithm = get_da_algorithm(cfg.da_algorithm.name)
    da_algorithm = DAAlgorithm(
        device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
        model=model,
        opt_method=optim.AdamW,
        opt_kwargs={"lr": cfg.training.lr, "weight_decay": cfg.training.weight_decay},
        clip_grad=cfg.training.gradient_clipping,
        da_loss_weight=cfg.da_algorithm.da_loss_weight,
        use_ema=cfg.training.use_ema,
        ema_decay=cfg.training.ema_decay,
        use_amp=cfg.training.use_amp,
        **cfg.da_algorithm.kwargs
        if cfg.da_algorithm.get("kwargs", None) is not None
        else {},
    )

    metrics = get_metrics(cfg)

    trainer = Trainer(
        datasets=datasets,
        dataloaders=dataloaders,
        da_algorithm=da_algorithm,
        device=torch.device("cuda") if torch.cuda.is_available() else "cpu",
        scheduler=cfg.training.scheduler,
        n_epochs=cfg.training.n_epochs,
        early_stopping_patience=cfg.training.early_stopping_patience,
        metrics=metrics,
        logger=logger,
        cfg=cfg,
    )
    trainer.run()

    # close wandb logger
    if cfg.logging.writer == "wandb":
        wandb.finish()
