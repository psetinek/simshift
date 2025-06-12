from omegaconf import OmegaConf
from torch import nn

from .mesh import UPT, GraphSAGE, PointNet, Transolver
from .registry import get_model_class

__all__ = ["get_model_class", "PointNet", "GraphSAGE", "Transolver", "UPT"]


def get_model(cfg, dataset) -> nn.Module:
    Model = get_model_class(cfg.model.name)

    hparams = OmegaConf.to_container(cfg.model.hparams)

    return Model(
        n_conds=dataset.n_conds,
        output_channels=dataset.n_channels,
        n_materials=getattr(dataset, "n_materials", None),
        out_deformation=cfg.dataset.out_deformation,
        space=cfg.dataset.space,
        **hparams,
    )
