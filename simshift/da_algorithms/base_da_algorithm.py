"""SIMSHIFT Domain Adaptation algorithm baseclass."""

from typing import Type, Optional
from abc import ABC, abstractmethod
from collections import defaultdict
from copy import deepcopy

import torch
import torch.nn as nn
from torch import optim
from torch.nn.utils import clip_grad_norm_

from simshift.train.utils import requires_grad, update_ema


class DAAlgorithm(nn.Module, ABC):
    """Abstract base class for SIMSHIFT DA algorithms.

    Wraps the model and handles all parameter updates (gradients and exponential moving
    average), in common across domain adaptation algorithms.

    Args:
        device (torch.device): Device to run computations on (eg cpu/cuda).
        model (nn.Module): Model to perform DA on.
        opt_method (Type): Optimizer class. Defaults to optim.AdamW.
        opt_kwargs (dict, optional): Additional optimizer arguments. Defaults to None.
        clip_grad (bool): Whether to clip gradients. Defaults to False.
        da_loss_weight (float): Weight for domain adaptation loss. Defaults to 0.0.
        use_ema (bool, optional): Whether to use EMA model. Defaults to False.
        ema_decay (float, optional): EMA decay rate. Defaults to None.
        use_amp: (bool, optional): Whether to use AMP. Defaults to False.
    """

    def __init__(
        self,
        device: torch.device,
        model: nn.Module,
        opt_method: Type = optim.AdamW,
        opt_kwargs: Optional[dict] = None,
        clip_grad: bool = False,
        da_loss_weight: float = 0.0,
        use_ema: Optional[bool] = False,
        ema_decay: Optional[float] = None,
        use_amp: Optional[bool] = False,
    ):
        super().__init__()
        # model and loss
        self.device = device
        self.model = model.to(device)
        self.mse_loss = nn.MSELoss()
        self.use_amp = use_amp

        # otpimizer
        opt_kwargs.setdefault("lr", 1e-3)
        self.opt = opt_method(model.parameters(), **opt_kwargs)
        self.scaler = torch.amp.GradScaler(str(device), enabled=self.use_amp)

        self.use_ema = use_ema
        if self.use_ema:
            self.ema_model = deepcopy(model).to(device)
            requires_grad(self.ema_model, False)
            update_ema(self.ema_model, model, decay=0)
            self.ema_model.eval()
            self.ema_decay = ema_decay
        self.clip_grad = clip_grad

        self.da_loss_weight = da_loss_weight
        self.loss = None
        self.loss_dict = defaultdict(int)

    def update(self, src_sample, trgt_samples, **kwargs):
        with torch.autocast(
            device_type=str(self.device), dtype=torch.float16, enabled=self.use_amp
        ):
            self._update(src_sample, trgt_samples, **kwargs)
        assert (
            self.loss is not None
        ), "Please make sure to set self.loss in the DA algorithm class' update \
            function!"
        self.opt.zero_grad()
        self.scaler.scale(self.loss).backward()
        self.scaler.unscale_(self.opt)
        if self.clip_grad:
            params = [p for group in self.opt.param_groups for p in group["params"]]
            clip_grad_norm_(params, 1.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        if self.use_ema:
            update_ema(self.ema_model, self.model, decay=self.ema_decay)

        return self.loss_dict

    @torch.no_grad()
    def predict(self, sample):
        # only return predictions, not latent_vectors
        return (
            self.model(**sample.as_dict())[0]
            if not self.use_ema
            else self.ema_model(**sample.as_dict())[0]
        )

    @abstractmethod
    def _update(self, sample):
        pass
        # subclasses should implement the forward pass + da algorithm and return the
        # epoch train losses to log
