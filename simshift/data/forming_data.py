"""Forming dataset."""

from dataclasses import dataclass
from typing import Dict, Literal, Optional

import torch

from simshift.data.base_data import BaseDataset, BaseSample


@dataclass
class FormingSample(BaseSample):
    """Forming mesh sample."""

    pass


class FormingDataset(BaseDataset):
    """Forming dataset class."""

    dataset_id = "forming"

    def __init__(
        self,
        path: str = "./datasets/forming",
        difficulty: Literal["easy", "medium", "hard"] = "medium",
        split: Literal["train", "val", "test"] = "train",
        domain: Literal["source", "target"] = "source",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        super().__init__(
            path=path,
            difficulty=difficulty,
            split=split,
            domain=domain,
            dtype=dtype,
            **kwargs,
        )


def get_forming_dataset(
    # path: str,
    split: str,
    normalization_method: Literal["zscore", "minmax"] = "zscore",
    normalization_stats: Optional[Dict] = None,
    **kwargs,
):
    """Return a configured forming dataset by loading it from disk."""
    # source domain
    dataset_source = FormingDataset(split=split, domain="source", **kwargs)

    if split == "train":
        normalization_stats = dataset_source.get_normalization_stats(
            method=normalization_method
        )
    assert normalization_stats is not None
    dataset_source.normalization_stats = normalization_stats
    dataset_source.normalize(method=normalization_method)

    # taget domain
    dataset_target = FormingDataset(split=split, domain="target", **kwargs)

    dataset_target.normalization_stats = normalization_stats
    dataset_target.normalize(method=normalization_method)

    return (dataset_source, dataset_target), normalization_stats
