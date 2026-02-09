"""Electric motor dataset."""

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Sequence

import torch

from simshift.data.base_data import BaseDataset, BaseSample


@dataclass
class MotorSample(BaseSample):
    """Motor mesh sample, with added material tag."""

    mesh_material: torch.Tensor

    def pin_memory(self):
        super().pin_memory()
        self.mesh_material = self.mesh_material.pin_memory()
        return self

    def to(self, device):
        super().to(device)
        self.mesh_material = self.mesh_material.to(device)
        return self


class MotorDataset(BaseDataset):
    """Electric Motor dataset class."""

    dataset_id = "motor"
    field_names = [
        "cauchy_stress",
        "logarithmic_strain",
    ]

    def __init__(
        self,
        path: str = "./datasets/motor",
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

    def __getitem__(self, idx: int) -> MotorSample:
        sample = self.data[idx]
        return MotorSample(
            cond=sample["cond"],
            y=sample["y"],
            mesh_coords=sample["mesh_coords"],
            y_mesh_coords=sample["y_mesh_coords"],
            mesh_edges=sample["edge_index"],
            mesh_material=sample["mesh_material"],
            sample_id=sample["sample_id"],
            mesh_element_connectivity=sample["mesh_element_connectivity"],
        )

    def collate(self, batch: Sequence[MotorSample]):
        base_sample = super().collate(batch)
        return MotorSample(
            cond=base_sample.cond,
            y=base_sample.y,
            mesh_coords=base_sample.mesh_coords,
            y_mesh_coords=base_sample.y_mesh_coords,
            mesh_edges=base_sample.mesh_edges,
            batch_index=base_sample.batch_index,
            mesh_material=torch.cat([sample.mesh_material for sample in batch], dim=0),
        )

    @property
    def n_materials(self):
        return int(self.data[0]["mesh_material"].max()) + 1


def get_motor_dataset(
    split: str,
    normalization_method: Literal["zscore", "minmax"] = "zscore",
    normalization_stats: Optional[Dict] = None,
    **kwargs,
):
    """Return a configured electric motor dataset by loading it from disk."""
    # source domain
    dataset_source = MotorDataset(split=split, domain="source", **kwargs)

    if split == "train":
        normalization_stats = dataset_source.get_normalization_stats(
            method=normalization_method
        )
    assert normalization_stats is not None
    dataset_source.normalization_stats = normalization_stats
    dataset_source.normalize(method=normalization_method)

    # taget domain
    dataset_target = MotorDataset(split=split, domain="target", **kwargs)

    dataset_target.normalization_stats = normalization_stats
    dataset_target.normalize(method=normalization_method)

    return (dataset_source, dataset_target), normalization_stats
