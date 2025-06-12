from dataclasses import dataclass
from typing import Dict, Literal, Optional

import numpy as np
import torch

from simshift.data.base_data import BaseDataset, BaseSample


@dataclass
class HeatsinkSample(BaseSample):
    pass


class HeatsinkDataset(BaseDataset):
    """Heatsink dataset class."""

    dataset_id = "heatsink"

    def __init__(
        self,
        n_subsampled_nodes: int,
        path: str = "./datasets/heatsink",
        difficulty: Literal["easy", "medium", "hard"] = "medium",
        split: Literal["train", "val", "test"] = "train",
        domain: Literal["source", "target"] = "source",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        self.n_subsampled_nodes = n_subsampled_nodes
        super().__init__(
            path=path,
            difficulty=difficulty,
            split=split,
            domain=domain,
            dtype=dtype,
            **kwargs,
        )

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        n_nodes = sample["y"].shape[0]
        if self.n_subsampled_nodes is not None:
            keep_indices = np.sort(
                np.random.choice(n_nodes, self.n_subsampled_nodes, replace=False)
            )  # sorted for consistency
            y = sample["y"][keep_indices, :]
            mesh_coords = sample["mesh_coords"][keep_indices, :]
            y_mesh_coords = sample["y_mesh_coords"][keep_indices, :]
        else:
            y = sample["y"]
            mesh_coords = sample["mesh_coords"]
            y_mesh_coords = sample["y_mesh_coords"]
        return HeatsinkSample(
            cond=sample["cond"],
            y=y,
            mesh_coords=mesh_coords,
            y_mesh_coords=y_mesh_coords,
            mesh_edges=sample["edge_index"],
        )

    def _load_sample(
        self,
        cond: np.ndarray,
        mesh_coords: np.ndarray,
        mesh_fields: np.ndarray,
        mesh_material: np.ndarray = None,
    ):
        p_mesh_coords = mesh_coords
        y_mesh_coords = mesh_coords

        # shift all samples to (0,0)
        p_mesh_coords = p_mesh_coords - np.min(p_mesh_coords, axis=0, keepdims=True)
        y_mesh_coords = y_mesh_coords - np.min(y_mesh_coords, axis=0, keepdims=True)

        mesh_edges = np.zeros([1, 1])
        dict_out = {
            "cond": cond,
            "y": mesh_fields,
            "mesh_coords": p_mesh_coords,
            "y_mesh_coords": y_mesh_coords,
            "edge_index": mesh_edges,
        }
        if mesh_material is not None:
            dict_out["mesh_material"] = mesh_material[0]
        return dict_out


def get_heatsink_dataset(
    split: str,
    normalization_method: Literal["zscore", "minmax"] = "zscore",
    normalization_stats: Optional[Dict] = None,
    **kwargs,
):
    # source domain
    dataset_source = HeatsinkDataset(split=split, domain="source", **kwargs)

    if split == "train":
        normalization_stats = dataset_source.get_normalization_stats(
            method=normalization_method
        )
    assert normalization_stats is not None
    dataset_source.normalization_stats = normalization_stats
    dataset_source.normalize(method=normalization_method)

    # taget domain
    dataset_target = HeatsinkDataset(split=split, domain="target", **kwargs)

    dataset_target.normalization_stats = normalization_stats
    dataset_target.normalize(method=normalization_method)

    return (dataset_source, dataset_target), normalization_stats
