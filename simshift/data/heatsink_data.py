from dataclasses import dataclass
from typing import Dict, Literal, Optional
import os.path as osp
import json
from tqdm import tqdm
import h5py

import numpy as np
import torch

from simshift.data.base_data import BaseDataset, BaseSample


@dataclass(kw_only=True)
@dataclass
class HeatsinkSample(BaseSample):
    """Heatsink mesh sample."""

    mesh_element_types: Optional[torch.Tensor] = None

    def as_dict(self):
        _dict = super().as_dict()
        _dict.pop("mesh_element_types", None)
        return _dict


class HeatsinkDataset(BaseDataset):
    """Heatsink dataset class."""

    dataset_id = "heatsink"
    field_names = ["U", "T", "p_rgh"]

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

    def subsample(self, n_subsampling_nodes):
        for k, v in self.data.items():
            n_nodes = v["y"].shape[0]
            keep_indices = np.sort(
                np.random.choice(n_nodes, n_subsampling_nodes, replace=False)
            )  # sorted for consistency
            v["y"] = v["y"][keep_indices, :]
            v["mesh_coords"] = v["mesh_coords"][keep_indices, :]
            v["y_mesh_coords"] = v["y_mesh_coords"][keep_indices, :]
            self.data[k] = v

    def load_data(self):
        # load metadata
        with open(self.splits_path, "r") as f:
            splits_metadata = json.load(f)
        split = self.split
        if self.domain == "source":
            domain = "src"
        if self.domain == "target":
            domain = "tgt"
            if self.split == "val":
                split = "test"
        sample_ids = splits_metadata[self.difficulty][domain][split]
        # sample_ids = sample_ids[:20]
        data = {}
        for idx, sample_id in tqdm(
            enumerate(sample_ids),
            desc=f"Loading data (split={split}, domain={domain})",
            total=len(sample_ids),
        ):
            with h5py.File(osp.join(self.path, str(sample_id) + ".h5"), "r") as h5f:
                # load mesh info and build nearest neighbor graph
                mesh_coords = torch.from_numpy(h5f["mesh/node_coords"][:, :]).to(
                    dtype=self.dtype
                )
                mesh_element_connectivity = torch.from_numpy(
                    h5f["mesh/element_connectivity"][:]
                ).to(torch.long)
                mesh_element_types = torch.from_numpy(h5f["mesh/element_types"][:]).to(
                    torch.uint8
                )
                mesh_coords = mesh_coords - mesh_coords.min(dim=0, keepdim=True).values
                mesh_coords_initial = mesh_coords
                mesh_coords_final = mesh_coords.clone().detach()

                mesh_edges = torch.zeros([1, 1], dtype=torch.int)

                # load fields
                fields, channels = [], {}
                cnt = 0
                for field_name in self.field_names:
                    _field = torch.from_numpy(h5f[f"node_fields/{field_name}"][:]).to(
                        dtype=self.dtype
                    )
                    if _field.ndim == 1:
                        _field = _field[:, None]
                    fields.append(_field)
                    channels[field_name] = slice(cnt, cnt + _field.shape[-1])
                    cnt += _field.shape[-1]
                fields = torch.cat(fields, dim=-1)

                # build cond vector
                row = self.metadata_df[self.metadata_df["sample_id"] == sample_id]
                # row = row.drop(columns="sample_id")
                row = row.drop(
                    columns=[
                        "sample_id",
                        "envTemp",
                        "flowVelocity",
                        "height1",
                        "length",
                        "pressure",
                        "turbulentKE",
                        "turbulentOmega",
                        "width",
                    ]
                )
                cond_np = row.iloc[0].to_numpy(dtype=np.float32)
                cond = torch.from_numpy(cond_np).to(self.dtype)

                # build sample
                sample = {
                    "cond": cond,
                    "y": fields,
                    "mesh_coords": mesh_coords_initial,
                    "y_mesh_coords": mesh_coords_final,
                    "edge_index": mesh_edges,
                    "mesh_element_connectivity": mesh_element_connectivity,
                    "mesh_element_types": mesh_element_types,
                }
                data[idx] = sample

        cond_names = row.columns.to_list()
        conds = {name: i for i, name in enumerate(cond_names)}
        return data, channels, conds

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
            mesh_element_connectivity=sample["mesh_element_connectivity"],
            mesh_element_types=sample["mesh_element_types"],
        )


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
