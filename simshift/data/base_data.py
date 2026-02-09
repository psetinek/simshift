"""SimShift Dataset baseclass."""

import json
import os.path as osp
from dataclasses import asdict, dataclass
from typing import Literal, Optional, Sequence

import h5py
import numpy as np
import torch
from einops import rearrange
from sklearn.neighbors import NearestNeighbors
from torch.utils.data import Dataset
from tqdm import tqdm
import pandas as pd

from simshift.data.utils import download_data


@dataclass(kw_only=True)
class BaseSample:
    """Generic mesh sample."""

    cond: torch.Tensor
    y: torch.Tensor
    y_mesh_coords: torch.Tensor
    mesh_coords: torch.Tensor
    mesh_edges: torch.Tensor
    mesh_element_connectivity: Optional[torch.Tensor] = None
    batch_index: Optional[torch.Tensor] = None
    sample_id: Optional[int] = None

    def pin_memory(self):
        self.cond = self.cond.pin_memory()
        self.y = self.y.pin_memory()
        self.mesh_coords = self.mesh_coords.pin_memory()
        self.mesh_edges = self.mesh_edges.pin_memory()
        self.y_mesh_coords = self.y_mesh_coords.pin_memory()
        if self.batch_index is not None:
            self.batch_index = self.batch_index.pin_memory()
        return self

    def to(self, device):
        # only move things we need as part of model forward
        self.cond = self.cond.to(device)
        self.y = self.y.to(device)
        self.y_mesh_coords = self.y_mesh_coords.to(device)
        self.mesh_coords = self.mesh_coords.to(device)
        self.mesh_edges = self.mesh_edges.to(device)
        if self.batch_index is not None:
            self.batch_index = self.batch_index.to(device)
        return self

    def as_dict(self):
        # return sample as dict for model's forward
        d = asdict(self)
        d.pop("y")
        d.pop("y_mesh_coords")
        d.pop("mesh_element_connectivity")
        d.pop("sample_id")
        return d


class BaseDataset(Dataset):
    """SIMSHIFT Dataset baseclass.

    Handles downloading, loading, processing and normalizing mesh-based SIMSHIFT data.

    Args:
        path (str): Path to .h5 dataset.
        difficulty ("easy", "medium", "hard): Domain-gap level. Defaults to "medium".
        split ("train", "val", "test"): Dataset split. Defaults to "train".
        domain ("source", "target"): Source or target domain. Defaults to "source".
        dtype (torch.dtype): Dataset dtype. Defaults to torch.float32.
        kwargs: Dataset-specific arguments
    """

    dataset_id = None
    field_names = None

    def __init__(
        self,
        path: str,
        splits_path: str = None,
        metadata_path: str = None,
        difficulty: Literal["easy", "medium", "hard"] = "medium",
        split: Literal["train", "val", "test"] = "train",
        domain: Literal["source", "target"] = "source",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        _ = kwargs
        # check if dataset is already downloaded
        if not osp.exists(path):
            print(
                f"Dataset ({self.dataset_id}) not found in '{path}'. \
                    Starting download."
            )
            download_data(
                repo_id="simshift/SIMSHIFT_data",
                filename=f"{self.dataset_id}.zip",
                local_dir=path,
            )
        self.path = path
        self.splits_path = (
            osp.join(path, "splits.json") if splits_path is None else splits_path
        )
        self.metadata_path = (
            osp.join(path, "metadata.csv") if metadata_path is None else metadata_path
        )
        self.metadata_df = pd.read_csv(self.metadata_path)
        self.difficulty = difficulty
        self.split = split
        self.domain = domain
        self.dtype = dtype
        self.normalization_stats = None

        self.data, self.channels, self.conds = self.load_data()

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        return BaseSample(
            cond=sample["cond"],
            y=sample["y"],
            mesh_coords=sample["mesh_coords"],
            y_mesh_coords=sample["y_mesh_coords"],
            mesh_edges=sample["edge_index"],
            mesh_element_connectivity=sample["mesh_element_connectivity"],
            sample_id=sample["sample_id"],
        )

    def load_data(self):
        # load metadata
        with open(self.splits_path, "r") as f:
            splits_metadata = json.load(f)
        split = self.split
        if self.domain == "source":
            domain = "src"
        if self.domain == "target":
            domain = "tgt"
        sample_ids = splits_metadata[self.difficulty][domain][split]
        data, channels, conds = [], {}, {}
        for sample_id in tqdm(
            sample_ids,
            desc=f"Loading data (split={split}, domain={domain})",
            total=len(sample_ids),
        ):
            with h5py.File(
                osp.join(self.path, str(sample_id) + ".h5"), "r", swmr=True
            ) as h5f:
                # load mesh info and build nearest neighbor graph
                mesh_cords_initial = torch.tensor(
                    h5f["mesh/node_coords"][0, :, :], dtype=self.dtype
                )
                mesh_coords_final = torch.tensor(
                    h5f["mesh/node_coords"][-1, :, :], dtype=self.dtype
                )
                mesh_element_connectivity = torch.tensor(
                    h5f["mesh/element_connectivity"][:, :], dtype=torch.long
                )
                mesh_cords_initial = (
                    mesh_cords_initial
                    - mesh_cords_initial.min(dim=0, keepdim=True).values
                )
                mesh_coords_final = (
                    mesh_coords_final
                    - mesh_coords_final.min(dim=0, keepdim=True).values
                )

                # build nearest neighbor graph
                x0_np = mesh_cords_initial.cpu().numpy()
                nbrs = NearestNeighbors(n_neighbors=5).fit(x0_np)
                _, indices = nbrs.kneighbors(mesh_cords_initial)
                # create source indices: [0,0,0,0,0, 1,1,1,1,1, ...]
                source_nodes = np.repeat(np.arange(x0_np.shape[0]), indices.shape[1])
                # flatten target indices: [n0_0, n0_1..., n1_0, n1_1...]
                target_nodes = indices.flatten()
                # stack to get (2, E) shape directly
                mesh_edges_np = np.stack([source_nodes, target_nodes], axis=0)
                mesh_edges = torch.from_numpy(mesh_edges_np).to(torch.long)

                # load fields
                fields, channels = [], {}
                cnt = 0
                for field_name in self.field_names:
                    if h5f[f"node_fields/{field_name}"].ndim == 3:
                        _field = torch.tensor(
                            h5f[f"node_fields/{field_name}"][-1, :, :], dtype=self.dtype
                        )
                    else:
                        _field = torch.tensor(
                            h5f[f"node_fields/{field_name}"][-1, :], dtype=self.dtype
                        ).unsqueeze(-1)
                    fields.append(_field)
                    channels[field_name] = slice(cnt, cnt + _field.shape[-1])
                    cnt += _field.shape[-1]
                fields = torch.cat(fields, dim=-1)

                # load material if available
                if "material" in h5f["node_fields"]:
                    material = torch.tensor(
                        h5f["node_fields/material"][-1, :], dtype=torch.int
                    ).unsqueeze(-1)

                # build cond vector
                row = self.metadata_df[self.metadata_df["sample_id"] == sample_id]
                row = row.drop(columns="sample_id")
                cond_np = row.iloc[0].to_numpy(dtype=np.float32)
                cond = torch.from_numpy(cond_np).to(self.dtype)

                # build sample
                sample = {
                    "cond": cond,
                    "y": fields,
                    "mesh_coords": mesh_cords_initial,
                    "y_mesh_coords": mesh_coords_final,
                    "edge_index": mesh_edges,
                    "mesh_element_connectivity": mesh_element_connectivity,
                    "sample_id": sample_id,
                }
                if "material" in h5f["node_fields"]:
                    sample["mesh_material"] = material
                data.append(sample)

            cond_names = row.columns.to_list()
            conds = {name: i for i, name in enumerate(cond_names)}
        return data, channels, conds

    def _collect_dataset_for_stats(self):
        # data_fids = sorted(self.data.keys())
        data_fids = list(range(len(self.data)))
        num_samples = len(self.data)
        n_nodes = [self.data[k]["y"].shape[0] for k in data_fids]
        space = self.data[data_fids[0]]["mesh_coords"].shape[-1]
        ptr = np.cumsum([0] + n_nodes)  # starting 0 for slicing
        ys = torch.empty((sum(n_nodes), self.n_channels), dtype=torch.float32)
        conds = torch.empty((num_samples, self.n_conds), dtype=torch.float32)
        meshes = torch.empty((sum(n_nodes), space), dtype=torch.float32)
        meshes_y = torch.empty((sum(n_nodes), space), dtype=torch.float32)
        for i, fidx in enumerate(data_fids):  # sorted for determinism
            ys[ptr[i] : ptr[i + 1], :] = self.data[fidx]["y"]
            conds[i] = self.data[fidx]["cond"]
            meshes[ptr[i] : ptr[i + 1], :] = self.data[fidx]["mesh_coords"]
            meshes_y[ptr[i] : ptr[i + 1], :] = self.data[fidx]["y_mesh_coords"]
        ys = rearrange(ys, "bn c -> c bn")
        meshes = rearrange(meshes, "bn dim -> dim bn")
        meshes_y = rearrange(meshes_y, "bn dim -> dim bn")

        return ys, conds, meshes, meshes_y

    def get_normalization_stats(
        self,
        method: Literal["zscore", "minmax"],
    ):
        """Compute and cache normalization parameters for the dataset."""
        ys, conds, meshes, meshes_y = self._collect_dataset_for_stats()

        if method == "zscore":
            means = torch.mean(ys, dim=-1)
            stds = torch.std(ys, dim=-1)
            cond_means = torch.mean(conds, dim=0)
            cond_stds = torch.std(conds, dim=0)
            # cond_means = torch.ones_like(conds).squeeze()
            # cond_stds = torch.ones_like(conds).squeeze()
            normalization_stats = {
                "mean": means,
                "std": stds,
                "cond_means": cond_means,
                "cond_stds": cond_stds,
            }
        else:
            mins = torch.min(ys, dim=-1).values
            maxs = torch.max(ys, dim=-1).values
            ranges = maxs - mins
            cond_mins = torch.min(conds, dim=0).values
            cond_maxs = torch.max(conds, dim=0).values
            cond_ranges = cond_maxs - cond_mins
            normalization_stats = {
                "min": mins,
                "range": ranges,
                "cond_mins": cond_mins,
                "cond_ranges": cond_ranges,
            }

        meshes_concat = torch.stack(
            [torch.max(meshes, dim=-1).values, torch.max(meshes_y, dim=-1).values]
        )
        normalization_stats["mesh_maxs"] = torch.max(
            meshes_concat, keepdim=True, dim=0
        ).values

        return normalization_stats

    def normalize(self, method: Literal["zscore", "minmax"]):
        # per dataset normalization
        assert (
            self.normalization_stats is not None
        ), "No normalization parameters set. Please calculate and set them first!"

        for idx in range(len(self.data)):
            # conditions and fields
            self.data[idx]["y"] = self.normalize_fields(self.data[idx]["y"], method)
            self.data[idx]["cond"] = self.normalize_cond(self.data[idx]["cond"], method)
            # coordinates
            self.data[idx]["y_mesh_coords"] = (
                self.data[idx]["y_mesh_coords"] / self.normalization_stats["mesh_maxs"]
            )
            self.data[idx]["mesh_coords"] = (
                self.data[idx]["mesh_coords"] / self.normalization_stats["mesh_maxs"]
            )

    def normalize_fields(self, y, method):
        # output fields
        params = self.normalization_stats
        if method == "zscore":
            mean = params["mean"].unsqueeze(0)
            std = params["std"].unsqueeze(0) + 1e-8
            y = (y - mean) / std
        else:
            # minmax
            min = params["min"].unsqueeze(0)
            range = params["range"].unsqueeze(0) + 1e-8
            y = 2 * (y - min) / range - 1
        return y

    def normalize_cond(self, cond: torch.Tensor, method: str) -> torch.Tensor:
        # conditionings
        params = self.normalization_stats
        if method == "zscore":
            cond = (cond - params["cond_means"]) / (params["cond_stds"] + 1e-8)
        else:
            # minmax
            cond_mins = params["cond_mins"]
            cond_ranges = params["cond_ranges"] + 1e-8
            cond = 2 * (cond - cond_mins) / cond_ranges - 1
        return cond

    def denormalize(self, conditionings, fields):
        cond_stds = self.normalization_stats["cond_stds"].to(fields.device)
        cond_means = self.normalization_stats["cond_means"].to(fields.device)
        fields_stds = self.normalization_stats["std"].unsqueeze(0).to(fields.device)
        fields_means = self.normalization_stats["mean"].unsqueeze(0).to(fields.device)
        if conditionings is not None:
            return (
                conditionings * cond_stds + cond_means,
                fields * fields_stds + fields_means,
            )
        else:
            return fields * fields_stds + fields_means

    def denormalize_coords(self, coords):
        return coords * self.normalization_stats["mesh_maxs"].to(coords.device)

    def collate(self, batch: Sequence[BaseSample]):
        n_nodes = [sample.mesh_coords.shape[0] for sample in batch]
        node_offsets = torch.cumsum(torch.tensor([0] + n_nodes[:-1]), dim=0)
        edge_index = torch.cat(
            [
                sample.mesh_edges + offset
                for sample, offset in zip(batch, node_offsets, strict=False)
            ],
            dim=1,
        )
        n_nodes = torch.tensor(n_nodes)
        graph_index = torch.repeat_interleave(torch.arange(n_nodes.shape[0]), n_nodes)

        return BaseSample(
            cond=torch.stack([sample.cond for sample in batch]),
            y=torch.cat([sample.y for sample in batch], dim=0),
            mesh_coords=torch.cat([sample.mesh_coords for sample in batch], dim=0),
            y_mesh_coords=torch.cat([sample.y_mesh_coords for sample in batch], dim=0),
            mesh_edges=edge_index,
            batch_index=graph_index,
        )

    def __len__(self):
        return len(self.data)

    @property
    def n_channels(self):
        return sum([slc.stop - slc.start for slc in self.channels.values()])

    @property
    def n_conds(self):
        return len(self.conds)
