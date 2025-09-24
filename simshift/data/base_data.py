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

from simshift.data.utils import download_data


@dataclass(kw_only=True)
class BaseSample:
    """Generic mesh sample."""

    cond: torch.Tensor
    y: torch.Tensor
    y_mesh_coords: torch.Tensor
    mesh_coords: torch.Tensor
    mesh_edges: torch.Tensor
    batch_index: Optional[torch.Tensor] = None

    def pin_memory(self):
        self.y = self.y.pin_memory()
        self.mesh_coords = self.mesh_coords.pin_memory()
        self.mesh_edges = self.mesh_edges.pin_memory()
        self.y_mesh_coords = self.y_mesh_coords.pin_memory()
        return self

    def to(self, device):
        self.cond = self.cond.to(device)
        self.y = self.y.to(device)
        self.y_mesh_coords = self.y_mesh_coords.to(device)
        self.mesh_coords = self.mesh_coords.to(device)
        self.mesh_edges = self.mesh_edges.to(device)
        self.batch_index = self.batch_index.to(device)
        return self

    def as_dict(self):
        # return sample as dict for model's forward
        d = asdict(self)
        d.pop("y")
        d.pop("y_mesh_coords")
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

    def __init__(
        self,
        path: str,
        difficulty: Literal["easy", "medium", "hard"] = "medium",
        split: Literal["train", "val", "test"] = "train",
        domain: Literal["source", "target"] = "source",
        dtype: torch.dtype = torch.float32,
        **kwargs,
    ):
        _ = kwargs
        self.path = osp.join(path, self.dataset_id + ".h5")
        self.splits_path = osp.join(path, "splits.json")
        self.difficulty = difficulty
        self.split = split
        self.domain = domain
        self.dtype = dtype
        self.normalization_stats = None

        # check if dataset is already downloaded
        if not osp.exists(path):
            print(
                f"Dataset ({self.dataset_id}) not found in '{osp.dirname(path)}'. \
                    Starting download."
            )
            download_data(
                repo_id="simshift/SIMSHIFT_data",
                filename=f"{self.dataset_id}.zip",
                local_dir=osp.dirname(path),
            )

        self.data, self.channels, self.conds = self.load_data()

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        return BaseSample(
            cond=sample["cond"],
            y=sample["y"],
            mesh_coords=sample["mesh_coords"],
            y_mesh_coords=sample["y_mesh_coords"],
            mesh_edges=sample["edge_index"],
        )

    def load_data(self):
        # load metadata
        with open(self.splits_path, "r") as f:
            splits_metadata = json.load(f)
        if self.domain == "source":
            domain = "src"
        if self.domain == "target":
            domain = "tgt"
        split = self.split
        data_indices = splits_metadata[self.difficulty][domain][split]
        # load data
        data = {}
        with h5py.File(self.path, "r", swmr=True) as h5f:
            channels = {k: v[:] for k, v in h5f["metadata/channels"].items()}
            conds = {k: v[:] for k, v in h5f["metadata/cond"].items()}
            has_material = "material" in list(h5f["data"]["0000"].keys())
            for i, data_index in tqdm(
                enumerate(data_indices),
                desc=f"Loading data (split={split}, domain={domain})",
                total=len(data_indices),
            ):
                sample_args = {
                    "cond": h5f["data"][str(data_index).zfill(4)]["cond"][:],
                    "mesh_coords": h5f["data"][str(data_index).zfill(4)]["coords"][:],
                    "mesh_fields": h5f["data"][str(data_index).zfill(4)]["fields"][:],
                    "mesh_material": None if not has_material else h5f["data"][str(data_index).zfill(4)]["material"][:]
                }
                sample_results = self._load_sample(**sample_args)
                sample = {}
                for key, v in sample_results.items():
                    if isinstance(v, np.ndarray):
                        sample[key] = torch.from_numpy(v)
                        if "edge_index" in key:
                            sample[key] = sample[key].to(dtype=torch.long)
                        elif "mesh_material" in key:
                            sample[key] = sample[key].to(dtype=torch.long)
                        else:
                            sample[key] = sample[key].to(dtype=self.dtype)
                data[i] = sample
        # remove coords, make as slices
        channels = {k: slice(c[0], c[-1] + 1) for k, c in channels.items()}
        return data, channels, conds

    def _load_sample(
        self,
        cond: np.ndarray,
        mesh_coords: np.ndarray,
        mesh_fields: np.ndarray,
        mesh_material: np.ndarray = None,
    ):
        p_mesh_coords = mesh_coords[0]
        y_mesh_coords = mesh_coords[1]
        # Only use mesh_fields at the final step
        mesh_fields = mesh_fields[1]  # after transformation

        # shift all samples to (0,0)
        p_mesh_coords = p_mesh_coords - np.min(p_mesh_coords, axis=0, keepdims=True)
        y_mesh_coords = y_mesh_coords - np.min(y_mesh_coords, axis=0, keepdims=True)

        nbrs = NearestNeighbors(n_neighbors=5).fit(p_mesh_coords)
        _, indices = nbrs.kneighbors(p_mesh_coords)
        mesh_edges = []
        for i, neighbors in enumerate(indices):
            for neighbor in neighbors:
                mesh_edges.append((i, neighbor))
        mesh_edges = np.array(mesh_edges)
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

    def _collect_dataset_for_stats(self):
        data_fids = sorted(self.data.keys())
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

        for idx in self.data.keys():
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
            dim=0,
        ).T
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
