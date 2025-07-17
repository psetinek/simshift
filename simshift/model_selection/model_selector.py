from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from simshift.model_selection import get_model_selection_algorithm
from simshift.model_selection.utils import (
    get_latents,
    get_mse_loss,
    get_predictions,
    train_domain_classifier,
)


class ModelSelector:
    """
    Model selection class that serves the purpose of unsupervised model selection in
    domain adaptation tasks.

    Evaluates multiple model selection algorithms, that choose models from candidate
    models, on both source and target domains.
    """

    def __init__(
        self,
        algorithm_names: List[str],
        candidate_models: List[nn.Module],
        trainset_source: Dataset,
        valset_source: Dataset,
        trainset_target: Dataset,
        testset_source: Dataset,
        testset_target: Dataset,
        batch_size: int = 16,
        device: Optional[str] = "cuda",
    ):
        """Initialize the model selector with datasets and candidate models.

        Args:
            algorithm_names (List[str]): Names of model selection algorithms to use.
                                         (Note, that they must be registered model
                                         selection strategies.)
            candidate_models (List[nn.Module]): List of candidate PyTorch models select
                                                from and evaluate.
            trainset_source (Dataset): Source domain training dataset.
            valset_source (Dataset): Source domain validation dataset.
            trainset_target (Dataset): Target domain training dataset.
            testset_source (Dataset): Source domain test dataset.
            testset_target (Dataset): Target domain test dataset.
            batch_size (int): Batch size for data loaders used for computation.
                              Default: 16
            device (str): Computation device ('cuda' or 'cpu'). Default: 'cuda'
        """
        self.algorithm_names = algorithm_names
        self.candidate_models = candidate_models
        self.trainset_source = trainset_source
        self.valset_source = valset_source
        self.testset_source = testset_source
        self.trainset_target = trainset_target
        self.testset_target = testset_target
        self.batch_size = batch_size
        self.device = device

        self.trainloader_source = DataLoader(
            self.trainset_source,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=trainset_source.collate,
        )
        self.valloader_source = DataLoader(
            self.valset_source,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=valset_source.collate,
        )
        self.testloader_source = DataLoader(
            self.testset_source,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=testset_source.collate,
        )
        self.trainloader_target = DataLoader(
            self.trainset_target,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=trainset_target.collate,
        )
        self.testloader_target = DataLoader(
            self.testset_target,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=testset_target.collate,
        )

        # retrieve model selection algorithms
        self.algorithms = [
            get_model_selection_algorithm(alg_name) for alg_name in algorithm_names
        ]

        self.model_weights = None

    def _estimate_density_ratios(self, model: nn.Module) -> torch.Tensor:
        """Estimate domain density ratios using a domain classifier.

        Args:
            model (nn.Module): Candidate model whose conditioning network is used for
                feature extraction.

        Returns:
            torch.Tensor: Density ratio (weights) tensor of shape
                          [n_validation_samples].

        Process:
            1. Extracts latent features from model's conditioning network
            2. Trains domain classifier to distinguish source/target features
            3. Estimates density ratios using classifier probabilities
        """
        # get latents
        (
            trainset_source_latents,
            valset_source_latents,
            trainset_target_latents,
        ) = get_latents(
            model.conditioning,
            self.trainset_source,
            self.valset_source,
            self.trainset_target,
            device=self.device,
        )

        # estimate the density ratio (weights) using a logistic regression
        domain_classifier = train_domain_classifier(
            trainset_source_latents,
            trainset_target_latents,
            num_epochs=1000,
            device=self.device,
        )
        domain_classifier.eval()
        with torch.no_grad():
            source_val_cls = domain_classifier(valset_source_latents)

        n_train_source = trainset_source_latents.size(0)
        n_train_target = trainset_target_latents.size(0)

        # weights
        weights = (n_train_target / n_train_source) * (
            (1 - source_val_cls) / source_val_cls
        )
        weights = weights.squeeze(1)

        return weights

    def compute_model_weights(self) -> None:
        """Compute aggregation weights for all candidate models using specified model
        selection algorithms.

        Computes:
            model_weights (torch.Tensor): Aggregation weights tensor of shape
                [n_algorithms, n_models]

        Process:
            1. Estimates importance weights for each model
            2. Gathers various losses needed for selection
            3. Applies each selection algorithm to compute final weights

        Note:
            Must be called before compute_test_performance()
        """
        model_weights = torch.zeros(
            [len(self.algorithms), len(self.candidate_models)], device="cpu"
        )

        importance_weights = torch.zeros(
            [len(self.candidate_models), len(self.valset_source)], device="cpu"
        )
        source_val_loss = torch.zeros(
            [len(self.candidate_models), len(self.valset_source)], device="cpu"
        )
        target_test_loss = torch.zeros([len(self.candidate_models)], device="cpu")

        for i, model in enumerate(self.candidate_models):
            # get density ration estimates (weights) and valset_source loss
            # (for all algorithms)
            _importance_weights = self._estimate_density_ratios(model)
            importance_weights[i, :] = _importance_weights

            _source_val_loss = get_mse_loss(
                model, self.valloader_source, device=self.device
            )

            source_val_loss[i, :] = _source_val_loss.to("cpu")

            _target_test_loss = get_mse_loss(
                model, self.testloader_target, device=self.device
            )
            target_test_loss[i] = _target_test_loss.to("cpu").mean()

        # compute aggregation weights for the different algorithms
        for i, algorithm in enumerate(self.algorithms):
            model_weights[i, :] = algorithm(
                weights=importance_weights,
                source_val_loss=source_val_loss,
                target_test_loss=target_test_loss,
            )

        self.model_weights = model_weights

        # clear gpu memory
        torch.cuda.empty_cache()

    def compute_test_performance(
        self,
    ) -> Tuple[
        torch.Tensor,  # source_loss_per_algorithm
        torch.Tensor,  # target_loss_per_algorithm
        torch.Tensor,  # rmse_src_per_field
        torch.Tensor,  # rmse_tgt_per_field
        torch.Tensor,  # rmse_src_deformation
        torch.Tensor,  # rmse_tgt_deformation
    ]:
        """Evaluate ensemble performance using computed model weights.

        Returns:
            Tuple[torch.Tensor, ...]: Performance metrics tuple containing:
            - source_loss_per_algorithm (torch.Tensor): Normalized RMSE on source domain
              of shape [n_algorithms]
            - target_loss_per_algorithm (torch.Tensor): Normalized RMSE on target domain
              of shape [n_algorithms]
            - rmse_src_per_field (torch.Tensor): Denormalized RMSE per field (source)
              of shape [n_algorithms, n_fields]
            - rmse_tgt_per_field (torch.Tensor): Denormalized RMSE per field (target)
              of shape [n_algorithms, n_fields]
            - rmse_src_deformation (torch.Tensor): Coordinate RMSE (source)
              of shape [n_algorithms]
            - rmse_tgt_deformation (torch.Tensor): Coordinate RMSE (target)
              of shape [n_algorithms]

        Process:
            1. Generates ensemble predictions using model weights
            2. Computes both normalized and denormalized metrics
            3. Calculates coordinate-specific deformation errors
        """
        assert self.model_weights is not None and self.model_weights.shape[0] == len(
            self.algorithms
        )
        # compute model predictions of all candidate models
        model_predictions_source = []
        model_predictions_target = []
        for model in self.candidate_models:
            source_preds, source_batch_index = get_predictions(
                model, self.testloader_source, device=self.device
            )
            source_preds = source_preds.to("cpu")
            source_batch_index = source_batch_index.to("cpu")
            target_preds, target_batch_index = get_predictions(
                model, self.testloader_target, device=self.device
            )
            target_preds = target_preds.to("cpu")
            target_batch_index = target_batch_index.to("cpu")
            model_predictions_source.append(source_preds)
            model_predictions_target.append(target_preds)
        model_predictions_source = torch.stack(
            model_predictions_source
        )  # [n_models, total_source_nodes, n_fields]
        model_predictions_target = torch.stack(
            model_predictions_target
        )  # [n_models, total_target_nodes, n_fields]

        # self.model_weights: [n_algorithms, n_models]
        ensemble_predictions_source = self.model_weights.reshape(
            [*self.model_weights.shape, 1, 1]
        ) * model_predictions_source.reshape(
            [1, *model_predictions_source.shape]
        )  # [n_algorithms, n_models, total_source_nodes, n_fields]
        ensemble_predictions_source = torch.sum(
            ensemble_predictions_source, dim=1
        )  # [n_algorithms, total_source_nodes, n_fields]
        ensemble_predictions_target = self.model_weights.reshape(
            [*self.model_weights.shape, 1, 1]
        ) * model_predictions_target.reshape(
            [1, *model_predictions_target.shape]
        )  # [n_algorithms, n_models, total_target_nodes, n_fields]
        ensemble_predictions_target = torch.sum(
            ensemble_predictions_target, dim=1
        )  # [n_algorithms, total_target_nodes, n_fields]

        source_loader = DataLoader(
            self.testset_source,
            batch_size=len(self.testset_source),
            collate_fn=self.testset_source.collate,
        )
        target_loader = DataLoader(
            self.testset_target,
            batch_size=len(self.testset_target),
            collate_fn=self.testset_target.collate,
        )
        source_sample = next(iter(source_loader))
        source_sample = source_sample.to("cpu")
        target_sample = next(iter(target_loader))
        target_sample = target_sample.to("cpu")

        source_gt = torch.cat([source_sample.y, source_sample.y_mesh_coords], dim=-1)
        source_gt = source_gt.unsqueeze(0)  # [1, total_source_nodes, n_fields]
        source_batch_index = source_sample.batch_index
        target_gt = torch.cat([target_sample.y, target_sample.y_mesh_coords], dim=-1)
        target_gt = target_gt.unsqueeze(0)  # [1, total_target_nodes, n_fields]
        target_batch_index = target_sample.batch_index

        # 1) compute losses: normalized rmse losses (across all fields and positions)
        source_loss = (ensemble_predictions_source[..., :-2] - source_gt[..., :-2]) ** 2
        # source_loss_per_node = source_loss.mean(dim=-1)
        source_loss_per_graph = torch.zeros(
            [len(self.algorithms), len(self.testset_source), source_gt.shape[-1]],
            device="cpu",
        )
        source_batch_index_expanded = (
            source_batch_index.unsqueeze(0).unsqueeze(-1).expand(source_loss.shape)
        )
        source_loss_per_graph.scatter_reduce_(
            dim=1, index=source_batch_index_expanded, src=source_loss, reduce="mean"
        )
        source_loss_per_algorithm = source_loss_per_graph.sqrt().mean(dim=(1, 2))
        # source_loss_per_algorithm = source_mse_loss_per_algorithm.sqrt()

        target_loss = (ensemble_predictions_target[..., :-2] - target_gt[..., :-2]) ** 2
        # target_loss_per_node = target_loss.mean(dim=-1)
        target_loss_per_graph = torch.zeros(
            [len(self.algorithms), len(self.testset_target), target_gt.shape[-1]],
            device="cpu",
        )
        target_batch_index_expanded = (
            target_batch_index.unsqueeze(0).unsqueeze(-1).expand(target_loss.shape)
        )
        target_loss_per_graph.scatter_reduce_(
            dim=1, index=target_batch_index_expanded, src=target_loss, reduce="mean"
        )
        target_loss_per_algorithm = target_loss_per_graph.sqrt().mean(dim=(1, 2))
        # target_loss_per_algorithm = target_mse_loss_per_algorithm.sqrt()

        # 2) compute losses: denormalized RMSE for each field and denormalized
        # coordinates loss
        # denormalize predictions
        # ensemble_predictions: [n_algorithms, total_nodes, n_fields]
        ensemble_predictions_source_denormalized = self.testset_source.denormalize(
            None,
            ensemble_predictions_source[..., : -source_sample.y_mesh_coords.shape[-1]],
        )  # slice to remove coordinates
        ensemble_predictions_target_denormalized = self.testset_target.denormalize(
            None,
            ensemble_predictions_target[..., : -target_sample.y_mesh_coords.shape[-1]],
        )

        # denormalize ground truth
        # gt: [1, total_nodes, n_fields]
        source_gt_denormalized = self.testset_source.denormalize(
            None, source_gt[..., : -source_sample.y_mesh_coords.shape[-1]]
        )
        target_gt_denormalized = self.testset_target.denormalize(
            None, target_gt[..., : -target_sample.y_mesh_coords.shape[-1]]
        )

        # RMSE across nodes
        mse_src_fields = (
            ensemble_predictions_source_denormalized - source_gt_denormalized
        ) ** 2
        mse_src_fields_per_graph = torch.zeros(
            [len(self.algorithms), len(self.testset_source), mse_src_fields.shape[-1]],
            device="cpu",
        )
        source_batch_index_expanded = (
            source_batch_index.unsqueeze(0).unsqueeze(-1).expand(mse_src_fields.shape)
        )
        mse_src_fields_per_graph.scatter_reduce_(
            dim=1, index=source_batch_index_expanded, src=mse_src_fields, reduce="mean"
        )
        rmse_src_per_field = mse_src_fields_per_graph.sqrt().mean(
            1
        )  # [n_algorithms, n_fields]
        # rmse_src_per_field = mse_src_per_field.sqrt()

        mse_tgt_fields = (
            ensemble_predictions_target_denormalized - target_gt_denormalized
        ) ** 2
        mse_tgt_fields_per_graph = torch.zeros(
            [len(self.algorithms), len(self.testset_target), mse_tgt_fields.shape[-1]],
            device="cpu",
        )
        target_batch_index_expanded = (
            target_batch_index.unsqueeze(0).unsqueeze(-1).expand(mse_tgt_fields.shape)
        )
        mse_tgt_fields_per_graph.scatter_reduce_(
            dim=1, index=target_batch_index_expanded, src=mse_tgt_fields, reduce="mean"
        )
        rmse_tgt_per_field = mse_tgt_fields_per_graph.sqrt().mean(
            1
        )  # [n_algorithms, n_fields]
        # rmse_tgt_per_field = mse_tgt_per_field.sqrt()

        # denormalize pred coords
        ensemble_coords_source_denormalized = self.testset_source.denormalize_coords(
            ensemble_predictions_source[..., -source_sample.y_mesh_coords.shape[-1] :]
        )
        ensemble_coords_target_denormalized = self.testset_target.denormalize_coords(
            ensemble_predictions_target[..., -target_sample.y_mesh_coords.shape[-1] :]
        )

        # denormalize gt coords
        source_gt_coords_denormalized = self.testset_source.denormalize_coords(
            source_gt[..., -source_sample.y_mesh_coords.shape[-1] :]
        )
        target_gt_coords_denormalized = self.testset_target.denormalize_coords(
            target_gt[..., -target_sample.y_mesh_coords.shape[-1] :]
        )

        # squared‐error per node, sum over coord_dim, then mean over nodes
        coord_rmse_src = (
            ((ensemble_coords_source_denormalized - source_gt_coords_denormalized) ** 2)
            .sum(dim=-1)
            .sqrt()
        )
        rmse_src_coords_per_graph = torch.zeros(
            [len(self.algorithms), len(self.testset_source)], device="cpu"
        )
        source_batch_index_expanded = source_batch_index.unsqueeze(0).expand(
            coord_rmse_src.shape
        )
        rmse_src_coords_per_graph.scatter_reduce_(
            dim=1, index=source_batch_index_expanded, src=coord_rmse_src, reduce="mean"
        )
        rmse_src_deformation = rmse_src_coords_per_graph.mean(1)

        coord_rmse_tgt = (
            ((ensemble_coords_target_denormalized - target_gt_coords_denormalized) ** 2)
            .sum(dim=-1)
            .sqrt()
        )
        rmse_tgt_coords_per_graph = torch.zeros(
            [len(self.algorithms), len(self.testset_target)], device="cpu"
        )
        target_batch_index_expanded = target_batch_index.unsqueeze(0).expand(
            coord_rmse_tgt.shape
        )
        rmse_tgt_coords_per_graph.scatter_reduce_(
            dim=1, index=target_batch_index_expanded, src=coord_rmse_tgt, reduce="mean"
        )
        rmse_tgt_deformation = rmse_tgt_coords_per_graph.mean(1)

        return (
            source_loss_per_algorithm,
            target_loss_per_algorithm,
            rmse_src_per_field,
            rmse_tgt_per_field,
            rmse_src_deformation,
            rmse_tgt_deformation,
        )
