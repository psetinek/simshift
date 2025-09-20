import argparse
import os
import os.path as osp

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from simshift.data import get_test_data
from simshift.model_selection.model_selector import ModelSelector
from simshift.model_selection.utils import compute_rmse, get_predictions
from simshift.utils import load_model, set_seed

from simshift.model_selection.utils import (
    rolling_evaluate_middle_line,
    forming_evaluate_middle_line,
    motor_evaluate_chord,
    heatsink_evaluate_middle_line,
)

from simshift.data.rolling_data import RollingDataset
from simshift.data.forming_data import FormingDataset
from simshift.data.motor_data import MotorDataset
from simshift.data.heatsink_data import HeatsinkDataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model selection and compute test errors."
    )
    parser.add_argument("--entity", type=str, required=True, help="wandb entity name")
    parser.add_argument("--project", type=str, required=True, help="wandb project name")
    parser.add_argument(
        "--experiment-id",
        type=str,
        required=True,
        help="Experiment filter (experiment_id)",
    )
    parser.add_argument(
        "--model-dir", type=str, required=True, help="Directory with model checkpoints"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for DataLoader"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="paper_stuff/tables",
        help="Directory to save results",
    )
    parser.add_argument(
        "--selection-algs",
        type=str,
        nargs="+",
        default=["IWV", "DEV", "SB", "TB"],
        help="Model selection algorithms to use",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    ENTITY = args.entity
    PROJECT = args.project
    EXP_ID = {
        "config.logging.experiment_id": args.experiment_id,
    }
    MODEL_CKP_DIR = args.model_dir
    MODEL_SELECTION_ALGS = args.selection_algs

    BATCH_SIZE = args.batch_size
    DEVICE = args.device

    os.makedirs(args.output_dir, exist_ok=True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    set_seed(42)

    api = wandb.Api()
    all_runs = api.runs(f"{ENTITY}/{PROJECT}", filters=EXP_ID)

    # load datasets once
    checkpoint_path = osp.join(
        MODEL_CKP_DIR,
        all_runs[0].name,
        "best.pth",
    )
    ckp_dict = load_model(checkpoint_path, load_opt=False, load_trainset=True)
    trainset_source = ckp_dict["trainset_source"]
    valset_source = ckp_dict["valset_source"]
    trainset_target = ckp_dict["trainset_target"]
    testset = get_test_data(ckp_dict["cfg"], trainset_source.normalization_stats)
    testset_source, testset_target = testset
    if PROJECT == "heatsink":
        testset_source.n_subsampled_nodes = None
        testset_target.n_subsampled_nodes = None

    # get unqiue models, da_algorithms and seeds
    unique_models = sorted({run.config["model"]["name"] for run in all_runs})
    unique_da_algorithms = sorted(
        {run.config["da_algorithm"]["name"] for run in all_runs}
    )
    unique_seeds = sorted({run.config["seed"] for run in all_runs})

    results = []
    for model_type_name in tqdm(unique_models, desc="Models"):
        for da_algorithm_type_name in tqdm(
            unique_da_algorithms, desc="DA Algorithms", leave=False
        ):
            for seed in tqdm(unique_seeds, desc="Seeds", leave=False):
                # get the runs for the different hyperparams lambda (exclude lambda = 0)
                filters = EXP_ID.copy()
                filters.update(
                    {
                        "config.seed": seed,
                        "config.model.name": model_type_name,
                        "config.da_algorithm.name": da_algorithm_type_name,
                        "config.da_algorithm.da_loss_weight": {"$ne": 0},
                    }
                )

                runs = api.runs(f"{ENTITY}/{PROJECT}", filters=filters)
                try:
                    if len(runs) == 0:
                        continue
                except Exception:
                    print("an error occured")
                    continue
                # load models
                models = []  # {model: config]}
                cfgs = []
                for run in runs:
                    checkpoint_path = osp.join(
                        MODEL_CKP_DIR,
                        run.name,
                        "best.pth",
                    )
                    ckp_dict = load_model(
                        checkpoint_path, load_opt=False, valset=valset_source
                    )
                    models.append(ckp_dict["model"])
                    cfgs.append(ckp_dict["cfg"])
                # model selection (aggregation and test loss)
                model_selector = ModelSelector(
                    algorithm_names=MODEL_SELECTION_ALGS,
                    candidate_models=models,
                    trainset_source=trainset_source,
                    valset_source=valset_source,
                    trainset_target=trainset_target,
                    testset_source=testset_source,
                    testset_target=testset_target,
                    batch_size=BATCH_SIZE,
                    device=DEVICE,
                )
                model_selector.compute_model_weights()
                (
                    test_loss_source_per_alg,
                    test_loss_target_per_alg,
                    rmse_source_per_field,
                    rmse_target_per_field,
                    rmse_source_deformation,
                    rmse_target_deformation,
                    custom_error_source,
                    custom_error_target,
                ) = model_selector.compute_test_performance()

                # record results
                for i, algo_name in enumerate(MODEL_SELECTION_ALGS):
                    result = {
                        "model_name": model_type_name,
                        "da_algorithm_name": da_algorithm_type_name,
                        "model_selection_algorithm_name": algo_name,
                        "seed": seed,
                        "test_loss_source": test_loss_source_per_alg[i].item(),
                        "test_loss_target": test_loss_target_per_alg[i].item(),
                    }
                    # add deformation rmse
                    result["test_loss_source_deformation"] = rmse_source_deformation[
                        i
                    ].item()
                    result["test_loss_target_deformation"] = rmse_target_deformation[
                        i
                    ].item()
                    # add rmse for each field
                    for field_name, field_slice in valset_source.channels.items():
                        result[f"test_loss_source_{field_name}"] = (
                            rmse_source_per_field[i, field_slice].mean().item()
                        )
                        result[f"test_loss_target_{field_name}"] = (
                            rmse_target_per_field[i, field_slice].mean().item()
                        )
                    # add custom evaluation metrics
                    result["test_loss_source_custom"] = custom_error_source[i].item()
                    result["test_loss_target_custom"] = custom_error_target[i].item()
                    results.append(result)

        # for every model type, include results for an unregularized run
        # (da_loss_weight=0)
        for seed in unique_seeds:
            filters = EXP_ID.copy()
            filters.update(
                {
                    "config.seed": seed,
                    "config.model.name": model_type_name,
                    "config.da_algorithm.name": "deep_coral",  # depending on where you
                    # run without da_loss
                    "config.da_algorithm.da_loss_weight": 0,
                }
            )
            runs = api.runs(f"{ENTITY}/{PROJECT}", filters=filters)
            try:
                if len(runs) == 0:
                    continue
            except Exception:
                print("an error occured")
                continue
            assert len(runs) == 1

            # load model
            checkpoint_path = osp.join(
                MODEL_CKP_DIR,
                runs[0].name,
                "best.pth",
            )
            ckp_dict = load_model(checkpoint_path, load_opt=False, valset=valset_source)
            model = ckp_dict["model"]
            cfg = ckp_dict["cfg"]

            # compute test losses
            source_loader = DataLoader(
                testset_source,
                batch_size=BATCH_SIZE,
                collate_fn=testset_source.collate,
            )
            target_loader = DataLoader(
                testset_target,
                batch_size=BATCH_SIZE,
                collate_fn=testset_target.collate,
            )

            source_preds, source_batch_index = get_predictions(
                model, source_loader, device=DEVICE
            )
            source_preds = source_preds.to("cpu")
            source_batch_index = source_batch_index.to("cpu")
            target_preds, target_batch_index = get_predictions(
                model, target_loader, device=DEVICE
            )
            target_preds = target_preds.to("cpu")
            target_batch_index = target_batch_index.to("cpu")
            # clear gpu memory
            torch.cuda.empty_cache()
            source_loader_long = DataLoader(
                testset_source,
                batch_size=len(testset_source),
                collate_fn=testset_source.collate,
            )
            target_loader_long = DataLoader(
                testset_target,
                batch_size=len(testset_target),
                collate_fn=testset_target.collate,
            )
            source_sample = next(iter(source_loader_long))
            source_sample = source_sample.to("cpu")
            target_sample = next(iter(target_loader_long))
            target_sample = target_sample.to("cpu")
            source_gt = torch.cat(
                [source_sample.y, source_sample.y_mesh_coords], dim=-1
            )
            target_gt = torch.cat(
                [target_sample.y, target_sample.y_mesh_coords], dim=-1
            )

            # 1) compute losses: normalized rmse losses
            # (across all fields and positions)
            source_loss = (source_preds[..., :-2] - source_gt[..., :-2]) ** 2
            source_loss_per_sample = torch.zeros(
                [source_sample.batch_index.max().item() + 1, source_loss.shape[-1]]
            ).to("cpu")
            source_batch_index_expanded = source_batch_index.unsqueeze(-1).expand(
                source_loss.shape
            )
            source_loss_per_sample.scatter_reduce_(
                dim=0, index=source_batch_index_expanded, src=source_loss, reduce="mean"
            )
            source_loss = source_loss_per_sample.sqrt().mean(dim=(0, 1))

            (
                source_loss_new,
                source_loss_positions,
                source_loss_new_denormalized,
                source_loss_positions_denormalized,
            ) = compute_rmse(model, source_loader, True, device=DEVICE)
            source_loss_new = source_loss_new.to("cpu")
            source_loss_positions = source_loss_positions.to("cpu")
            source_loss_new_denormalized = source_loss_new_denormalized.to("cpu")
            source_loss_positions_denormalized = source_loss_positions_denormalized.to(
                "cpu"
            )

            target_loss = (target_preds[..., :-2] - target_gt[..., :-2]) ** 2
            # target_loss_per_node = target_loss.mean(dim=-1)
            target_loss_per_sample = torch.zeros(
                [target_sample.batch_index.max().item() + 1, target_loss.shape[-1]]
            ).to("cpu")
            target_batch_index_expanded = target_batch_index.unsqueeze(-1).expand(
                target_loss.shape
            )
            target_loss_per_sample.scatter_reduce_(
                dim=0, index=target_batch_index_expanded, src=target_loss, reduce="mean"
            )
            target_loss = target_loss_per_sample.sqrt().mean(dim=(0, 1))

            (
                target_loss_new,
                target_loss_positions,
                target_loss_new_denormalized,
                target_loss_positions_denormalized,
            ) = compute_rmse(model, target_loader, True, device=DEVICE)
            target_loss_new = target_loss_new.to("cpu")
            target_loss_positions = target_loss_positions.to("cpu")
            target_loss_new_denormalized = target_loss_new_denormalized.to("cpu")
            target_loss_positions_denormalized = target_loss_positions_denormalized.to(
                "cpu"
            )

            source_loader = DataLoader(
                testset_source,
                batch_size=len(testset_source),
                collate_fn=testset_source.collate,
            )
            target_loader = DataLoader(
                testset_target,
                batch_size=len(testset_target),
                collate_fn=testset_target.collate,
            )
            source_sample = next(iter(source_loader))
            source_sample = source_sample.to("cpu")
            target_sample = next(iter(target_loader))
            target_sample = target_sample.to("cpu")
            # 2) compute losses: denormalized RMSE for each field and denormalized
            # coordinates loss
            # denormalize predictions
            # ensemble_predictions: [n_algorithms, total_nodes, n_fields]
            cond_source_denormalized, predictions_source_denormalized = (
                testset_source.denormalize(
                    source_sample.cond,
                    source_preds[..., : -source_sample.y_mesh_coords.shape[-1]],
                )
            )  # slice to remove coordinates
            cond_target_denormalized, predictions_target_denormalized = (
                testset_target.denormalize(
                    target_sample.cond,
                    target_preds[..., : -target_sample.y_mesh_coords.shape[-1]],
                )
            )

            # denormalize ground truth
            # gt: [total_nodes, n_fields]
            source_gt_denormalized = testset_source.denormalize(
                None, source_gt[..., : -source_sample.y_mesh_coords.shape[-1]]
            )
            target_gt_denormalized = testset_target.denormalize(
                None, target_gt[..., : -target_sample.y_mesh_coords.shape[-1]]
            )

            # RMSE across nodes
            mse_src_fields = (
                predictions_source_denormalized - source_gt_denormalized
            ) ** 2
            mse_src_fields_per_sample = torch.zeros(
                [len(testset_source), mse_src_fields.shape[-1]], device="cpu"
            )
            source_batch_index_expanded = source_batch_index.unsqueeze(-1).expand(
                mse_src_fields.shape
            )
            mse_src_fields_per_sample.scatter_reduce_(
                dim=0,
                index=source_batch_index_expanded,
                src=mse_src_fields,
                reduce="mean",
            )
            rmse_src_fields = mse_src_fields_per_sample.sqrt().mean(0)

            mse_tgt_fields = (
                predictions_target_denormalized - target_gt_denormalized
            ) ** 2
            mse_tgt_fields_per_sample = torch.zeros(
                [len(testset_target), mse_tgt_fields.shape[-1]], device="cpu"
            )
            target_batch_index_expanded = target_batch_index.unsqueeze(-1).expand(
                mse_tgt_fields.shape
            )
            mse_tgt_fields_per_sample.scatter_reduce_(
                dim=0,
                index=target_batch_index_expanded,
                src=mse_tgt_fields,
                reduce="mean",
            )
            rmse_tgt_fields = mse_tgt_fields_per_sample.sqrt().mean(0)

            # denormalize pred coords
            ensemble_coords_source_denormalized = testset_source.denormalize_coords(
                source_preds[..., -source_sample.y_mesh_coords.shape[-1] :]
            )
            ensemble_coords_target_denormalized = testset_target.denormalize_coords(
                target_preds[..., -target_sample.y_mesh_coords.shape[-1] :]
            )

            # denormalize gt coords
            source_gt_coords_denormalized = testset_source.denormalize_coords(
                source_gt[..., -source_sample.y_mesh_coords.shape[-1] :]
            )
            target_gt_coords_denormalized = testset_target.denormalize_coords(
                target_gt[..., -target_sample.y_mesh_coords.shape[-1] :]
            )

            # squared‐error per node, sum over coord_dim, then mean over nodes
            coord_rmse_src = (
                (
                    (
                        ensemble_coords_source_denormalized
                        - source_gt_coords_denormalized
                    )
                    ** 2
                )
                .sum(dim=-1)
                .sqrt()
            )
            rmse_src_coords_per_graph = torch.zeros([len(testset_source)], device="cpu")
            rmse_src_coords_per_graph.scatter_reduce_(
                dim=0, index=source_batch_index, src=coord_rmse_src, reduce="mean"
            )
            rmse_src_deformation = rmse_src_coords_per_graph.mean()

            coord_rmse_tgt = (
                (
                    (
                        ensemble_coords_target_denormalized
                        - target_gt_coords_denormalized
                    )
                    ** 2
                )
                .sum(dim=-1)
                .sqrt()
            )
            rmse_tgt_coords_per_graph = torch.zeros([len(testset_target)], device="cpu")
            rmse_tgt_coords_per_graph.scatter_reduce_(
                dim=0, index=target_batch_index, src=coord_rmse_tgt, reduce="mean"
            )
            rmse_tgt_deformation = rmse_tgt_coords_per_graph.mean()

            # Dataset specific evaluation loss
            if isinstance(testset_source, RollingDataset):
                custom_error_source = rolling_evaluate_middle_line(
                    preds=predictions_source_denormalized.unsqueeze(0),
                    gts=source_gt_denormalized.unsqueeze(0),
                    coords=source_gt_coords_denormalized.unsqueeze(0),
                    batch_indices=source_batch_index,
                    x_rel_tol=0.001,
                    channel=testset_source.channels["nodes_PEEQ"],
                    eps=0.001,
                )
                custom_error_target = rolling_evaluate_middle_line(
                    preds=predictions_target_denormalized.unsqueeze(0),
                    gts=target_gt_denormalized.unsqueeze(0),
                    coords=target_gt_coords_denormalized.unsqueeze(0),
                    batch_indices=target_batch_index,
                    x_rel_tol=0.001,
                    channel=testset_target.channels["nodes_PEEQ"],
                    eps=0.001,
                )

            elif isinstance(testset_source, FormingDataset):
                custom_error_source = forming_evaluate_middle_line(
                    preds=predictions_source_denormalized.unsqueeze(0),
                    gts=source_gt_denormalized.unsqueeze(0),
                    coords=source_gt_coords_denormalized.unsqueeze(0),
                    conds=cond_source_denormalized,
                    batch_indices=source_batch_index,
                    x_rel_tol=0.01,
                    dataset=testset_source,
                    eps=1,
                )
                custom_error_target = forming_evaluate_middle_line(
                    preds=predictions_target_denormalized.unsqueeze(0),
                    gts=target_gt_denormalized.unsqueeze(0),
                    coords=target_gt_coords_denormalized.unsqueeze(0),
                    conds=cond_target_denormalized,
                    batch_indices=target_batch_index,
                    x_rel_tol=0.01,
                    dataset=testset_target,
                    eps=1,
                )

            elif isinstance(testset_source, MotorDataset):
                custom_error_source = motor_evaluate_chord(
                    preds=predictions_source_denormalized.unsqueeze(0),
                    gts=source_gt_denormalized.unsqueeze(0),
                    coords=source_gt_coords_denormalized.unsqueeze(0),
                    conds=cond_source_denormalized,
                    batch_indices=source_batch_index,
                    x_rel_tol=0.05,
                    channel=testset_source.channels["stress_mises"],
                    dataset=testset_source,
                    eps=1,
                )
                custom_error_target = motor_evaluate_chord(
                    preds=predictions_target_denormalized.unsqueeze(0),
                    gts=target_gt_denormalized.unsqueeze(0),
                    coords=target_gt_coords_denormalized.unsqueeze(0),
                    conds=cond_target_denormalized,
                    batch_indices=target_batch_index,
                    x_rel_tol=0.05,
                    channel=testset_source.channels["stress_mises"],
                    dataset=testset_target,
                    eps=1,
                )

            elif isinstance(testset_source, HeatsinkDataset):
                custom_error_source = heatsink_evaluate_middle_line(
                    preds=predictions_source_denormalized.unsqueeze(0),
                    gts=source_gt_denormalized.unsqueeze(0),
                    coords=source_gt_coords_denormalized.unsqueeze(0),
                    batch_indices=source_batch_index,
                    x_rel_tol=0.05,
                    z_rel_tol=0.05,
                    z_fixed=0.025,
                    channel=testset_source.channels["T"],
                    eps=1e-2,
                )
                custom_error_target = heatsink_evaluate_middle_line(
                    preds=predictions_target_denormalized.unsqueeze(0),
                    gts=target_gt_denormalized.unsqueeze(0),
                    coords=target_gt_coords_denormalized.unsqueeze(0),
                    batch_indices=target_batch_index,
                    x_rel_tol=0.05,
                    z_rel_tol=0.05,
                    z_fixed=0.025,
                    channel=testset_target.channels["T"],
                    eps=1e-2,
                )

            else:
                raise ValueError("Wrong dataset?!")

            # record result
            result = {
                "model_name": model_type_name,
                "da_algorithm_name": "-",
                "model_selection_algorithm_name": "-",
                "seed": seed,
                "test_loss_source": source_loss.item(),
                "test_loss_target": target_loss.item(),
            }
            # add deformation rmse
            result["test_loss_source_deformation"] = rmse_src_deformation.item()
            result["test_loss_target_deformation"] = rmse_tgt_deformation.item()
            # add rmse for each field
            for field_name, field_slice in valset_source.channels.items():
                result[f"test_loss_source_{field_name}"] = (
                    rmse_src_fields[field_slice].mean().item()
                )
                result[f"test_loss_target_{field_name}"] = (
                    rmse_tgt_fields[field_slice].mean().item()
                )
            # add custom losses
            result["test_loss_source_custom"] = custom_error_source[0].item()
            result["test_loss_target_custom"] = custom_error_target[0].item()
            results.append(result)
            # clear gpu memory
            torch.cuda.empty_cache()

    # save results as df and csv
    os.makedirs(args.output_dir, exist_ok=True)
    df_results = pd.DataFrame(results)
    df_results.to_pickle(osp.join(args.output_dir, f"results_{PROJECT}.pkl"))
    df_results.to_csv(osp.join(args.output_dir, f"results_{PROJECT}.csv"), index=False)
