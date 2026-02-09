import h5py
import pandas as pd
from copy import deepcopy
import json
import os
import os.path as osp
import argparse
import numpy as np
from typing import Dict, Iterable, Tuple

from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split HDF5 dataset into source/target domains and train/val/test splits."
    )
    p.add_argument("--data-path", required=True, help="Path to the dataset folder.")
    p.add_argument(
        "--config-file",
        required=True,
        default=None,
        help="Path to json file with the domain ranges and the ratios.",
    )
    p.add_argument(
        "--output-name",
        required=False,
        default="splits.json",
        help="Name for the output splits file (e.g. splits.json).",
    )
    return p.parse_args()


def split_domains_multi_difficulty(
    cond_df: pd.DataFrame,
    src_range: list,
    target_ranges: dict,
    split_col: str,
    src_ratios: tuple = (0.6, 0.2, 0.2),
    tgt_ratios: tuple = (0.6, 0.4),
    seed: int = 42,
    exclusive_upper: bool = True,
) -> dict:
    """
    Split domains into easy, medium, and hard based on difficulty ranges.
    
    Args:
        cond_df: DataFrame with condition columns
        src_range: [min, max] range for source domain
        target_ranges: Dict with keys 'easy', 'medium', 'hard' mapping to [min, max] ranges for target
        split_col: Column name to use for splitting
        src_ratios: Tuple of (train, val, test) ratios for source domain
        tgt_ratios: Tuple of (train, test) ratios for target domain
        seed: Random seed for reproducibility
        exclusive_upper: If True, use [min, max) intervals to avoid overlap
    
    Returns:
        Dict with splits for each difficulty level
    """
    original_cond_df = deepcopy(cond_df)
    splits = {}
    
    # Fixed source domain
    source_range = src_range
    if exclusive_upper:
        filtered_df = original_cond_df[
            (original_cond_df[split_col] >= source_range[0]) &
            (original_cond_df[split_col] < source_range[1])
        ]
    else:
        filtered_df = original_cond_df[
            (original_cond_df[split_col] >= source_range[0]) &
            (original_cond_df[split_col] <= source_range[1])
        ]
    sample_ids_source = list(filtered_df["sample_id"])
    
    # Create source splits (same for all difficulties)
    src_train, src_val_test = train_test_split(
        sample_ids_source, train_size=src_ratios[0], random_state=seed
    )
    src_val, src_test = train_test_split(
        src_val_test, train_size=src_ratios[1]/(1-src_ratios[0]), random_state=seed
    )
    
    # Track all used sample IDs to detect overlap
    all_target_samples = {}
    
    # Create target splits for each difficulty
    for difficulty, target_range in target_ranges.items():
        if exclusive_upper:
            filtered_target_df = original_cond_df[
                (original_cond_df[split_col] >= target_range[0]) &
                (original_cond_df[split_col] < target_range[1])
            ]
        else:
            filtered_target_df = original_cond_df[
                (original_cond_df[split_col] >= target_range[0]) &
                (original_cond_df[split_col] <= target_range[1])
            ]
        
        sample_ids_target = list(filtered_target_df["sample_id"])
        all_target_samples[difficulty] = set(sample_ids_target)
        
        tgt_train, tgt_test = train_test_split(
            sample_ids_target, train_size=tgt_ratios[0], random_state=seed
        )
        
        splits[difficulty] = {
            "src": {
                "train": src_train,
                "val": src_val,
                "test": src_test
            },
            "tgt": {
                "train": tgt_train,
                "val": [],
                "test": tgt_test
            }
        }

    return splits



def main():
    args = parse_args()

    # get arguments
    data_path = args.data_path
    config_file = args.config_file
    output_name = args.output_name

    # load metadata and splitting config
    metadata_df = pd.read_csv(osp.join(data_path, "metadata.csv"))
    with open(config_file, "r") as f:
        config = json.load(f)

    splits = split_domains_multi_difficulty(
        cond_df=metadata_df,
        src_range=config["source_range"],
        target_ranges=config["target_ranges"],
        split_col=config["split_param"],
        src_ratios=tuple(config["source_ratios"]),
        tgt_ratios=tuple(config["target_ratios"]),
        seed=config["seed"])
    
    # print some split statistics
    for difficulty, split in splits.items():
        print(f"Difficulty: {difficulty}")
        print(f"  Source - Train: {len(split['src']['train'])}, Val: {len(split['src']['val'])}, Test: {len(split['src']['test'])}")
        print(f"  Target - Train: {len(split['tgt']['train'])}, Test: {len(split['tgt']['test'])}")
    
    # save splits
    with open(osp.join(data_path, output_name), "w") as f:
        json.dump(splits, f, indent=4)


if __name__ == "__main__":
    main()