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
        description="Split HDF5 dataset into source/target domains and splits."
    )
    p.add_argument("--data-path", required=True, help="Path to the .h5 dataset file.")
    p.add_argument(
        "--src-filters",
        required=False,
        default=None,
        help="JSON dict mapping condition -> [min,max], e.g. \"{'thickness':[50,150], 'temp_core':[900,1100]}\"",
    )

    p.add_argument(
        "--src-ratios",
        nargs=3,
        type=float,
        default=(0.5, 0.25, 0.25),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Source domain ratios (must sum to 1). Default: 0.5 0.25 0.25",
    )
    p.add_argument(
        "--tgt-ratios",
        nargs=2,
        type=float,
        default=(0.5, 0.5),
        metavar=("TRAIN", "TEST"),
        help="Target domain ratios (must sum to 1). Default: 0.5 0.5",
    )
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument(
        "--output-dir", required=True, help="Directory to write splits.json."
    )
    return p.parse_args()


def summarize_conditions(df: pd.DataFrame) -> dict:
    """
    Summarize ranges for each condition column (min, max).

    Returns:
        dict of {cond_name: {"min": float, "max": float}}
    """
    summary = {}
    for col in df.columns:
        if col == "sample_id":
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        summary[col] = {
            "min": float(np.nanmin(vals.values)),
            "max": float(np.nanmax(vals.values)),
        }
    return summary


def get_conditioning_dataframe(dataset_path: str) -> pd.DataFrame:
    """
    Build a DataFrame of per-sample condition values.

    The resulting DataFrame has:
        'sample_id' : str
        <cond_name> : float         (one column per condition)

    Args:
        dataset_path: Path to the .h5 dataset.

    Returns:
        DataFrame with one row per sample and columns as described above.
    """
    rows = []
    with h5py.File(dataset_path, "r") as f:
        # get conditioning slices lookup df
        condition_names = list(f["metadata"]["cond"].keys())
        condition_slices = {}
        for condition_name in condition_names:
            condition_slices[condition_name] = f["metadata"]["cond"][condition_name][:]

        # gather dict rows
        sample_ids = list(f["data"].keys())
        for sample_id in sample_ids:
            df_row = {}
            df_row["sample_id"] = sample_id
            for cond_name, cond_slice in condition_slices.items():
                df_row[cond_name] = f["data"][sample_id]["cond"][:][cond_slice].item()
            rows.append(df_row)
    return pd.DataFrame(rows)


def split_domains(
    cond_df: pd.DataFrame,
    src_filters: Dict[str, Iterable | float],
    src_ratios: Tuple[float, float, float] = (0.5, 0.25, 0.25),
    tgt_ratios: Tuple[float, float] = (0.5, 0.5),
    seed: int = 42,
) -> dict:
    """
    Filter a dataframe of conditions into source/target domains and then
    split each into train/val/test (source) and train/test (target).

    Args:
        cond_df: DataFrame with columns ["sample_id", <cond1>, <cond2>, ...].
        src_filters: dict mapping condition name -> scalar or [min, max] range.
                     Rows satisfying all filter ranges go to the source domain;
                     the complement goes to the target domain.
        src_ratios: (train, val, test) fractions that sum to 1.0 for source.
        tgt_ratios: (train, test) fractions that sum to 1.0 for target.
        seed: RNG seed for reproducible splits.

    Returns:
        Nested dict:
        {
          "medium": {
            "src": {"train": [...], "val": [...], "test": [...]},
            "tgt": {"train": [...], "val": [],   "test": [...]}
          }
        }
    """
    original_cond_df = deepcopy(cond_df)
    # filter source and target indices
    for condition_name, source_range in src_filters.items():
        cond_df = cond_df[
            (cond_df[condition_name] >= min(source_range))
            & (cond_df[condition_name] <= max(source_range))
        ]
    sample_ids_source = list(cond_df["sample_id"])
    sample_ids_target = list(
        original_cond_df[~original_cond_df["sample_id"].isin(sample_ids_source)][
            "sample_id"
        ]
    )

    # create train, test, val splits
    src_train, src_val_test = train_test_split(
        sample_ids_source, train_size=src_ratios[0], random_state=seed
    )
    src_val, src_test = train_test_split(
        src_val_test, train_size=src_ratios[1] / (1 - src_ratios[0]), random_state=seed
    )
    tgt_train, tgt_test = train_test_split(
        sample_ids_target, train_size=tgt_ratios[0], random_state=seed
    )
    splits = {
        "medium": {
            "src": {"train": src_train, "val": src_val, "test": src_test},
            "tgt": {"train": tgt_train, "val": [], "test": tgt_test},
        }
    }
    return splits


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # build conditioning dataframe
    cond_df = get_conditioning_dataframe(dataset_path=args.data_path)

    # load filters
    src_filters = json.loads(args.src_filters)
    if not src_filters:
        print("\nNo --src-filters provided. Available conditions and ranges:")
        summary = summarize_conditions(cond_df)
        print(json.dumps(summary, indent=2))
        raise SystemExit(
            "\nPlease re-run with --src-filters specifying ranges, e.g.\n"
            "  --src-filters '{\"thickness\":[50,150], \"temp_core\":[900,1100]}'\n"
        )

    # compute splits
    splits = split_domains(
        cond_df=cond_df,
        src_filters=src_filters,
        src_ratios=tuple(args.src_ratios),
        tgt_ratios=tuple(args.tgt_ratios),
        seed=args.seed,
    )

    # save splits
    out_path = osp.join(args.output_dir, "splits.json")
    with open(out_path, "w") as f:
        json.dump(splits, f, indent=2)
    print(f"Saved splits to: {out_path}")

    # print a quick summary
    src_counts = {k: len(v) for k, v in splits["medium"]["src"].items()}
    tgt_counts = {k: len(v) for k, v in splits["medium"]["tgt"].items()}
    print("\nCounts:")
    print("  Source:", src_counts)
    print("  Target:", tgt_counts)


if __name__ == "__main__":
    main()
