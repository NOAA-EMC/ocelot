"""
Offline computation of per-variable DA increment (anal - ges) statistics.

Mirrors DARegionalGraphDataset._process_state_nodes (graph_dataset.py) but skips
all mesh/graph construction — it only needs ges/anal features_norm per bin.

Currently the model normalizes the increment implicitly, by taking the difference
of two field-normalized tensors:

    y = anal_norm - ges_norm = (anal_raw - ges_raw) / field_std

Since field_std is a constant shared by anal/ges, the increment-normalized target
graphcast_residuals.md recommends is just a per-variable rescale of the existing y:

    y_new = y / std(y)     # std(y) computed per variable, across the training set

This script computes that per-variable std(y) (plus mean, for completeness) and
prints/saves it in the same {feature_key: [mean, std]} shape as FEATURE_STATS, so
it can be dropped into an obs_config module and consumed the same way.

Usage:
    python -m ocelot.compute_increment_stats \
        --data_path /scratch3/NCEPDEV/stmp/Xin.C.Jin/data/ocelot/data_v6/diag_urma/ \
        --start_date 2025-02-01 --end_date 2025-02-28 --delta_time 1 \
        --obs_config urma --exp_type regional_da
"""
import argparse
import json

import numpy as np
import pandas as pd

from ocelot.dataset_timeseries import ParquetDataManager, generate_binned_timestamp_list
from ocelot.logger import logger
from ocelot.observation_config import load_observation_config

_STATE_GES = "ges"
_STATE_ANAL = "anal"


class WelfordAccumulator:
    """Per-column running mean/variance (Welford's algorithm); numerically stable
    across many small batches without holding all data in memory at once."""

    def __init__(self, n_features: int):
        self.count = np.zeros(n_features, dtype=np.int64)
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.m2 = np.zeros(n_features, dtype=np.float64)

    def update(self, values: np.ndarray, valid: np.ndarray) -> None:
        """values, valid: (n_rows, n_features)."""
        for j in range(values.shape[1]):
            col_valid = valid[:, j]
            n_new = int(col_valid.sum())
            if n_new == 0:
                continue
            col = values[col_valid, j].astype(np.float64)
            batch_mean = col.mean()
            batch_var = col.var()  # population variance for this batch

            n_a, mean_a, m2_a = self.count[j], self.mean[j], self.m2[j]
            n_b = n_new
            delta = batch_mean - mean_a
            n_ab = n_a + n_b
            mean_ab = mean_a + delta * n_b / n_ab
            m2_ab = m2_a + batch_var * n_b + delta**2 * n_a * n_b / n_ab

            self.count[j] = n_ab
            self.mean[j] = mean_ab
            self.m2[j] = m2_ab

    def finalize(self):
        std = np.sqrt(np.where(self.count > 1, self.m2 / np.maximum(self.count, 1), 0.0))
        return self.mean, std, self.count


def main():
    parser = argparse.ArgumentParser(description="Compute per-variable DA increment (anal - ges) stats offline.")
    parser.add_argument("--data_path", type=str,
                         default="/scratch3/NCEPDEV/stmp/Xin.C.Jin/data/ocelot/data_v6/diag_urma/")
    parser.add_argument("--start_date", type=str, default="2025-02-01")
    parser.add_argument("--end_date", type=str, default="2025-02-28")
    parser.add_argument("--delta_time", type=int, default=1)
    parser.add_argument("--obs_config", type=str, default="urma")
    parser.add_argument("--exp_type", type=str, default="regional_da")
    parser.add_argument("--out", type=str, default=None,
                         help="Optional path to write the resulting stats as JSON.")
    args = parser.parse_args()

    observation_config, feature_stats, fill_values, _, _ = load_observation_config(
        exp_type=args.exp_type, config_name=args.obs_config
    )
    ges_cfg = observation_config.get("state", {}).get(_STATE_GES, {})
    if not ges_cfg:
        raise ValueError(f"No state/{_STATE_GES} entry in observation_config for obs_config={args.obs_config!r}")
    feature_keys = ges_cfg["features"]

    ges_obs_config = {"state": {_STATE_GES: ges_cfg}}
    ges_loader = ParquetDataManager(
        data_dir=args.data_path,
        observation_config=ges_obs_config,
        feature_stats=feature_stats,
        fill_values=fill_values,
        delta_time=args.delta_time,
    )

    anal_zarr_name = ges_cfg.get("anal_zarr_name", _STATE_ANAL)
    anal_obs_config = {"state": {_STATE_ANAL: {**ges_cfg, "zarr_name": anal_zarr_name}}}
    anal_feature_stats = {**feature_stats, _STATE_ANAL: feature_stats.get(_STATE_GES, {})}
    anal_loader = ParquetDataManager(
        data_dir=args.data_path,
        observation_config=anal_obs_config,
        feature_stats=anal_feature_stats,
        fill_values=fill_values,
        delta_time=args.delta_time,
    )

    start_ts = pd.to_datetime(args.start_date)
    end_ts = pd.to_datetime(args.end_date)
    bin_names = generate_binned_timestamp_list(
        start_ts.strftime("%m/%d"), end_ts.strftime("%m/%d"), start_ts.year, args.delta_time
    )

    acc = WelfordAccumulator(len(feature_keys))

    for i, bin_name in enumerate(bin_names):
        ges_bin = ges_loader.get_data_for_bin(bin_name)
        anal_bin = anal_loader.get_data_for_bin(bin_name)
        ges_data = (ges_bin or {}).get("state", {}).get(_STATE_GES)
        anal_data = (anal_bin or {}).get("state", {}).get(_STATE_ANAL)
        if ges_data is None or anal_data is None:
            logger.warning(f"[{i+1}/{len(bin_names)}] {bin_name}: missing ges/anal, skipping.")
            continue

        ges_norm = ges_data["features_norm"]
        anal_norm = anal_data["features_norm"]
        if ges_norm.numel() == 0 or anal_norm.numel() == 0 or ges_norm.shape[0] != anal_norm.shape[0]:
            logger.warning(f"[{i+1}/{len(bin_names)}] {bin_name}: empty or mismatched rows, skipping.")
            continue

        ges_valid = ges_data["features_valid_mask"].numpy()
        anal_valid = anal_data["features_valid_mask"].numpy()
        both_valid = ges_valid & anal_valid

        increment = (anal_norm - ges_norm).numpy()
        acc.update(increment, both_valid)

        logger.info(f"[{i+1}/{len(bin_names)}] {bin_name}: rows={increment.shape[0]}, "
                    f"running n={acc.count.tolist()}")

    mean, std, count = acc.finalize()

    print("\nPer-variable DA increment stats (normalized-field space, matches data[target].y):")
    print(f"{'feature':<24}{'count':>10}{'mean':>12}{'std':>12}")
    result = {}
    for key, m, s, n in zip(feature_keys, mean, std, count):
        print(f"{key:<24}{n:>10}{m:>12.5f}{s:>12.5f}")
        result[key] = [float(m), float(s)]

    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote stats to {args.out}")
    else:
        print("\nSTATE_INCREMENT_STATS = " + json.dumps(result, indent=4))


if __name__ == "__main__":
    main()
