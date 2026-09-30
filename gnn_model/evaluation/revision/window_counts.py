#!/usr/bin/env python
"""Observations the model actually sees per 12-h window (after QC and subsampling).

Counts rows (observation locations) in the obs-space prediction CSVs, which contain exactly the
target observations of each forecast window after QC, satellite-ID filtering, and the configured
random subsampling. Input windows are built from the same archives with the same rules, so their
size is comparable. Reports, per instrument, the number of windows found, the mean and median
observations per window, and the mean number of valid channel values per window; then
extrapolates to the 6,574 training windows (2015-2023) for the Sec. 3.1 dataset-size statement.

Usage (CPU, from gnn_model/):
    python evaluation/revision/window_counts.py --pred_dir predictions/clim_dump_2015_2023 --recursive \
        --out evaluation/revision/results/window_counts_train.csv
    python evaluation/revision/window_counts.py \
        --pred_dir predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space \
        --out evaluation/revision/results/window_counts_test2025.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import list_prediction_files  # noqa: E402

N_TRAIN_WINDOWS = 6574


def _count(path: str) -> tuple[int, int]:
    """(rows, valid channel values) for one CSV; reads only the mask columns."""
    cols = pd.read_csv(path, nrows=0).columns
    mask_cols = [c for c in cols if c.startswith("mask_")]
    if not mask_cols:
        with open(path, "rb") as f:
            return max(sum(1 for _ in f) - 1, 0), 0
    m = pd.read_csv(path, usecols=mask_cols)
    vals = m.astype(str).apply(lambda s: s.str.strip().str.lower().isin(["true", "1", "1.0"])).to_numpy()
    return int(len(m)), int(vals.sum())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    files = list_prediction_files(args.pred_dir, args.recursive)
    if not files:
        raise SystemExit(f"No prediction CSVs under {args.pred_dir}")
    recs = []
    for i, (path, inst, init) in enumerate(files):
        rows, vals = _count(path)
        recs.append(dict(instrument=inst, init=init, rows=rows, valid_values=vals))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(files)} files", flush=True)
    df = pd.DataFrame(recs)
    g = df.groupby("instrument")
    out = pd.DataFrame({
        "n_windows": g.size(),
        "mean_obs_per_window": g["rows"].mean().round(0),
        "median_obs_per_window": g["rows"].median(),
        "mean_valid_values_per_window": g["valid_values"].mean().round(0),
    })
    out["train_total_obs_est"] = (out["mean_obs_per_window"] * N_TRAIN_WINDOWS).round(0)
    out.loc["ALL"] = [np.nan, out["mean_obs_per_window"].sum(), np.nan,
                      out["mean_valid_values_per_window"].sum(), out["train_total_obs_est"].sum()]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out)
    pd.set_option("display.float_format", lambda v: f"{v:,.0f}")
    print(out.to_string())
    print(f"(train_total_obs_est = mean per window x {N_TRAIN_WINDOWS:,} training windows; "
          f"each window's observations serve as targets once and as inputs once)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
