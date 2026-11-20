#!/usr/bin/env python
"""Compare combined closure under the two background-endpoint definitions.

Reads FSOI/fsoi_outputs/endpoint_check_padded/<target>_<month>_<variant>/csv/fsoi_combined_closure.csv,
keeps the four test cycles (the padding cycles sit near the range edges), pairs
cycles by curr_bin, and also checks the all_channels rerun against the
seasonal run so that any code drift since the seasonal runs is visible.
"""
from pathlib import Path

import numpy as np
import pandas as pd

FSOI = Path(__file__).resolve().parent
ROOT = FSOI / "fsoi_outputs" / "endpoint_check_padded"
TEST_CYCLES = {m: [f"bin2025{mm}{d}" for d in ("1012", "1100", "1112", "1200")]
               for m, mm in (("jan", "01"), ("jul", "07"))}
SEASONAL = FSOI / "fsoi_outputs" / "seasonal_inclusion_weighted_final"
COLS = ["curr_bin", "fsoi_raw_sampled", "delta_j_actual", "closure_ratio", "sign_agreement"]


def closure(path):
    return pd.read_csv(path)[COLS]


rows = []
for target in ("aircraft", "radiosonde", "surface_obs"):
    for month in ("jan", "jul"):
        runs = {v: ROOT / f"{target}_{month}_{v}" / "csv" / "fsoi_combined_closure.csv"
                for v in ("all_channels", "valid_only")}
        if not all(p.exists() for p in runs.values()):
            print(f"{target} {month}: not finished")
            continue
        old, new = closure(runs["all_channels"]), closure(runs["valid_only"])
        old = old[old.curr_bin.isin(TEST_CYCLES[month])]
        seasonal = closure(SEASONAL / f"{target}_{month}2025" / "csv" / "fsoi_combined_closure.csv")
        m = (old.merge(new, on="curr_bin", suffixes=("_all", "_valid"))
                .merge(seasonal, on="curr_bin").rename(columns={"closure_ratio": "closure_ratio_seasonal"}))
        for r in m.itertuples():
            rows.append(dict(target=target, curr_bin=r.curr_bin,
                             seasonal=r.closure_ratio_seasonal, all_channels=r.closure_ratio_all,
                             valid_only=r.closure_ratio_valid,
                             dJ_all=r.delta_j_actual_all, dJ_valid=r.delta_j_actual_valid,
                             I_all=r.fsoi_raw_sampled_all, I_valid=r.fsoi_raw_sampled_valid,
                             sign_valid=r.sign_agreement_valid))

d = pd.DataFrame(rows)
if d.empty:
    raise SystemExit("No completed pairs yet")
d["drift"] = (d.all_channels - d.seasonal).abs()
d["shift"] = d.valid_only - d.all_channels
d["dJ_change_pct"] = 100 * (d.dJ_valid - d.dJ_all) / d.dJ_all.abs()
d["I_change_pct"] = 100 * (d.I_valid - d.I_all) / d.I_all.abs()
pd.set_option("display.width", 200)
print(d.round(4).to_string(index=False))
print("\nPer network (medians):")
print(d.groupby("target").agg(n=("curr_bin", "size"), closure_all=("all_channels", "median"),
                              closure_valid=("valid_only", "median"),
                              median_abs_shift=("shift", lambda s: s.abs().median()),
                              max_abs_shift=("shift", lambda s: s.abs().max()),
                              dJ_change_pct=("dJ_change_pct", "median"),
                              max_drift=("drift", "max")).round(4).to_string())
print(f"\nSign agreement under valid_only: {int(d.sign_valid.sum())} of {len(d)}")
print("max_drift is all_channels rerun vs seasonal run; it should be ~0 (reproducibility).")
