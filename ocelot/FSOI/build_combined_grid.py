#!/usr/bin/env python
"""Pool the per-cycle combined-metric 5-degree grids from the 12 map runs.

Reads <root>/<target>_<month>2025/csv/fsoi_combined_by_grid_5deg.csv, drops
cycles valid on the first of the month (as in every reported statistic),
checks each run against its seasonal closure file, and writes
<out>/fsoi_grid_5deg.csv in the layout the map plot reads:
target, instrument, ilat, ilon, fsoi_sum, fsoi_sum_ht, n.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FSOI = Path(__file__).resolve().parent
TARGETS = ("aircraft", "radiosonde", "surface_obs")
MONTHS = ("jan", "apr", "jul", "oct")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=FSOI / "fsoi_outputs" / "seasonal_combined_maps")
    parser.add_argument("--seasonal", type=Path,
                        default=FSOI / "fsoi_outputs" / "seasonal_inclusion_weighted_final")
    parser.add_argument("--out", type=Path, default=FSOI / "fsoi_outputs" / "combined_maps" / "figure_reductions")
    args = parser.parse_args()

    grids, report = [], []
    for target in TARGETS:
        for month in MONTHS:
            run = args.root / f"{target}_{month}2025" / "csv"
            g = pd.read_csv(run / "fsoi_combined_by_grid_5deg.csv")
            g = g[~g.curr_bin.astype(str).str[9:11].eq("01")]
            # Reproducibility: the rerun must match the seasonal closure cycle for cycle.
            new = pd.read_csv(run / "fsoi_combined_closure.csv")[["curr_bin", "closure_ratio"]]
            old = pd.read_csv(args.seasonal / f"{target}_{month}2025" / "csv" / "fsoi_combined_closure.csv")
            m = new.merge(old[["curr_bin", "closure_ratio"]], on="curr_bin", suffixes=("", "_seasonal"))
            drift = float((m.closure_ratio - m.closure_ratio_seasonal).abs().max())
            report.append((target, month, g.curr_bin.nunique(), len(m), drift))
            grids.append(g.assign(target=target))

    r = pd.DataFrame(report, columns=["target", "month", "cycles_in_grid", "cycles_matched", "max_closure_drift"])
    print(r.to_string(index=False))
    if r.max_closure_drift.max() > 1e-6:
        raise SystemExit("A rerun does not reproduce its seasonal run; inspect before plotting.")

    pooled = (pd.concat(grids, ignore_index=True)
              .groupby(["target", "instrument", "ilat", "ilon"], as_index=False)
              [["fsoi_sum", "fsoi_sum_ht", "n_rows"]].sum()
              .rename(columns={"n_rows": "n"}))
    args.out.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.out / "fsoi_grid_5deg.csv", index=False)
    print(f"\nwrote {args.out / 'fsoi_grid_5deg.csv'}: {len(pooled)} cells, "
          f"{r.cycles_in_grid.sum()} cycles, instruments {sorted(pooled.instrument.unique())}")


if __name__ == "__main__":
    main()
