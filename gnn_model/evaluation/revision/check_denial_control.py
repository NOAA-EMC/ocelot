#!/usr/bin/env python
"""Check #2 of README_REVISION: the denial control must reproduce the manuscript forecasts.

The denial experiments express impact as a ratio of denied MSE to control MSE, so the whole
matrix is only meaningful if the control run reproduces the forecasts the manuscript reports.
The control withholds nothing, so on the initializations it shares with the main 2025
evaluation it should give the same errors up to reduced-precision differences.

This compares the two ``per_init_stats.csv`` files on their common initializations and reports,
per target, the relative difference in RMSE. Both runs must have been scored the same way
(the same --verify_qc, or neither), otherwise the sample sizes differ and the comparison is
meaningless; the script checks that first and says so.

Usage (from gnn_model/):
    python evaluation/revision/check_denial_control.py \
        --control evaluation/revision/results/denial/control \
        --reference evaluation/revision/results/v1_2025
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

KEY = ["instrument", "variable", "lead"]


def _load(d: str) -> pd.DataFrame:
    p = os.path.join(d, "per_init_stats.csv")
    if not os.path.exists(p):
        sys.exit(f"missing {p}")
    s = pd.read_csv(p, dtype={"init": str})
    if "plev" in s.columns:
        s = s[s["plev"] == "all"]
    return s[KEY + ["init", "n", "sse"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True, help="results dir of the denial control run")
    ap.add_argument("--reference", required=True, help="results dir of the main 2025 evaluation")
    ap.add_argument("--tol_pct", type=float, default=0.5,
                    help="relative RMSE difference, in percent, treated as reduced-precision noise")
    ap.add_argument("--out", default=None, help="optional CSV of the per-target comparison")
    args = ap.parse_args()

    c, r = _load(args.control), _load(args.reference)
    inits = sorted(set(c["init"]) & set(r["init"]))
    if not inits:
        sys.exit("the two runs share no initializations")
    print(f"control {c['init'].nunique()} inits, reference {r['init'].nunique()} inits, "
          f"{len(inits)} in common")

    c, r = c[c["init"].isin(inits)], r[r["init"].isin(inits)]
    g = (c.groupby(KEY)[["n", "sse"]].sum()
          .join(r.groupby(KEY)[["n", "sse"]].sum(), lsuffix="_ctl", rsuffix="_ref", how="inner")
          .reset_index())
    if g.empty:
        sys.exit("no targets in common between the two runs")

    g["rmse_ctl"] = np.sqrt(g["sse_ctl"] / g["n_ctl"])
    g["rmse_ref"] = np.sqrt(g["sse_ref"] / g["n_ref"])
    g["diff_pct"] = 100.0 * (g["rmse_ctl"] / g["rmse_ref"] - 1.0)
    g["n_match"] = g["n_ctl"] == g["n_ref"]

    mismatched = g[~g["n_match"]]
    if len(mismatched):
        print(f"\nWARNING: {len(mismatched)} of {len(g)} targets have different sample sizes. The two "
              "runs were not scored the same way (check --verify_qc on both). Largest gaps:")
        m = mismatched.assign(gap=(mismatched["n_ctl"] - mismatched["n_ref"]).abs()).nlargest(5, "gap")
        for _, x in m.iterrows():
            print(f"  {x['instrument']:>12s} {x['variable']:<22s} lead {x['lead']:>3} "
                  f"n {int(x['n_ctl']):,} vs {int(x['n_ref']):,}")

    ok = g[g["n_match"]]
    worst = ok.assign(a=ok["diff_pct"].abs()).nlargest(10, "a")
    print(f"\nRelative RMSE difference, control vs reference, on {len(ok)} comparable targets:")
    print(f"  median {ok['diff_pct'].median():+.3f}%   max |diff| {ok['diff_pct'].abs().max():.3f}%")
    print(f"\n{'instrument':>12s} {'variable':<22s} {'lead':>4s} {'rmse_ctl':>10s} {'rmse_ref':>10s} {'diff%':>8s}")
    for _, x in worst.iterrows():
        print(f"{x['instrument']:>12s} {x['variable']:<22s} {str(x['lead']):>4s} "
              f"{x['rmse_ctl']:10.4f} {x['rmse_ref']:10.4f} {x['diff_pct']:+8.3f}")

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        g.to_csv(args.out, index=False)
        print(f"\nwrote {args.out}")

    bad = ok[ok["diff_pct"].abs() > args.tol_pct]
    if len(bad):
        print(f"\nFAIL: {len(bad)} targets differ by more than {args.tol_pct}%. The control does not "
              "reproduce the manuscript forecasts; check the checkpoint and the observation config "
              "before using the denial matrix.")
        return 1
    print(f"\nPASS: every comparable target agrees to within {args.tol_pct}%, so the denial control "
          "reproduces the manuscript forecasts and the impact ratios can be trusted.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
