#!/usr/bin/env python
"""Condense a revision_metrics.py results directory into the blocks used for the manuscript.

  A  per instrument and lead: how many targets beat climatology, median ACC, MSESS range, coherence
  B  how many targets beat climatology at +12 h
  C  the weakest targets at +12 h
  D  the conventional variables and the channels shown in Fig. 2, in full
  E  verification-QC removals, if qc_removal_summary.csv is present

Usage (from gnn_model/):
    python evaluation/revision/summarize_metrics.py --results evaluation/revision/results/v1_2025_qc
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

FIG2 = {("atms", "bt_channel_7"), ("amsua", "bt_channel_7"), ("ssmis", "bt_ch_5")}
CONV = {"surface_obs", "radiosonde", "aircraft", "ascat", "avhrr"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--weak_below", type=float, default=0.15, help="MSESS threshold for block C")
    args = ap.parse_args()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)

    c = pd.read_csv(os.path.join(args.results, "metrics_compact_all_targets.csv"))

    print("=== A. per instrument and lead ===")
    g = c.groupby(["instrument", "lead"])
    print(pd.DataFrame({
        "targets": g.size(),
        "skill>0": g.msess_clim.apply(lambda s: int((s > 0).sum())),
        "inits": g.n_inits.max() if "n_inits" in c else g.size(),
        "ACC_med": g.acc_centered_mean.median().round(2),
        "MSESS_min": g.msess_clim.min().round(2),
        "MSESS_med": g.msess_clim.median().round(2),
        "MSESS_max": g.msess_clim.max().round(2),
        "coh_med": g.error_coherence.median().round(2),
    }).to_string())

    t = c[c.lead == 12]
    print(f"\n=== B. +12 h: {len(t)} targets | beating climatology: {int((t.msess_clim > 0).sum())} ===")

    print(f"\n=== C. weakest targets at +12 h (MSESS_clim < {args.weak_below}) ===")
    w = t[t.msess_clim < args.weak_below][
        ["instrument", "variable", "n_obs", "rmse", "rmse_clim", "acc_centered_mean", "msess_clim"]]
    print(w.to_string(index=False) if len(w) else "   (none)")

    print("\n=== D. conventional variables and the Fig. 2 channels, all leads ===")
    fig2 = pd.Series([(i, v) in FIG2 for i, v in zip(c.instrument, c.variable)], index=c.index)
    keep = c.instrument.isin(CONV) | fig2
    print(c[keep].round(3).to_string(index=False))

    qc = os.path.join(args.results, "qc_removal_summary.csv")
    if os.path.exists(qc):
        q = pd.read_csv(qc)
        q = q[q.pct_removed_total > 0.01].sort_values("pct_removed_total", ascending=False)
        print("\n=== E. verification QC: targets losing more than 0.01% of observations ===")
        cols = [x for x in ("instrument", "variable", "n_before", "pct_range", "pct_flag", "pct_pressure",
                            "pct_relation", "pct_outlier", "pct_removed_total") if x in q.columns]
        print(q[cols].round(3).to_string(index=False) if len(q) else "   (none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
