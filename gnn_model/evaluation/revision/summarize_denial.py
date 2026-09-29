#!/usr/bin/env python
"""Summarize observation-denial (data-withholding) experiments against the control.

Each experiment directory holds ``per_init_stats.csv`` written by revision_metrics.py for
the same initializations as the control. For every target instrument we report the change
in forecast MSE when a group of observing systems is withheld from the encoder:

    impact = 100 * (MSE_denied / MSE_control - 1)   [%]   (> 0: the withheld data helped)

computed per (variable/channel, lead) and averaged across variables/channels as a geometric
mean of the MSE ratios, so channels with different units and error magnitudes weigh
equally. 95% CIs come from a paired bootstrap over the common initializations.

Usage:
    python evaluation/revision/summarize_denial.py \
        --control evaluation/revision/results/v1_2025 \
        --exp no_mw_sounders=evaluation/revision/results/deny_mw_sounders ... \
        --out evaluation/revision/results/denial_summary.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


def _load(d: str) -> pd.DataFrame:
    s = pd.read_csv(os.path.join(d, "per_init_stats.csv"), dtype={"init": str})
    return s[s["plev"] == "all"][["instrument", "variable", "lead", "init", "n", "sse"]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--control", required=True)
    ap.add_argument("--exp", action="append", required=True, help="name=dir (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--by_lead", action="store_true", help="Report each lead separately (default: pooled over leads)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()

    ctl = _load(args.control)
    rng = np.random.default_rng(args.seed)
    rows = []
    for spec in args.exp:
        name, d = spec.split("=", 1)
        exp = _load(d)
        m = ctl.merge(exp, on=["instrument", "variable", "lead", "init"], suffixes=("_c", "_d"))
        # Denial removes inputs, not targets, so target counts must match exactly.
        m = m[m["n_c"] == m["n_d"]]
        lead_key = ["lead"] if args.by_lead else []
        for key, g in m.groupby(["instrument"] + lead_key):
            inits = np.sort(g["init"].unique())
            idx = {v: i for i, v in enumerate(inits)}
            W = rng.multinomial(len(inits), np.full(len(inits), 1 / len(inits)), size=args.n_boot).astype(float)
            log_ratios, log_ratios_b = [], []
            for _, gv in g.groupby(["variable", "lead"]):
                ii = gv["init"].map(idx).to_numpy()
                sc = np.bincount(ii, gv["sse_c"].to_numpy(float), len(inits))
                sd = np.bincount(ii, gv["sse_d"].to_numpy(float), len(inits))
                if sc.sum() <= 0:
                    continue
                log_ratios.append(np.log(sd.sum() / sc.sum()))
                with np.errstate(divide="ignore", invalid="ignore"):
                    log_ratios_b.append(np.log((W @ sd) / (W @ sc)))
            if not log_ratios:
                continue
            est = 100 * (np.exp(np.mean(log_ratios)) - 1)
            boot = 100 * (np.exp(np.nanmean(np.vstack(log_ratios_b), axis=0)) - 1)
            key = key if isinstance(key, tuple) else (key,)
            rec = dict(experiment=name, target_instrument=key[0])
            if args.by_lead:
                rec["lead"] = key[1]
            rec.update(n_inits=len(inits), n_series=len(log_ratios), mse_change_pct=est,
                       ci_lo=np.nanpercentile(boot, 2.5), ci_hi=np.nanpercentile(boot, 97.5))
            rows.append(rec)

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, float_format="%.3f")
    piv = out.pivot_table(index="experiment", columns="target_instrument", values="mse_change_pct")
    piv.to_csv(os.path.splitext(args.out)[0] + "_matrix.csv", float_format="%.2f")
    print(piv.round(1).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
