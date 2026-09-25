#!/usr/bin/env python
"""Summarize the directional checks across metrics, perturbation sizes and supports.

Reads the runs produced by scripts/submit_directional_coverage.sh and reports, for each
verification metric, perturbation size and direction support, the per-instrument relative
error between the automatic-differentiation and finite-difference directional
derivatives, plus the pass criteria used in the manuscript.
"""
import argparse
from pathlib import Path

import pandas as pd

FSOI = Path(__file__).resolve().parent
TARGETS = ("radiosonde", "aircraft", "surface_obs")
SUPPORTS = ("valid_only", "all_entries")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=FSOI / "fsoi_outputs" / "directional_coverage")
    args = parser.parse_args()

    frames = []
    for target in TARGETS:
        for support in SUPPORTS:
            p = args.root / f"{target}_{support}" / "evaluation" / "fd_directional_validation.csv"
            if not p.is_file():
                print(f"[missing] {p}")
                continue
            d = pd.read_csv(p)
            d["metric"] = target
            d["support"] = d.get("direction_support", pd.Series(support, index=d.index)).fillna(support)
            frames.append(d)
    if not frames:
        raise SystemExit("No directional results found")
    d = pd.concat(frames, ignore_index=True)
    d = d[~d.curr_bin.astype(str).str[9:11].eq("01")]

    pd.set_option("display.width", 220)
    print("Per metric, support and perturbation size\n")
    g = d.groupby(["metric", "support", "epsilon"]).agg(
        trials=("rel_error", "size"),
        instruments=("inst_name", "nunique"),
        median_rel_error=("rel_error", "median"),
        max_rel_error=("rel_error", "max"),
        min_pearson=("pearson_r", "min"),
        min_ulp=("n_ulp", "min"),
        all_pass=("status", lambda s: bool((s == "PASS").all())))
    print(g.round(6).to_string())

    print("\nPer instrument (median relative error), metric x epsilon, valid-entry directions\n")
    v = d[d.support.eq("valid_entries") | d.support.eq("valid_only")]
    if not v.empty:
        print(v.pivot_table(index="inst_name", columns=["metric", "epsilon"],
                            values="rel_error", aggfunc="median").round(5).to_string())

    fails = d[d.status.ne("PASS")]
    print(f"\nnon-PASS combinations: {len(fails)}")
    if not fails.empty:
        print(fails[["metric", "support", "epsilon", "inst_name", "curr_bin", "status",
                     "rel_error", "pearson_r", "n_ulp"]].to_string(index=False))
    dest = args.root / "directional_coverage_summary.csv"
    g.to_csv(dest)
    print(f"\nwrote {dest}")
    print("\nPass criteria, as in the manuscript: Pearson r >= 0.99 across the five")
    print("directions, mean relative error < 0.05, expected signal > 2 float32 ULP at J.")


if __name__ == "__main__":
    main()
