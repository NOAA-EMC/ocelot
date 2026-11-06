#!/usr/bin/env python
"""Convergence of path-integrated FSOI with 5, 9 and 17 points.

Pools the 5-point cases already reported with the 9- and 17-point reruns produced by
scripts/submit_path_convergence.sh, and reports, for each cycle, the closure ratio and
the absolute residual |I - dJ| at each grid. A case has converged when the residual
stops changing as the grid is refined; a case that keeps moving has not.
"""
import argparse
from pathlib import Path

import pandas as pd

FSOI = Path(__file__).resolve().parent
FIVE = FSOI / "fsoi_outputs" / "ose_frozen_metric_surface"
MONTHS = ("jan", "apr", "jul", "oct")
COLS = ["curr_bin", "delta_j_actual", "matched_fsoi", "matched_closure_ratio",
        "path_integrated_fsoi", "path_closure_ratio"]


def load(path, grid):
    d = pd.read_csv(path)
    d = d[d.path_integrated_fsoi.notna()][COLS].copy()
    d["grid"] = grid
    d["cycle"] = d.curr_bin.astype(str).str[3:]
    d["residual"] = (d.path_integrated_fsoi - d.delta_j_actual).abs()
    d["residual_two_point"] = (d.matched_fsoi - d.delta_j_actual).abs()
    return d


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=FSOI / "fsoi_outputs" / "path_convergence")
    args = parser.parse_args()

    frames = []
    for m in MONTHS:
        five = FIVE / f"ose_seviri_asr_{m}2025_background_replacement" / "evaluation" / "ose_results.csv"
        if five.is_file():
            frames.append(load(five, 5))
        for grid in (9, 17):
            p = args.root / f"seviri_{m}_{grid}pt" / "evaluation" / "ose_results.csv"
            if p.is_file():
                frames.append(load(p, grid))
            else:
                print(f"[missing] {p}")
    if not frames:
        raise SystemExit("No path-integration results found")
    d = pd.concat(frames, ignore_index=True)
    d = d[~d.cycle.str[6:8].eq("01")]          # first-of-month cycles are not scored

    ratio = d.pivot_table(index="cycle", columns="grid", values="path_closure_ratio")
    resid = d.pivot_table(index="cycle", columns="grid", values="residual")
    two = d.groupby("cycle").residual_two_point.first()
    rho2 = d.groupby("cycle").matched_closure_ratio.first()
    dj = d.groupby("cycle").delta_j_actual.first()

    out = pd.DataFrame({"dJ": dj, "rho_2pt": rho2})
    for g in sorted(ratio.columns):
        out[f"rho_{g}pt"] = ratio[g]
    out["resid_2pt"] = two
    for g in sorted(resid.columns):
        out[f"resid_{g}pt"] = resid[g]
    if {9, 17}.issubset(set(resid.columns)):
        out["change_9_to_17"] = (resid[17] - resid[9]).abs() / resid[9]
    pd.set_option("display.width", 220)
    print(out.to_string(float_format=lambda v: f"{v: .4g}"))
    print("\nA case has converged when resid stops changing between 9 and 17 points")
    print("(change_9_to_17 near zero). The three cases outside 0.97-1.05 at five points")
    print("are 2025-04-14, 2025-07-09 and 2025-07-14.")
    dest = args.root / "path_convergence_summary.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest)
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
