#!/usr/bin/env python
"""Build an observation-space climatology from TRAINING-period observations.

Input: obs-space prediction CSVs written by ``predict_gnn.py --eval-mode`` over sampled
2015-2023 dates (see run_climatology_dump.sh). Only the ``true_*`` / ``mask_*`` columns
are used, so the climatology passes through exactly the QC, subsampling and inverse
normalization used for the 2025 verification, and it never touches 2024/2025 data.

Output (per instrument): ``<out_dir>/clim_<inst>.csv.gz`` with the stratified means at the
three fallback levels defined in revision_common.LEVELS, plus ``scan_edges.json``.

Usage:
    python evaluation/revision/build_obs_climatology.py \
        --pred_dir predictions/clim_dump_2015_2023/pred_csv/obs-space --recursive \
        --out_dir evaluation/revision/climatology
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import CONVENTIONAL, LEVELS, N_SCAN_BINS, VerifyQC, list_prediction_files, strat_keys, valid_rows, variables_in  # noqa: E402


def _scan_edges(files_by_inst: dict[str, list[str]], max_files: int = 60) -> dict[str, list[float]]:
    edges = {}
    for inst, paths in files_by_inst.items():
        vmax = 0.0
        for p in paths[:: max(1, len(paths) // max_files)]:
            try:
                sa = pd.read_csv(p, usecols=["scan_angle_0"])["scan_angle_0"].to_numpy(float)
            except (ValueError, KeyError):
                break  # instrument has no scan-angle column
            sa = np.abs(sa[np.isfinite(sa)])
            if sa.size:
                vmax = max(vmax, float(np.nanpercentile(sa, 99.9)))
        if vmax > 0:
            edges[inst] = np.linspace(0.0, vmax * 1.0001, N_SCAN_BINS + 1).tolist()
    return edges


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--max_year", type=int, default=2023, help="Refuse files with init year > max_year (leakage guard)")
    ap.add_argument("--consolidate_every", type=int, default=40)
    ap.add_argument("--sat_value_range", type=float, nargs=2, default=None, metavar=("LO", "HI"),
                    help="Only use satellite brightness temperatures within [LO, HI] K (e.g. 50 400); "
                         "keep this consistent with verify_qc.yaml")
    ap.add_argument("--range_exclude", default="ascat", help="Comma-separated satellite instruments exempt from the range")
    ap.add_argument("--sat_range_per_instrument", default=None, metavar="INST=LO:HI,...",
                    help="Per-instrument override of --sat_value_range, e.g. ssmis=50:330")
    ap.add_argument("--instruments", default=None,
                    help="Comma-separated subset to (re)build. NOTE: scan_edges.json in out_dir is rewritten "
                         "for this subset only, so build a subset into a separate out_dir and copy its clim files")
    ap.add_argument("--verify_qc", default=None,
                    help="Apply the conventional rules of this verify_qc.yaml (e.g. max_level_offset) to the "
                         "climatology, so the reference uses the same sample as verification")
    args = ap.parse_args()

    files = list_prediction_files(args.pred_dir, args.recursive)
    bad = [f for f in files if int(f[2][:4]) > args.max_year]
    if bad:
        raise SystemExit(f"{len(bad)} files have init year > {args.max_year} (e.g. {bad[0][0]}); refusing to build climatology.")
    if not files:
        raise SystemExit(f"No prediction CSVs under {args.pred_dir}")

    per_inst = {}
    if args.sat_range_per_instrument:
        for item in args.sat_range_per_instrument.split(","):
            name, rng = item.split("=")
            per_inst[name.strip()] = [float(x) for x in rng.split(":")]
    only = set(args.instruments.split(",")) if args.instruments else None
    qc = VerifyQC(args.verify_qc) if args.verify_qc else None

    by_inst: dict[str, list[str]] = defaultdict(list)
    for path, inst, _ in files:
        if only is None or inst in only:
            by_inst[inst].append(path)
    os.makedirs(args.out_dir, exist_ok=True)

    scan_edges = _scan_edges(by_inst)
    with open(os.path.join(args.out_dir, "scan_edges.json"), "w") as f:
        json.dump(scan_edges, f, indent=1)

    for inst, paths in sorted(by_inst.items()):
        edges = np.asarray(scan_edges[inst]) if inst in scan_edges else None
        parts: dict[str, list[pd.DataFrame]] = {lvl: [] for lvl in LEVELS}

        def _consolidate():
            for lvl, cols in LEVELS.items():
                if len(parts[lvl]) > 1:
                    parts[lvl] = [pd.concat(parts[lvl]).groupby(["var"] + cols, as_index=False)[["sum", "count"]].sum()]

        for i, p in enumerate(paths):
            df = pd.read_csv(p, low_memory=False)
            if df.empty:
                continue
            keys = strat_keys(df, inst, edges)
            for var in variables_in(df):
                ok = valid_rows(df, var)
                rng = per_inst.get(inst, args.sat_value_range)
                if rng and inst not in CONVENTIONAL and inst not in args.range_exclude.split(","):
                    obs = df[f"true_{var}"].to_numpy(float)
                    ok &= (obs >= rng[0]) & (obs <= rng[1])
                if qc is not None and inst in CONVENTIONAL:
                    ok, _ = qc.apply(df, inst, var, ok)
                if not ok.any():
                    continue
                sub = keys[ok].copy()
                sub["var"] = var
                sub["sum"] = df.loc[ok, f"true_{var}"].to_numpy(float)
                sub["count"] = 1
                for lvl, cols in LEVELS.items():
                    parts[lvl].append(sub.groupby(["var"] + cols, as_index=False)[["sum", "count"]].sum())
            if (i + 1) % args.consolidate_every == 0:
                _consolidate()
                print(f"[{inst}] {i + 1}/{len(paths)} files", flush=True)
        _consolidate()

        out = []
        for lvl in LEVELS:
            if parts[lvl]:
                t = parts[lvl][0]
                t["mean"] = t["sum"] / t["count"]
                t.insert(0, "level", lvl)
                out.append(t.drop(columns="sum"))
        if out:
            path = os.path.join(args.out_dir, f"clim_{inst}.csv.gz")
            pd.concat(out, ignore_index=True).to_csv(path, index=False)
            print(f"[{inst}] wrote {path} from {len(paths)} files")
    return 0


if __name__ == "__main__":
    sys.exit(main())
