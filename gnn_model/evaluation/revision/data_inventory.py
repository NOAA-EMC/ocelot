#!/usr/bin/env python
"""Dataset-size inventory for the manuscript (Reviewer 2, minor comment on Sec. 3.1).

Per instrument and split (train 2015-2023 / val 2024 / test 2025) it reports the number of
archived observation locations (rows), the fraction passing the configured satellite-ID
filter (estimated from sampled blocks), the approximate number actually sampled by the
OCELOT pipeline (after the random subsampling stride), and optionally the archive size.
It also prints the checkpoint epoch and trainable-parameter count.

Split counts use binary search on the (sorted) Zarr time array, so the whole run takes
minutes; unsorted arrays fall back to a chunked scan. Results are written after each
instrument.

Usage (CPU, from gnn_model/):
    python evaluation/revision/data_inventory.py --data_path /scratch4/.../ocelot/data/v7 \
        --ckpt /scratch3/NCEPDEV/da/Azadeh.Gholoubi/PaperCheckpoint/Epoch3079.ckpt --out evaluation/revision/results/data_inventory.csv
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
import zarr

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from gnn_datamodule import _resolve_zarr_path  # noqa: E402
from process_timeseries import _sampled_non_decreasing, _zarr_bisect_left  # noqa: E402

SPLITS = {"train": ("2015-01-01", "2024-01-01"), "val": ("2024-01-01", "2025-01-01"), "test": ("2025-01-01", "2026-01-01")}


def _stride(pipeline: dict, obs_type: str, inst: str) -> int:
    sub = ((pipeline or {}).get("subsample") or {}).get(obs_type, {}) or {}
    return int(sub.get(inst, sub.get("_default", 1)))


def _du(path: str) -> str:
    try:
        return subprocess.run(["du", "-sh", path], capture_output=True, text=True, timeout=1800).stdout.split()[0]
    except Exception:
        return "?"


class _Indexable:
    """len()/scalar-indexing view of a Zarr array (zarr v3 arrays have no len())."""

    def __init__(self, arr):
        self.arr = arr

    def __len__(self):
        return int(self.arr.shape[0])

    def __getitem__(self, i):
        return self.arr[i]


def _split_bounds(t, bounds, chunk):
    """Return {split: (lo, hi)} row ranges, or None if the time array is not sorted."""
    ti = _Indexable(t)
    if not _sampled_non_decreasing(ti, n_checks=64):
        return None
    return {k: (_zarr_bisect_left(ti, a), _zarr_bisect_left(ti, b)) for k, (a, b) in bounds.items()}


def _scan_counts(t, bounds, chunk):
    counts = {k: 0 for k in bounds}
    for i0 in range(0, t.shape[0], chunk):
        tt = t[i0:i0 + chunk]
        for k, (a, b) in bounds.items():
            counts[k] += int(((tt >= a) & (tt < b)).sum())
    return counts


def _sat_keep_fraction(z, sat_field, sat_ids, lo, hi, rng, n_blocks=20, block=200_000):
    """Fraction of rows in [lo, hi) whose satellite ID is in the configured list (sampled)."""
    if sat_ids is None or hi <= lo:
        return 1.0
    kept = total = 0
    for s in rng.integers(lo, max(lo + 1, hi - block), size=n_blocks):
        v = z[sat_field][int(s):int(min(hi, s + block))]
        kept += int(np.isin(v, sat_ids).sum())
        total += v.size
    return kept / total if total else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--cfg_path", default="configs/observation_config.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--du", action="store_true", help="Also report archive size with du (slow on large Zarr stores)")
    ap.add_argument("--chunk", type=int, default=20_000_000)
    args = ap.parse_args()

    if args.ckpt:
        import torch
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        n = sum(v.numel() for k, v in sd.items() if hasattr(v, "numel") and v.is_floating_point() and not k.startswith(("mesh_", "_")))
        print(f"Checkpoint epoch={ck.get('epoch')} global_step={ck.get('global_step')}", flush=True)
        print(f"Parameter count (floating-point state_dict tensors, excl. mesh buffers): {n:,}", flush=True)
        del ck, sd

    for k, (a, b) in SPLITS.items():
        n_win = int((pd.Timestamp(b) - pd.Timestamp(a)) / pd.Timedelta(hours=12))
        print(f"{k}: {n_win} possible 12-h forecast windows (00/12 UTC)", flush=True)

    cfg = yaml.safe_load(open(args.cfg_path))
    bounds = {k: (int(pd.Timestamp(a, tz="UTC").timestamp()), int(pd.Timestamp(b, tz="UTC").timestamp()))
              for k, (a, b) in SPLITS.items()}
    rng = np.random.default_rng(0)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    for obs_type, insts in cfg["observation_config"].items():
        for inst, icfg in insts.items():
            zpath, _ = _resolve_zarr_path(args.data_path, icfg.get("zarr_name", inst), "2015-01-01")
            z = zarr.open(zpath, mode="r")
            t = z["time"]
            sat_field = "satelliteId" if "satelliteId" in z else ("satelliteIdentifier" if "satelliteIdentifier" in z else None)
            sat_ids = np.asarray(icfg.get("sat_ids", [])) if obs_type == "satellite" and sat_field else None
            rng_split = _split_bounds(t, bounds, args.chunk)
            if rng_split is not None:
                counts = {k: hi - lo for k, (lo, hi) in rng_split.items()}
                method = "bisect"
            else:
                counts = _scan_counts(t, bounds, args.chunk)
                method = "scan"
            stride = _stride(cfg.get("pipeline"), obs_type, inst)
            size = _du(zpath) if args.du else ""
            for k in SPLITS:
                lo, hi = rng_split[k] if rng_split is not None else (0, int(t.shape[0]))  # scan: whole-archive estimate
                frac = _sat_keep_fraction(z, sat_field, sat_ids, lo, hi, rng)
                est = int(round(counts[k] * frac)) if np.isfinite(frac) else counts[k]
                rows.append(dict(obs_type=obs_type, instrument=inst, split=k, archived_rows=counts[k],
                                 sat_filter_keep_frac=round(frac, 4) if np.isfinite(frac) else "",
                                 rows_after_sat_filter=est, subsample_stride=stride,
                                 sampled_rows_approx=est // max(stride, 1), n_channels=len(icfg.get("features", [])),
                                 zarr=os.path.basename(zpath), archive_size=size if k == "train" else "", count_method=method))
            pd.DataFrame(rows).to_csv(args.out, index=False)
            print(f"{inst} [{method}]: " + ", ".join(f"{k}={v:,}" for k, v in counts.items())
                  + (f" | size={size}" if size else ""), flush=True)

    df = pd.DataFrame(rows)
    print(df.groupby("split")[["archived_rows", "rows_after_sat_filter", "sampled_rows_approx"]].sum().to_string())
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
