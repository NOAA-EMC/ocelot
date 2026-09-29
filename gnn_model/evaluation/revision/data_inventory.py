#!/usr/bin/env python
"""Dataset-size inventory for the manuscript (Reviewer 2, minor comment on Sec. 3.1).

Per instrument and split (train 2015-2023 / val 2024 / test 2025) it reports the number of
archived observations (after the satellite-ID filter), the expected number actually sampled
by the OCELOT pipeline (after the configured random subsampling stride), the number of
12-h forecast windows, and the on-disk archive size. Optionally prints the trainable
parameter count of a checkpoint.

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

SPLITS = {"train": ("2015-01-01", "2024-01-01"), "val": ("2024-01-01", "2025-01-01"), "test": ("2025-01-01", "2026-01-01")}


def _stride(pipeline: dict, obs_type: str, inst: str) -> int:
    sub = ((pipeline or {}).get("subsample") or {}).get(obs_type, {}) or {}
    return int(sub.get(inst, sub.get("_default", 1)))


def _du(path: str) -> str:
    try:
        return subprocess.run(["du", "-sh", path], capture_output=True, text=True, timeout=600).stdout.split()[0]
    except Exception:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--cfg_path", default="configs/observation_config.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunk", type=int, default=20_000_000)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.cfg_path))
    rows = []
    for obs_type, insts in cfg["observation_config"].items():
        for inst, icfg in insts.items():
            zpath, _ = _resolve_zarr_path(args.data_path, icfg.get("zarr_name", inst), "2015-01-01")
            z = zarr.open(zpath, mode="r")
            t = z["time"]
            sat_field = "satelliteId" if "satelliteId" in z else ("satelliteIdentifier" if "satelliteIdentifier" in z else None)
            sat_ids = np.asarray(icfg.get("sat_ids", [])) if obs_type == "satellite" and sat_field else None
            counts = {k: 0 for k in SPLITS}
            bounds = {k: (int(pd.Timestamp(a, tz="UTC").timestamp()), int(pd.Timestamp(b, tz="UTC").timestamp())) for k, (a, b) in SPLITS.items()}
            for i0 in range(0, t.shape[0], args.chunk):
                tt = t[i0:i0 + args.chunk]
                keep = np.ones(tt.shape, bool) if sat_ids is None else np.isin(z[sat_field][i0:i0 + args.chunk], sat_ids)
                for k, (a, b) in bounds.items():
                    counts[k] += int(((tt >= a) & (tt < b) & keep).sum())
            stride = _stride(cfg.get("pipeline"), obs_type, inst)
            for k in SPLITS:
                rows.append(dict(obs_type=obs_type, instrument=inst, split=k, archived_obs=counts[k], subsample_stride=stride,
                                 sampled_obs_approx=counts[k] // max(stride, 1),
                                 n_channels=len(icfg.get("features", [])), zarr=os.path.basename(zpath),
                                 archive_size=_du(zpath) if k == "train" else ""))
            print(f"{inst}: " + ", ".join(f"{k}={v:,}" for k, v in counts.items()), flush=True)

    df = pd.DataFrame(rows)
    for k, (a, b) in SPLITS.items():
        n_win = int((pd.Timestamp(b) - pd.Timestamp(a)) / pd.Timedelta(hours=12))
        print(f"{k}: {n_win} possible 12-h forecast windows (00/12 UTC)")
    tot = df.groupby("split")[["archived_obs", "sampled_obs_approx"]].sum()
    print(tot.to_string())
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)

    if args.ckpt:
        import torch
        ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
        sd = ck.get("state_dict", ck)
        n = sum(v.numel() for k, v in sd.items() if hasattr(v, "numel") and v.is_floating_point() and not k.startswith(("mesh_", "_")))
        print(f"Parameter count (floating-point state_dict tensors, excl. mesh buffers): {n:,}")
        print(f"Checkpoint epoch={ck.get('epoch')} global_step={ck.get('global_step')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
