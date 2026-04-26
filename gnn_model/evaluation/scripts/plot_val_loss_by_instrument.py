#!/usr/bin/env python3
"""Plot per-instrument val_loss curves from Lightning CSVLogger metrics.csv.

Author: Azadeh Gholoubi

This repo logs metrics at different times within an epoch, so a single epoch can
span multiple rows. This script reconstructs an "epoch snapshot" by taking the
last non-NaN value for each metric within each epoch.

Usage:
    # Single file
    # (from gnn_model/)
    python evaluation/scripts/plot_val_loss_by_instrument.py --metrics logs/<run>/version_0/metrics.csv

    # All versions (resubmits) via glob
    python evaluation/scripts/plot_val_loss_by_instrument.py --metrics 'logs/<run>/version_*/metrics.csv'

    # Or pass the run directory and it will auto-discover version_*/metrics.csv
    python evaluation/scripts/plot_val_loss_by_instrument.py --metrics logs/<run>/

Outputs:
  - PNG file (default: alongside metrics.csv)
  - Console summary: best epoch, best val_loss, and per-instrument values
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
import re
from glob import glob

import numpy as np
import pandas as pd


def _reexec_under_micromamba_if_needed() -> None:
    """Re-exec this script under the stable micromamba env if needed.

    This lets you run the script even if your current `python` points to a broken
    or incomplete environment.
    """

    if os.environ.get("OCELOT_SKIP_MICROMAMBA_REEXEC") == "1":
        return
    if os.environ.get("OCELOT_IN_MICROMAMBA") == "1":
        return

    env_home = os.environ.get(
        "OCELOT_ENV_HOME",
        "/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/ocelot_env",
    )
    mm = os.environ.get(
        "OCELOT_MM",
        os.path.join(env_home, "micromamba", "bin", "micromamba"),
    )
    root_prefix = os.environ.get(
        "MAMBA_ROOT_PREFIX",
        os.path.join(env_home, "micromamba_root"),
    )
    env_name = os.environ.get("OCELOT_ENV_NAME", "ocelot-cu121")

    if not (os.path.exists(mm) and os.access(mm, os.X_OK)):
        return

    new_env = os.environ.copy()
    new_env["MAMBA_ROOT_PREFIX"] = root_prefix
    new_env["OCELOT_IN_MICROMAMBA"] = "1"
    cmd = [mm, "run", "-n", env_name, "python"] + sys.argv
    os.execvpe(mm, cmd, new_env)


_reexec_under_micromamba_if_needed()


def _last_valid_per_epoch(df: pd.DataFrame, metric: str) -> pd.Series:
    if metric not in df.columns:
        return pd.Series(dtype=float)

    def _last_non_nan(s: pd.Series) -> float:
        s = s.dropna()
        return float(s.iloc[-1]) if len(s) else np.nan

    # df is expected to be sorted in chronological order.
    return df.groupby("epoch", sort=False)[metric].apply(_last_non_nan)


def _expand_metrics_inputs(raw: list[str]) -> list[Path]:
    paths: list[Path] = []

    def _add_matches(pattern: str) -> None:
        for m in sorted(glob(pattern)):
            paths.append(Path(m))

    for item in raw:
        p = Path(item)
        # Directory -> auto-discover
        if p.exists() and p.is_dir():
            _add_matches(str(p / "version_*" / "metrics.csv"))
            if (p / "metrics.csv").exists():
                paths.append(p / "metrics.csv")
            continue

        # Wildcards not expanded by shell (e.g. quotes) -> expand ourselves
        if any(ch in item for ch in ["*", "?", "["]):
            _add_matches(item)
            continue

        # Plain file path
        paths.append(p)

    # Filter to existing files + de-dup while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in paths:
        if p.exists() and p.is_file() and p.name == "metrics.csv" and p not in seen:
            out.append(p)
            seen.add(p)

    return out


def _infer_run_dir(metrics_paths: list[Path]) -> Path:
    # Prefer logs/<run>/ as output when given logs/<run>/version_*/metrics.csv
    candidates: list[Path] = []
    for mp in metrics_paths:
        parent = mp.parent
        if parent.name.startswith("version_"):
            candidates.append(parent.parent)
        else:
            candidates.append(parent)

    if not candidates:
        return Path(".")

    # If all same directory, use it.
    first = candidates[0].resolve()
    if all(c.resolve() == first for c in candidates):
        return first

    # Fallback: common path
    try:
        common = Path(os.path.commonpath([str(c.resolve()) for c in candidates]))
        return common
    except Exception:
        return Path(".")


def _version_key(p: Path) -> tuple[int, str]:
    # Sort version_N numerically if present
    m = re.search(r"version_(\d+)", str(p))
    return (int(m.group(1)) if m else 10**9, str(p))


def _load_and_merge(metrics_paths: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for file_idx, mp in enumerate(sorted(metrics_paths, key=_version_key)):
        df = pd.read_csv(mp)
        df["__src"] = str(mp)
        df["__file_idx"] = file_idx
        df["__row_in_file"] = np.arange(len(df), dtype=np.int64)
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True, sort=False)

    if "epoch" not in df_all.columns:
        raise SystemExit("metrics.csv missing required column: epoch")

    df_all["epoch"] = pd.to_numeric(df_all["epoch"], errors="coerce")
    if "step" in df_all.columns:
        df_all["step"] = pd.to_numeric(df_all["step"], errors="coerce")
    else:
        df_all["step"] = np.nan

    # Create a robust chronological sort key:
    # - Use global_step (step) when present
    # - Otherwise fall back to epoch + file order
    step = df_all["step"].to_numpy(dtype=float)
    fallback = (df_all["epoch"].fillna(-1).to_numpy(dtype=float) * 1e9) + (df_all["__file_idx"].to_numpy(dtype=float) * 1e6) + df_all[
        "__row_in_file"
    ].to_numpy(dtype=float)
    df_all["__sort"] = np.where(np.isfinite(step), step, fallback)

    df_all = df_all.sort_values(["__sort", "epoch", "__file_idx", "__row_in_file"], kind="mergesort")

    # NOTE: Do NOT drop duplicates on (epoch, step).
    # Lightning often logs multiple rows with the same (epoch, step) where only one
    # row contains val_loss (and another contains learning_rate/train_loss_epoch).
    # Dropping those would erase val_loss and produce an "empty" plot.
    return df_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        required=True,
        help=(
            "One or more inputs: a metrics.csv file, a glob like logs/<run>/version_*/metrics.csv, "
            "or a run directory logs/<run>/ to auto-discover version_*/metrics.csv"
        ),
    )
    ap.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output PNG path (default: <run_dir>/val_loss_by_instruments.png)",
    )
    ap.add_argument(
        "--include",
        type=str,
        default="val_loss_",
        help="Prefix for per-instrument metric columns (default: val_loss_)",
    )
    args = ap.parse_args()

    metrics_paths = _expand_metrics_inputs(args.metrics)
    if not metrics_paths:
        raise SystemExit(f"No metrics.csv files found from: {args.metrics}")

    run_dir = _infer_run_dir(metrics_paths)
    out_path = Path(args.out) if args.out else run_dir / "val_loss_by_instruments.png"

    df = _load_and_merge(metrics_paths)

    # Global val_loss series
    val_loss = _last_valid_per_epoch(df, "val_loss")
    if val_loss.dropna().empty:
        raise SystemExit("No val_loss values found in metrics.csv")

    best_epoch = int(val_loss.idxmin())
    best_val = float(val_loss.min())

    # Per-instrument metrics: keep node-type losses and aircraft_target (exclude aircraft per-channel)
    per_cols = [c for c in df.columns if c.startswith(args.include) and c != "val_loss"]
    per_cols = [c for c in per_cols if (not c.startswith("val_loss_aircraft_") or c.endswith("_target"))]

    per_epoch = {}
    for c in sorted(per_cols):
        s = _last_valid_per_epoch(df, c)
        if not s.dropna().empty:
            per_epoch[c] = s

    # Snapshot at best epoch for quick diagnosis
    snap = {
        k: float(v.loc[best_epoch])
        for k, v in per_epoch.items()
        if best_epoch in v.index and np.isfinite(v.loc[best_epoch])
    }

    if len(metrics_paths) == 1:
        print(f"[INFO] metrics: {metrics_paths[0]}")
    else:
        print(f"[INFO] metrics inputs: {len(metrics_paths)} files")
        for mp in metrics_paths:
            print(f"  - {mp}")
    print(f"[INFO] best epoch: {best_epoch}  best val_loss: {best_val:.6f}")
    if snap:
        print("[INFO] per-instrument val_loss_* at best epoch:")
        for k, v in sorted(snap.items(), key=lambda kv: kv[1], reverse=True):
            print(f"  {k}: {v:.6f}")

    # Plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable; skipping plot. ({e})")
        return 2

    n = 1 + len(per_epoch)

    # Layout: one column, multiple rows (simple + readable)
    fig_h = max(4.0, 1.8 * n)
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]

    # Global val_loss
    ax0 = axes[0]
    ax0.plot(val_loss.index.values, val_loss.values, label="val_loss", linewidth=2)
    ax0.axvline(best_epoch, linestyle="--", linewidth=1)
    ax0.set_ylabel("val_loss")
    ax0.grid(True, alpha=0.3)
    ax0.legend(loc="best")

    # Per-instrument curves
    for i, (name, series) in enumerate(sorted(per_epoch.items()), start=1):
        ax = axes[i]
        ax.plot(series.index.values, series.values, linewidth=1)
        ax.axvline(best_epoch, linestyle="--", linewidth=1)
        ax.set_ylabel(name)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("epoch")

    fig.suptitle(f"Validation loss by instrument (best epoch={best_epoch}, val_loss={best_val:.6f})")
    fig.tight_layout(rect=[0, 0.0, 1, 0.995])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    print(f"[OK] wrote: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
