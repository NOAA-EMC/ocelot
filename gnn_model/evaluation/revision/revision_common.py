"""Shared helpers for the AIES revision evaluation scripts.

Reads OCELOT obs-space prediction CSVs (``pred_<inst>_target_init_<YYYYMMDDHH>.csv``)
and defines the stratification keys used by the observation-space climatology.

Climatology keys (coarsest fallback last):
    L1: instrument, variable, grid cell, month, 3-h UTC bin, pressure level, scan-angle bin
    L2: instrument, variable, grid cell, month,              pressure level, scan-angle bin
    L3: instrument, variable, 5-degree latitude band, month, pressure level, scan-angle bin
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

CONVENTIONAL = {"surface_obs", "radiosonde", "aircraft"}
FILE_RE = re.compile(r"pred_(?P<inst>.+)_target_init_(?P<init>\d{10})\.csv$")
LAT_BAND_DEG = 5.0
N_SCAN_BINS = 8


def cell_deg_for(inst: str) -> float:
    """Grid-cell size: 1 deg for conventional networks, 2 deg for satellites."""
    return 1.0 if inst in CONVENTIONAL else 2.0


def list_prediction_files(pred_dir: str, recursive: bool = False) -> list[tuple[str, str, str]]:
    """Return (path, instrument, init) for every obs-space prediction CSV."""
    pattern = os.path.join(pred_dir, "**", "pred_*_target_init_*.csv") if recursive else os.path.join(
        pred_dir, "pred_*_target_init_*.csv"
    )
    out = []
    for path in sorted(glob.glob(pattern, recursive=recursive)):
        m = FILE_RE.search(os.path.basename(path))
        if m:  # excludes *_vs_gfs.csv and other derived files
            out.append((path, m.group("inst"), m.group("init")))
    return out


def variables_in(df: pd.DataFrame) -> list[str]:
    return [c[len("true_"):] for c in df.columns if c.startswith("true_") and f"pred_{c[len('true_'):]}" in df.columns]


def valid_rows(df: pd.DataFrame, var: str) -> np.ndarray:
    ok = np.isfinite(df[f"pred_{var}"].to_numpy(float)) & np.isfinite(df[f"true_{var}"].to_numpy(float))
    mcol = f"mask_{var}"
    if mcol in df.columns:
        m = df[mcol]
        if m.dtype != bool:
            m = m.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
        ok &= m.to_numpy(bool)
    return ok


def _time_unix(df: pd.DataFrame) -> np.ndarray:
    t = np.full(len(df), -1, dtype=np.int64)
    for col in ("obs_time_unix", "valid_time_unix"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").fillna(-1).to_numpy(np.int64)
            t = np.where(t >= 0, t, v)
    return t


def scan_bin(df: pd.DataFrame, edges: np.ndarray | None) -> np.ndarray:
    if edges is None or "scan_angle_0" not in df.columns:
        return np.full(len(df), -1, dtype=np.int16)
    sa = np.abs(pd.to_numeric(df["scan_angle_0"], errors="coerce").to_numpy(float))
    b = np.clip(np.searchsorted(edges, sa, side="right") - 1, 0, len(edges) - 2)
    return np.where(np.isfinite(sa), b, -1).astype(np.int16)


def strat_keys(df: pd.DataFrame, inst: str, scan_edges: np.ndarray | None) -> pd.DataFrame:
    """Per-row climatology keys for one prediction CSV."""
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(float)
    lon = np.mod(pd.to_numeric(df["lon"], errors="coerce").to_numpy(float) + 180.0, 360.0)
    d = cell_deg_for(inst)
    nlon = int(round(360.0 / d))
    lat_i = np.clip(np.floor((lat + 90.0) / d), 0, int(round(180.0 / d)) - 1)
    lon_i = np.clip(np.floor(lon / d), 0, nlon - 1)
    t = pd.to_datetime(_time_unix(df), unit="s", utc=True)
    plev = (
        pd.to_numeric(df["pressure_level_idx"], errors="coerce").fillna(-1).to_numpy(np.int16)
        if "pressure_level_idx" in df.columns
        else np.full(len(df), -1, dtype=np.int16)
    )
    return pd.DataFrame(
        {
            "cell": (lat_i * nlon + lon_i).astype(np.int32),
            "band": np.clip(np.floor((lat + 90.0) / LAT_BAND_DEG), 0, 180 / LAT_BAND_DEG - 1).astype(np.int16),
            "month": t.month.to_numpy(np.int8),
            "hour3": (t.hour.to_numpy() // 3).astype(np.int8),
            "plev": plev,
            "sab": scan_bin(df, scan_edges),
        }
    )


LEVELS = {
    "L1": ["cell", "month", "hour3", "plev", "sab"],
    "L2": ["cell", "month", "plev", "sab"],
    "L3": ["band", "month", "plev", "sab"],
}
