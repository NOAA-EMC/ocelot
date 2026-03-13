#!/usr/bin/env python
"""Interpolate GFS onto OCELOT mesh-grid prediction points.

Author: Azadeh Gholoubi

Mesh-grid prediction CSVs are produced when `enable_mesh_pred: true` and should look like:
  <mesh_dir>/<instrument>_init_<YYYYMMDDHH>_f<FFF>.csv

This script reads those mesh CSVs and adds `gfs_*` columns by interpolating the
corresponding GFS GRIB (0.25°) to the mesh lat/lon points.

Output is a CSV per lead time:
    <out_dir>/<instrument>_init_<YYYYMMDDHH>_f<FFF>_gfs_on_ocelot_mesh.csv

Notes
-----
- Mesh-grid outputs are *predictions only* (no truth). This script therefore
  compares OCELOT vs GFS on the same grid/points.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _lon_to_360(lon_deg: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_deg, dtype=np.float64)
    lon = np.mod(lon, 360.0)
    lon = np.where(lon >= 360.0, lon - 360.0, lon)
    return lon


@dataclass(frozen=True)
class GfsKey:
    init_ymdh: str  # YYYYMMDDHH
    fhr: int


def _gfs_path(gfs_root: str, key: GfsKey) -> str:
    ymd = key.init_ymdh[:8]
    hh = key.init_ymdh[8:10]
    return os.path.join(gfs_root, ymd, f"gfs.{ymd}.t{hh}z.pgrb2.0p25.f{key.fhr:03d}")


def _open_cfgrib(path: str, *, type_of_level: str, level: int | None, short_name: str):
    import cfgrib

    keys = {"typeOfLevel": type_of_level, "shortName": short_name}
    if level is not None:
        keys["level"] = int(level)

    return cfgrib.open_dataset(path, indexpath="", filter_by_keys=keys)


def _interp_2d(ds, var: str, lat: np.ndarray, lon360: np.ndarray, method: str) -> np.ndarray:
    import xarray as xr

    lat_da = xr.DataArray(lat.astype(np.float64), dims="points")
    lon_da = xr.DataArray(lon360.astype(np.float64), dims="points")
    out = ds[var].interp(latitude=lat_da, longitude=lon_da, method=method)
    return out.values.astype(np.float64)


def _pick_var_name(ds, preferred: str) -> str:
    if preferred in getattr(ds, "data_vars", {}):
        return preferred
    vars_ = list(getattr(ds, "data_vars", {}).keys())
    if len(vars_) == 1:
        return vars_[0]
    raise KeyError(f"Variable {preferred!r} not found; dataset has: {vars_}")


_MESH_FNAME_RE = re.compile(r"^(?P<inst>.+?)_init_(?P<init>\d{10})_f(?P<fhr>\d{3})\.csv$")


def _parse_mesh_filename(path: str) -> tuple[str, str, int]:
    m = _MESH_FNAME_RE.match(os.path.basename(path))
    if not m:
        raise ValueError(f"Unexpected mesh CSV filename: {os.path.basename(path)}")
    inst = m.group("inst")
    init = m.group("init")
    fhr = int(m.group("fhr"))
    return inst, init, fhr


def add_gfs_to_surface_mesh(mesh_csv: str, gfs_root: str, method: str) -> pd.DataFrame:
    df = pd.read_csv(mesh_csv)
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("mesh CSV must include lat/lon")
    if "mesh_idx" not in df.columns:
        raise ValueError(
            "mesh CSV must include mesh_idx for strict pointwise alignment. "
            "Re-run prediction with updated code that writes mesh_idx."
        )

    # Parse/validate mesh_idx early so downstream consumers can merge safely.
    mesh_idx = pd.to_numeric(df["mesh_idx"], errors="coerce").astype("Int64")
    if mesh_idx.isna().any():
        raise ValueError("Found NaN mesh_idx after parsing; cannot align reliably.")
    if mesh_idx.duplicated().any():
        raise ValueError("Found duplicate mesh_idx values; cannot align reliably.")
    df = df.copy()
    df["mesh_idx"] = mesh_idx.astype(np.int64)

    inst, init, fhr = _parse_mesh_filename(mesh_csv)
    key = GfsKey(init_ymdh=init, fhr=fhr)
    gfs_path = _gfs_path(gfs_root, key)
    if not os.path.exists(gfs_path):
        raise FileNotFoundError(f"Missing GFS file: {gfs_path}")

    lat = df["lat"].to_numpy(dtype=np.float64)
    lon360 = _lon_to_360(df["lon"].to_numpy(dtype=np.float64))

    # Open required fields (only if the pred_* columns exist)
    need_u = "pred_wind_u" in df.columns
    need_v = "pred_wind_v" in df.columns
    need_t = "pred_airTemperature" in df.columns
    need_sp = "pred_airPressure_prepbufr_event_1" in df.columns or "pred_airPressure" in df.columns

    gfs_u = gfs_v = gfs_t2m_c = gfs_sp_hpa = None

    if need_u:
        ds = _open_cfgrib(gfs_path, type_of_level="heightAboveGround", level=10, short_name="10u")
        gfs_u = _interp_2d(ds, _pick_var_name(ds, "u10"), lat, lon360, method=method)
    if need_v:
        ds = _open_cfgrib(gfs_path, type_of_level="heightAboveGround", level=10, short_name="10v")
        gfs_v = _interp_2d(ds, _pick_var_name(ds, "v10"), lat, lon360, method=method)
    if need_t:
        ds = _open_cfgrib(gfs_path, type_of_level="heightAboveGround", level=2, short_name="2t")
        gfs_t2m_c = _interp_2d(ds, _pick_var_name(ds, "t2m"), lat, lon360, method=method) - 273.15
    if need_sp:
        ds = _open_cfgrib(gfs_path, type_of_level="surface", level=None, short_name="sp")
        gfs_sp_hpa = _interp_2d(ds, _pick_var_name(ds, "sp"), lat, lon360, method=method) / 100.0

    out = df.copy()
    if gfs_u is not None:
        out["gfs_u10"] = gfs_u
        out["ocelot_minus_gfs_u10"] = out["pred_wind_u"] - out["gfs_u10"]
    if gfs_v is not None:
        out["gfs_v10"] = gfs_v
        out["ocelot_minus_gfs_v10"] = out["pred_wind_v"] - out["gfs_v10"]
    if gfs_t2m_c is not None:
        out["gfs_t2m_C"] = gfs_t2m_c
        out["ocelot_minus_gfs_t2m"] = out["pred_airTemperature"] - out["gfs_t2m_C"]
    if gfs_sp_hpa is not None:
        out["gfs_sp_hPa"] = gfs_sp_hpa
        pcol = "pred_airPressure_prepbufr_event_1" if "pred_airPressure_prepbufr_event_1" in out.columns else "pred_airPressure"
        out["ocelot_minus_gfs_sp"] = out[pcol] - out["gfs_sp_hPa"]

    out["init_time"] = init
    out["fhr"] = fhr
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_csv", required=True, help="Mesh-grid prediction CSV, e.g. surface_obs_init_YYYYMMDDHH_f003.csv")
    ap.add_argument("--gfs_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--interp", default="nearest", choices=["nearest", "linear"])
    args = ap.parse_args()

    # For now we only support surface-style mesh fields (u10/v10/t2m/sp).
    out = add_gfs_to_surface_mesh(args.mesh_csv, gfs_root=args.gfs_root, method=args.interp)

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv} (rows={len(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
