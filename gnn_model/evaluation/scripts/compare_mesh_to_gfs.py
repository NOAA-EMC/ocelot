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


def _surface_pressure_pred_col(df: pd.DataFrame) -> str | None:
    for col in (
        "pred_airPressure_prepbufr_event_1",
        "pred_airPressure",
        "pred_pressureMeanSeaLevel_prepbufr",
    ):
        if col in df.columns:
            return col
    return None


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


def _open_gfs_mean_sea_pressure(path: str):
    errors: list[str] = []
    for short_name in ("prmsl", "msl"):
        try:
            return _open_cfgrib(path, type_of_level="meanSea", level=None, short_name=short_name)
        except Exception as exc:
            errors.append(f"{short_name}: {type(exc).__name__}: {exc}")
    raise RuntimeError(
        "Could not open GFS mean sea-level pressure (meanSea/prmsl or meanSea/msl). "
        + " | ".join(errors)
    )


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


def _get_unique_pressure_hpa(df: pd.DataFrame) -> int:
    if "pressure_hPa" not in df.columns:
        raise ValueError("mesh CSV for radiosonde/aircraft must include pressure_hPa")
    p = pd.to_numeric(df["pressure_hPa"], errors="coerce")
    p = p[np.isfinite(p.to_numpy(dtype=np.float64, na_value=np.nan))]
    if p.empty:
        raise ValueError("pressure_hPa is all-NaN; cannot select isobaric GFS level")
    # Mesh outputs are generated at a single configured level; enforce that.
    p0 = float(p.iloc[0])
    if not np.isfinite(p0):
        raise ValueError("pressure_hPa first value is not finite")
    if (np.abs(p.to_numpy(dtype=np.float64) - p0) > 1e-6).any():
        raise ValueError("mesh CSV contains multiple pressure_hPa values; expected a single fixed level")
    return int(round(p0))


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
    sp_pred_col = _surface_pressure_pred_col(df)
    need_sp = sp_pred_col is not None

    gfs_u = gfs_v = gfs_t2m_c = gfs_mslp_hpa = None

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
        ds = _open_gfs_mean_sea_pressure(gfs_path)
        gfs_mslp_hpa = _interp_2d(ds, _pick_var_name(ds, "prmsl"), lat, lon360, method=method) / 100.0

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
    if gfs_mslp_hpa is not None:
        out["gfs_mslp_hPa"] = gfs_mslp_hpa
        out["ocelot_minus_gfs_sp"] = out[sp_pred_col] - out["gfs_mslp_hPa"]

    out["init_time"] = init
    out["fhr"] = fhr
    return out


def add_gfs_to_isobaric_mesh(mesh_csv: str, gfs_root: str, method: str) -> pd.DataFrame:
    """Add GFS (isobaric) fields to a radiosonde/aircraft mesh-grid prediction CSV."""

    df = pd.read_csv(mesh_csv)
    if not {"lat", "lon"}.issubset(df.columns):
        raise ValueError("mesh CSV must include lat/lon")
    if "mesh_idx" not in df.columns:
        raise ValueError(
            "mesh CSV must include mesh_idx for strict pointwise alignment. "
            "Re-run prediction with updated code that writes mesh_idx."
        )

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

    level_hpa = _get_unique_pressure_hpa(df)

    lat = df["lat"].to_numpy(dtype=np.float64)
    lon360 = _lon_to_360(df["lon"].to_numpy(dtype=np.float64))

    need_u = "pred_wind_u" in df.columns
    need_v = "pred_wind_v" in df.columns
    need_t = "pred_airTemperature" in df.columns

    gfs_u = gfs_v = gfs_t_c = None

    if need_u:
        ds = _open_cfgrib(gfs_path, type_of_level="isobaricInhPa", level=level_hpa, short_name="u")
        gfs_u = _interp_2d(ds, _pick_var_name(ds, "u"), lat, lon360, method=method)
    if need_v:
        ds = _open_cfgrib(gfs_path, type_of_level="isobaricInhPa", level=level_hpa, short_name="v")
        gfs_v = _interp_2d(ds, _pick_var_name(ds, "v"), lat, lon360, method=method)
    if need_t:
        ds = _open_cfgrib(gfs_path, type_of_level="isobaricInhPa", level=level_hpa, short_name="t")
        gfs_t_c = _interp_2d(ds, _pick_var_name(ds, "t"), lat, lon360, method=method) - 273.15

    out = df.copy()
    out["gfs_level_hPa"] = float(level_hpa)
    if gfs_u is not None:
        out["gfs_u"] = gfs_u
        out["ocelot_minus_gfs_u"] = out["pred_wind_u"] - out["gfs_u"]
    if gfs_v is not None:
        out["gfs_v"] = gfs_v
        out["ocelot_minus_gfs_v"] = out["pred_wind_v"] - out["gfs_v"]
    if gfs_t_c is not None:
        out["gfs_airTemperature_C"] = gfs_t_c
        out["ocelot_minus_gfs_airTemperature"] = out["pred_airTemperature"] - out["gfs_airTemperature_C"]

    out["init_time"] = init
    out["fhr"] = fhr
    out["instrument"] = inst
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh_csv", required=True, help="Mesh-grid prediction CSV, e.g. surface_obs_init_YYYYMMDDHH_f003.csv")
    ap.add_argument("--gfs_root", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--interp", default="nearest", choices=["nearest", "linear"])
    args = ap.parse_args()

    inst, _init, _fhr = _parse_mesh_filename(args.mesh_csv)
    if inst in {"surface_obs", "surface"}:
        out = add_gfs_to_surface_mesh(args.mesh_csv, gfs_root=args.gfs_root, method=args.interp)
    elif inst in {"radiosonde", "aircraft"}:
        out = add_gfs_to_isobaric_mesh(args.mesh_csv, gfs_root=args.gfs_root, method=args.interp)
    else:
        raise SystemExit(
            f"Unsupported mesh instrument={inst!r} in {os.path.basename(args.mesh_csv)}. "
            "Supported: surface_obs, radiosonde."
        )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv} (rows={len(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
