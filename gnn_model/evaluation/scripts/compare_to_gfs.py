#!/usr/bin/env python
"""Compare Ocelot conventional-observation predictions vs GFS.

Author: Azadeh Gholoubi

Inputs
------
Ocelot CSV must contain at least:
- `datetime` (ISO8601 UTC). In this repo this may be a nominal valid time.
- `obs_time_unix` (seconds since epoch, UTC) is preferred when present and is treated as the per-observation timestamp.
- `lat`, `lon`
- for `surface_obs` (any subset; script will compare what exists):
    - winds: `pred_wind_u`, `pred_wind_v`, `true_wind_u`, `true_wind_v`
    - temperature: `pred_airTemperature`, `true_airTemperature` (°C)
    - pressure: `pred_airPressure_prepbufr_event_1`, `true_airPressure_prepbufr_event_1` (hPa)
- for `radiosonde`: `pressure_hPa`, `pred_wind_u/v`, `true_wind_u/v`, optional `pred_airTemperature`
- for `aircraft`: `pressure_hPa`, `pred_windU/V`, `true_windU/V`, optional `pred_airTemperature`

GFS directory tree must look like:
  <gfs_root>/<YYYYMMDD>/gfs.<YYYYMMDD>.t<HH>z.pgrb2.0p25.f<FFF>

Example (surface obs for 2025030100)
-----------------------------------
conda run -n gnn-env python evaluation/scripts/compare_to_gfs.py \
  --instrument surface_obs \
  --ocelot_csv predictions/pred_csv/obs-space/pred_surface_obs_target_init_2025030100.csv \
  --gfs_root /scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25 \
  --out_csv predictions/pred_csv/obs-space/pred_surface_obs_target_init_2025030100_vs_gfs.csv \
  --init_mode from_csv \
  --interp nearest

If your prediction CSV does not yet have `datetime` and init-time columns (`init_datetime` or `init_time_unix`), rerun prediction
with the updated `gnn_model.py` in this repo first.
"""

from __future__ import annotations

import argparse
import glob
import os
import socket
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np
import pandas as pd


def _parse_datetime_utc(series: pd.Series) -> pd.Series:
    dt = pd.to_datetime(series, utc=True, errors="coerce")
    if dt.isna().any():
        bad = int(dt.isna().sum())
        if bad:
            print(f"[WARN] {bad} rows have invalid datetime; they will be dropped")
    return dt


def _lon_to_360(lon_deg: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_deg, dtype=np.float64)
    lon = np.mod(lon, 360.0)
    lon = np.where(lon >= 360.0, lon - 360.0, lon)
    return lon


def _infer_cycle_from_valid(valid: pd.Timestamp, cycle_hours=(0, 6, 12, 18)) -> pd.Timestamp:
    if valid.tzinfo is None:
        valid = valid.tz_localize("UTC")
    valid = valid.tz_convert("UTC")
    h = int(valid.hour)
    cycle_h = max([c for c in cycle_hours if c <= h], default=max(cycle_hours))
    init = valid.normalize() + pd.Timedelta(hours=cycle_h)
    if cycle_h > h:
        init = init - pd.Timedelta(days=1)
    return init


def _round_to_step(hours: float, step_hours: int, tol_hours: float | None = None) -> int:
    """Round a lead time to the nearest model step.

    If tol_hours is provided, require the raw lead to be within tol_hours of a
    step multiple; otherwise return -1 so the row can be dropped/flagged.

    Note: Python's round() uses bankers rounding at exact halves. The tolerance
    check is the primary guardrail against ambiguous 0.5-step cases.
    """
    if not np.isfinite(hours):
        return -1
    step_hours = int(step_hours)
    if step_hours <= 0:
        return -1

    fhr = float(step_hours) * float(round(float(hours) / float(step_hours)))
    if tol_hours is not None and np.isfinite(float(tol_hours)):
        if abs(float(hours) - float(fhr)) > float(tol_hours):
            return -1
    return int(fhr)


def _time_brackets(obs_fhr_hours: np.ndarray, step_hours: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (fhr_lo, fhr_hi, w_hi) for linear time interpolation.

    For each fractional lead hour t, find the bracketing model output times:
        fhr_lo = floor(t/step)*step
        fhr_hi = fhr_lo + step
        w_hi   = (t - fhr_lo) / step

    If t is exactly on an output time, fhr_hi == fhr_lo and w_hi == 0.
    """
    step_hours = int(step_hours)
    obs = np.asarray(obs_fhr_hours, dtype=np.float64)

    f0 = np.floor(obs / float(step_hours)) * float(step_hours)
    r = obs - f0

    eps = 1e-9
    on_edge = np.isfinite(r) & (np.abs(r) < eps)

    f1 = f0 + float(step_hours)
    w = r / float(step_hours)

    f1[on_edge] = f0[on_edge]
    w[on_edge] = 0.0

    return f0.astype(int), f1.astype(int), w.astype(np.float64)


@dataclass(frozen=True)
class GfsKey:
    init: pd.Timestamp  # UTC
    fhr: int


def _gfs_path(gfs_root: str, key: GfsKey) -> str:
    init = key.init.tz_convert("UTC")
    ymd = init.strftime("%Y%m%d")
    hh = init.strftime("%H")
    return os.path.join(gfs_root, ymd, f"gfs.{ymd}.t{hh}z.pgrb2.0p25.f{key.fhr:03d}")


@lru_cache(maxsize=32)
def _open_gfs_height10(path: str, short_name: str):
    import cfgrib

    return cfgrib.open_dataset(
        path,
        indexpath="",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10, "shortName": short_name},
    )


@lru_cache(maxsize=32)
def _open_gfs_height2(path: str, short_name: str):
    import cfgrib

    return cfgrib.open_dataset(
        path,
        indexpath="",
        filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2, "shortName": short_name},
    )


@lru_cache(maxsize=32)
def _open_gfs_surface(path: str, short_name: str):
    import cfgrib

    return cfgrib.open_dataset(
        path,
        indexpath="",
        filter_by_keys={"typeOfLevel": "surface", "shortName": short_name},
    )


@lru_cache(maxsize=16)
def _open_gfs_isobaric(path: str, short_name: str):
    import cfgrib

    return cfgrib.open_dataset(
        path,
        indexpath="",
        filter_by_keys={"typeOfLevel": "isobaricInhPa", "shortName": short_name},
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


def _interp_isobaric(ds, var: str, p_hpa: np.ndarray, lat: np.ndarray, lon360: np.ndarray, method: str) -> np.ndarray:
    import xarray as xr

    p_da = xr.DataArray(p_hpa.astype(np.float64), dims="points")
    lat_da = xr.DataArray(lat.astype(np.float64), dims="points")
    lon_da = xr.DataArray(lon360.astype(np.float64), dims="points")

    out = ds[var].interp(isobaricInhPa=p_da, method=method).interp(latitude=lat_da, longitude=lon_da, method=method)
    return out.values.astype(np.float64)


def _load_ocelot_csvs(patterns: Iterable[str]) -> pd.DataFrame:
    paths: list[str] = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)))
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        raise FileNotFoundError("No input CSVs matched")

    frames = []
    for p in paths:
        df = pd.read_csv(p)
        df["_source_csv"] = p
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _compute_gfs_keys(
    df: pd.DataFrame,
    init_mode: str,
    fhr_step: int,
    cycle_hours: tuple[int, ...],
    fhr_tolerance_hours: float | None,
    *,
    gfs_time_mode: str,
) -> pd.DataFrame:
    if "datetime" not in df.columns:
        raise ValueError("CSV must include a 'datetime' column")

    # NOTE:
    # In this repo's eval-mode prediction CSVs, `datetime` can represent the *per-observation timestamp*
    # inside the target sub-window (e.g., within [init, init+3h)). That is NOT necessarily the nominal
    # forecast valid time for a 3h/6h/... lead.
    #
    # For robust GFS matching we therefore prefer `lead_hours_nominal` when present.
    # For per-row observation timestamps, prefer `obs_time_unix` if available (seconds since epoch),
    # otherwise fall back to parsing the CSV `datetime` column.
    dt_obs = None
    if "obs_time_unix" in df.columns:
        obs_unix = pd.to_numeric(df["obs_time_unix"], errors="coerce")
        dt_from_unix = pd.to_datetime(obs_unix, unit="s", utc=True, errors="coerce")
        if dt_from_unix.notna().any():
            dt_obs = dt_from_unix
    if dt_obs is None:
        dt_obs = _parse_datetime_utc(df["datetime"])
    df = df.assign(_obs_dt=dt_obs)
    df = df.loc[df["_obs_dt"].notna()].copy()

    if init_mode == "from_csv" and "init_datetime" in df.columns:
        dt_init = _parse_datetime_utc(df["init_datetime"])
        df["_init_dt"] = dt_init
    elif init_mode == "from_csv" and "init_time_unix" in df.columns:
        init_unix = pd.to_numeric(df["init_time_unix"], errors="coerce")
        df["_init_dt"] = pd.to_datetime(init_unix, unit="s", utc=True, errors="coerce")
    elif init_mode == "infer_from_valid":
        # Best-effort inference from the observation datetime in CSV.
        df["_init_dt"] = df["_obs_dt"].apply(lambda x: _infer_cycle_from_valid(x, cycle_hours=cycle_hours))
    else:
        raise ValueError(f"Unsupported init_mode={init_mode!r}. Use from_csv or infer_from_valid")

    # Compute forecast hour.
    fhr_raw = None
    fhr_raw_source = "unknown"
    if "lead_hours_nominal" in df.columns:
        lead_nom = pd.to_numeric(df["lead_hours_nominal"], errors="coerce")
        if lead_nom.notna().any():
            fhr_raw = lead_nom
            fhr_raw_source = "lead_hours_nominal"

    if fhr_raw is None:
        # Fallback: treat CSV datetime as a valid time.
        # This can mis-map buckets (e.g., map many rows to f000 for the 3h bucket), so we warn below.
        lead_hours = (df["_obs_dt"] - df["_init_dt"]).dt.total_seconds() / 3600.0
        fhr_raw = lead_hours
        fhr_raw_source = "datetime_minus_init"

    df["_fhr_raw"] = fhr_raw
    df["_fhr_raw_source"] = fhr_raw_source
    df["_fhr"] = df["_fhr_raw"].apply(lambda x: _round_to_step(float(x), fhr_step, tol_hours=fhr_tolerance_hours))

    # Per-row observation lead (hours from init). Used for obs-time GFS baselines.
    df["_obs_fhr"] = (df["_obs_dt"] - df["_init_dt"]).dt.total_seconds() / 3600.0

    # Nominal valid time used for GFS matching.
    df["_valid_dt"] = df["_init_dt"] + pd.to_timedelta(df["_fhr"].astype(float), unit="h")

    # If the CSV datetime looks like per-observation times within a bucket, call that out explicitly.
    # This happens when (obs_dt - init_dt) spans a wide range within the same nominal fhr group.
    try:
        delta_h = (df["_obs_dt"] - df["_valid_dt"]).dt.total_seconds().abs() / 3600.0
        med = float(delta_h.median()) if len(delta_h) else 0.0
        p95 = float(delta_h.quantile(0.95)) if len(delta_h) else 0.0
        if (
            str(gfs_time_mode) == "nominal"
            and np.isfinite(med)
            and np.isfinite(p95)
            and (med > 0.25 or p95 > 1.0)
        ):
            print(
                "[WARN] CSV 'datetime' appears to be per-observation time, not nominal forecast valid time. "
                "GFS matching will use init_datetime + fhr (from lead_hours_nominal if present). "
                f"median|obs-valid|={med:.2f}h p95={p95:.2f}h"
            )
    except Exception:
        pass

    bad = (df["_fhr"] < 0) | (~np.isfinite(df["_fhr"].astype(float)))
    if bad.any():
        print(f"[WARN] Dropping {int(bad.sum())} rows with invalid fhr")
        df = df.loc[~bad].copy()

    df["_init_dt"] = pd.to_datetime(df["_init_dt"], utc=True)
    return df


def _iter_index_chunks(ii: np.ndarray, chunk_size: int):
    if chunk_size is None or int(chunk_size) <= 0:
        yield ii
        return
    chunk_size = int(chunk_size)
    for s in range(0, len(ii), chunk_size):
        yield ii[s: s + chunk_size]


def compare_surface_obs(df: pd.DataFrame, gfs_root: str, method: str, chunk_size: int = 200_000) -> pd.DataFrame:
    needed = {"lat", "lon"}
    missing = sorted([c for c in needed if c not in df.columns])
    if missing:
        raise ValueError(f"surface_obs CSV missing columns: {missing}")

    need_u10 = any(c in df.columns for c in ["pred_wind_u", "true_wind_u"])
    need_v10 = any(c in df.columns for c in ["pred_wind_v", "true_wind_v"])
    need_t2m = any(c in df.columns for c in ["pred_airTemperature", "true_airTemperature"])
    need_sp = any(
        c in df.columns
        for c in ["pred_airPressure_prepbufr_event_1", "true_airPressure_prepbufr_event_1", "pred_airPressure", "true_airPressure"]
    )

    lat = df["lat"].to_numpy(dtype=np.float64)
    lon360 = _lon_to_360(df["lon"].to_numpy(dtype=np.float64))

    gfs_u = np.full(len(df), np.nan, dtype=np.float64)
    gfs_v = np.full(len(df), np.nan, dtype=np.float64)
    gfs_t2m_c = np.full(len(df), np.nan, dtype=np.float64)
    gfs_sp_hpa = np.full(len(df), np.nan, dtype=np.float64)

    if not all(c in df.columns for c in ["_gfs_fhr0", "_gfs_fhr1", "_gfs_w"]):
        raise ValueError("Internal error: missing GFS time-bracketing columns (_gfs_fhr0/_gfs_fhr1/_gfs_w)")

    w_all = df["_gfs_w"].to_numpy(dtype=np.float64)

    for (init_dt, fh0, fh1), idx in df.groupby(["_init_dt", "_gfs_fhr0", "_gfs_fhr1"]).groups.items():
        if int(fh0) < 0 or int(fh1) < 0:
            continue

        init_ts = pd.Timestamp(init_dt).tz_convert("UTC")
        path0 = _gfs_path(gfs_root, GfsKey(init=init_ts, fhr=int(fh0)))
        path1 = _gfs_path(gfs_root, GfsKey(init=init_ts, fhr=int(fh1))) if int(fh1) != int(fh0) else path0

        has0 = os.path.exists(path0)
        has1 = os.path.exists(path1)
        if not has0 and not has1:
            print(f"[WARN] Missing GFS files: {path0} and {path1}")
            continue
        if not has0:
            print(f"[WARN] Missing GFS file: {path0} (falling back to {path1})")
        if not has1:
            print(f"[WARN] Missing GFS file: {path1} (falling back to {path0})")

        ds0_u = ds0_v = ds0_t = ds0_sp = None
        ds1_u = ds1_v = ds1_t = ds1_sp = None

        if has0:
            if need_u10:
                try:
                    ds0_u = _open_gfs_height10(path0, "10u")
                except Exception as e:
                    print(f"[WARN] Failed to open GFS 10u: {path0} ({e})")
            if need_v10:
                try:
                    ds0_v = _open_gfs_height10(path0, "10v")
                except Exception as e:
                    print(f"[WARN] Failed to open GFS 10v: {path0} ({e})")
            if need_t2m:
                try:
                    ds0_t = _open_gfs_height2(path0, "2t")
                except Exception as e:
                    print(f"[WARN] Failed to open GFS 2t: {path0} ({e})")
            if need_sp:
                try:
                    ds0_sp = _open_gfs_surface(path0, "sp")
                except Exception as e:
                    print(f"[WARN] Failed to open GFS surface pressure (sp): {path0} ({e})")

        if has1 and path1 != path0:
            if need_u10:
                try:
                    ds1_u = _open_gfs_height10(path1, "10u")
                except Exception as e:
                    print(f"[WARN] Failed to open GFS 10u: {path1} ({e})")
            if need_v10:
                try:
                    ds1_v = _open_gfs_height10(path1, "10v")
                except Exception as e:
                    print(f"[WARN] Failed to open GFS 10v: {path1} ({e})")
            if need_t2m:
                try:
                    ds1_t = _open_gfs_height2(path1, "2t")
                except Exception as e:
                    print(f"[WARN] Failed to open GFS 2t: {path1} ({e})")
            if need_sp:
                try:
                    ds1_sp = _open_gfs_surface(path1, "sp")
                except Exception as e:
                    print(f"[WARN] Failed to open GFS surface pressure (sp): {path1} ({e})")

        ii = np.asarray(idx, dtype=np.int64)
        for jj in _iter_index_chunks(ii, chunk_size=chunk_size):
            ww = w_all[jj]

            if need_u10:
                v0 = (
                    _interp_2d(ds0_u, _pick_var_name(ds0_u, "u10"), lat[jj], lon360[jj], method=method)
                    if ds0_u is not None
                    else np.full(len(jj), np.nan)
                )
                v1 = (
                    _interp_2d(ds1_u, _pick_var_name(ds1_u, "u10"), lat[jj], lon360[jj], method=method)
                    if ds1_u is not None
                    else v0
                )
                if ds0_u is None and ds1_u is not None:
                    v0 = v1
                gfs_u[jj] = (1.0 - ww) * v0 + ww * v1

            if need_v10:
                v0 = (
                    _interp_2d(ds0_v, _pick_var_name(ds0_v, "v10"), lat[jj], lon360[jj], method=method)
                    if ds0_v is not None
                    else np.full(len(jj), np.nan)
                )
                v1 = (
                    _interp_2d(ds1_v, _pick_var_name(ds1_v, "v10"), lat[jj], lon360[jj], method=method)
                    if ds1_v is not None
                    else v0
                )
                if ds0_v is None and ds1_v is not None:
                    v0 = v1
                gfs_v[jj] = (1.0 - ww) * v0 + ww * v1

            if need_t2m:
                v0 = (
                    _interp_2d(ds0_t, _pick_var_name(ds0_t, "t2m"), lat[jj], lon360[jj], method=method) - 273.15
                    if ds0_t is not None
                    else np.full(len(jj), np.nan)
                )
                v1 = (
                    _interp_2d(ds1_t, _pick_var_name(ds1_t, "t2m"), lat[jj], lon360[jj], method=method) - 273.15
                    if ds1_t is not None
                    else v0
                )
                if ds0_t is None and ds1_t is not None:
                    v0 = v1
                gfs_t2m_c[jj] = (1.0 - ww) * v0 + ww * v1

            if need_sp:
                v0 = (
                    _interp_2d(ds0_sp, _pick_var_name(ds0_sp, "sp"), lat[jj], lon360[jj], method=method) / 100.0
                    if ds0_sp is not None
                    else np.full(len(jj), np.nan)
                )
                v1 = (
                    _interp_2d(ds1_sp, _pick_var_name(ds1_sp, "sp"), lat[jj], lon360[jj], method=method) / 100.0
                    if ds1_sp is not None
                    else v0
                )
                if ds0_sp is None and ds1_sp is not None:
                    v0 = v1
                gfs_sp_hpa[jj] = (1.0 - ww) * v0 + ww * v1

    out = df.copy()
    out["gfs_u10"] = gfs_u
    out["gfs_v10"] = gfs_v
    out["gfs_t2m_C"] = gfs_t2m_c
    out["gfs_sp_hPa"] = gfs_sp_hpa

    if "pred_wind_u" in out.columns and "true_wind_u" in out.columns:
        out["ocelot_minus_gfs_u10_pred"] = out["pred_wind_u"] - out["gfs_u10"]
        out["truth_minus_gfs_u10"] = out["true_wind_u"] - out["gfs_u10"]

    if "pred_wind_v" in out.columns and "true_wind_v" in out.columns:
        out["ocelot_minus_gfs_v10_pred"] = out["pred_wind_v"] - out["gfs_v10"]
        out["truth_minus_gfs_v10"] = out["true_wind_v"] - out["gfs_v10"]

    if "pred_airTemperature" in out.columns and "true_airTemperature" in out.columns:
        out["ocelot_minus_gfs_t2m_pred"] = out["pred_airTemperature"] - out["gfs_t2m_C"]
        out["truth_minus_gfs_t2m"] = out["true_airTemperature"] - out["gfs_t2m_C"]

    # Station pressure in the conv CSV is in hPa; GFS sp is in Pa.
    p_pred_col = None
    p_true_col = None
    for base in ["airPressure_prepbufr_event_1", "airPressure", "pressure"]:
        if f"pred_{base}" in out.columns and f"true_{base}" in out.columns:
            p_pred_col = f"pred_{base}"
            p_true_col = f"true_{base}"
            break

    if p_pred_col is not None:
        out["ocelot_minus_gfs_sp_pred"] = out[p_pred_col] - out["gfs_sp_hPa"]
        out["truth_minus_gfs_sp"] = out[p_true_col] - out["gfs_sp_hPa"]

    return out


def compare_isobaric(
    df: pd.DataFrame,
    gfs_root: str,
    method: str,
    wind_u_col: str,
    wind_v_col: str,
    chunk_size: int = 10_000,
) -> pd.DataFrame:
    if "pressure_hPa" not in df.columns:
        raise ValueError("CSV missing pressure_hPa (needed for isobaric GFS match)")

    lat = df["lat"].to_numpy(dtype=np.float64)
    lon360 = _lon_to_360(df["lon"].to_numpy(dtype=np.float64))
    p = df["pressure_hPa"].to_numpy(dtype=np.float64)

    gfs_u = np.full(len(df), np.nan, dtype=np.float64)
    gfs_v = np.full(len(df), np.nan, dtype=np.float64)
    gfs_t_c = np.full(len(df), np.nan, dtype=np.float64)

    if not all(c in df.columns for c in ["_gfs_fhr0", "_gfs_fhr1", "_gfs_w"]):
        raise ValueError("Internal error: missing GFS time-bracketing columns (_gfs_fhr0/_gfs_fhr1/_gfs_w)")

    w_all = df["_gfs_w"].to_numpy(dtype=np.float64)

    for (init_dt, fh0, fh1), idx in df.groupby(["_init_dt", "_gfs_fhr0", "_gfs_fhr1"]).groups.items():
        if int(fh0) < 0 or int(fh1) < 0:
            continue

        init_ts = pd.Timestamp(init_dt).tz_convert("UTC")
        path0 = _gfs_path(gfs_root, GfsKey(init=init_ts, fhr=int(fh0)))
        path1 = _gfs_path(gfs_root, GfsKey(init=init_ts, fhr=int(fh1))) if int(fh1) != int(fh0) else path0

        has0 = os.path.exists(path0)
        has1 = os.path.exists(path1)
        if not has0 and not has1:
            print(f"[WARN] Missing GFS files: {path0} and {path1}")
            continue

        ds0_u = ds0_v = ds0_t = None
        ds1_u = ds1_v = ds1_t = None
        if has0:
            ds0_u = _open_gfs_isobaric(path0, "u")
            ds0_v = _open_gfs_isobaric(path0, "v")
            ds0_t = _open_gfs_isobaric(path0, "t")
        if has1 and path1 != path0:
            ds1_u = _open_gfs_isobaric(path1, "u")
            ds1_v = _open_gfs_isobaric(path1, "v")
            ds1_t = _open_gfs_isobaric(path1, "t")

        ii = np.asarray(idx, dtype=np.int64)
        for jj in _iter_index_chunks(ii, chunk_size=chunk_size):
            ww = w_all[jj]

            u0 = _interp_isobaric(ds0_u, "u", p[jj], lat[jj], lon360[jj], method=method) if ds0_u is not None else np.full(len(jj), np.nan)
            u1 = _interp_isobaric(ds1_u, "u", p[jj], lat[jj], lon360[jj], method=method) if ds1_u is not None else u0
            if ds0_u is None and ds1_u is not None:
                u0 = u1
            gfs_u[jj] = (1.0 - ww) * u0 + ww * u1

            v0 = _interp_isobaric(ds0_v, "v", p[jj], lat[jj], lon360[jj], method=method) if ds0_v is not None else np.full(len(jj), np.nan)
            v1 = _interp_isobaric(ds1_v, "v", p[jj], lat[jj], lon360[jj], method=method) if ds1_v is not None else v0
            if ds0_v is None and ds1_v is not None:
                v0 = v1
            gfs_v[jj] = (1.0 - ww) * v0 + ww * v1

            t0 = (_interp_isobaric(ds0_t, "t", p[jj], lat[jj], lon360[jj], method=method) - 273.15) if ds0_t is not None else np.full(len(jj), np.nan)
            t1 = (_interp_isobaric(ds1_t, "t", p[jj], lat[jj], lon360[jj], method=method) - 273.15) if ds1_t is not None else t0
            if ds0_t is None and ds1_t is not None:
                t0 = t1
            gfs_t_c[jj] = (1.0 - ww) * t0 + ww * t1

    out = df.copy()
    out["gfs_u"] = gfs_u
    out["gfs_v"] = gfs_v
    out["gfs_airTemperature_C"] = gfs_t_c

    if "pred_airTemperature" in out.columns and "true_airTemperature" in out.columns:
        out["ocelot_minus_gfs_temp_pred"] = out["pred_airTemperature"] - out["gfs_airTemperature_C"]
        out["truth_minus_gfs_temp"] = out["true_airTemperature"] - out["gfs_airTemperature_C"]

    if f"pred_{wind_u_col}" in out.columns and f"true_{wind_u_col}" in out.columns:
        out["ocelot_minus_gfs_u_pred"] = out[f"pred_{wind_u_col}"] - out["gfs_u"]
        out["truth_minus_gfs_u"] = out[f"true_{wind_u_col}"] - out["gfs_u"]

    if f"pred_{wind_v_col}" in out.columns and f"true_{wind_v_col}" in out.columns:
        out["ocelot_minus_gfs_v_pred"] = out[f"pred_{wind_v_col}"] - out["gfs_v"]
        out["truth_minus_gfs_v"] = out[f"true_{wind_v_col}"] - out["gfs_v"]

    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--instrument", required=True, choices=["surface_obs", "radiosonde", "aircraft"])
    ap.add_argument("--ocelot_csv", required=True, action="append", help="CSV path or glob; can be repeated")
    ap.add_argument("--gfs_root", required=True)
    ap.add_argument("--out_csv", required=True)

    ap.add_argument(
        "--init_mode",
        default="from_csv",
        choices=["from_csv", "infer_from_valid"],
        help="How to choose GFS init time. from_csv uses init_* columns if present.",
    )
    ap.add_argument("--fhr_step", type=int, default=3, help="Round lead hours to nearest multiple of this (GFS step)")
    ap.add_argument(
        "--fhr_tolerance_hours",
        type=float,
        default=0.25,
        help=(
            "Fail-fast guardrail: require raw lead hours to be within this tolerance of a fhr_step multiple. "
            "Rows outside tolerance are dropped. Use a negative value to disable. Default: 0.25h (15 min)."
        ),
    )
    ap.add_argument("--cycle_hours", default="0,6,12,18", help="Comma-separated cycle hours for infer_from_valid")
    ap.add_argument("--interp", default="nearest", choices=["nearest", "linear"], help="Spatial/vertical interpolation")
    ap.add_argument(
        "--gfs_time_mode",
        default="nominal",
        choices=["nominal", "obs_interp"],
        help=(
            "How to choose the GFS valid time for each row. "
            "nominal uses init_datetime + fhr (bucket end). "
            "obs_interp linearly interpolates GFS in time to each row's observation timestamp "
            "(prefers CSV 'obs_time_unix' when present; falls back to parsing 'datetime')."
        ),
    )
    ap.add_argument(
        "--chunk_size",
        type=int,
        default=10_000,
        help="Process points in chunks to reduce memory (esp. isobaric interpolation).",
    )
    ap.add_argument(
        "--allow_login",
        action="store_true",
        help="Allow running on login/service nodes (not recommended).",
    )

    args = ap.parse_args()

    host = socket.gethostname()
    if (host.startswith("ufe") or host.startswith("login")) and not args.allow_login:
        raise SystemExit(
            f"Refusing to run on login node {host!r}. "
            "Submit this script via sbatch/srun, or pass --allow_login if you really know what you're doing."
        )

    cycle_hours = tuple(int(x) for x in args.cycle_hours.split(",") if x.strip() != "")

    tol = float(args.fhr_tolerance_hours)
    if tol < 0:
        tol = None

    df = _load_ocelot_csvs(args.ocelot_csv)
    df = _compute_gfs_keys(
        df,
        init_mode=args.init_mode,
        fhr_step=int(args.fhr_step),
        cycle_hours=cycle_hours,
        fhr_tolerance_hours=tol,
        gfs_time_mode=str(args.gfs_time_mode),
    )

    # Compute time-bracketing fhrs + weights for GFS. In nominal mode, these collapse to a single file.
    if str(args.gfs_time_mode) == "obs_interp":
        f0, f1, w1 = _time_brackets(df["_obs_fhr"].to_numpy(dtype=np.float64), step_hours=int(args.fhr_step))
        df["_gfs_fhr0"] = f0
        df["_gfs_fhr1"] = f1
        df["_gfs_w"] = w1
        df["_gfs_valid_dt0"] = df["_init_dt"] + pd.to_timedelta(df["_gfs_fhr0"].astype(float), unit="h")
        df["_gfs_valid_dt1"] = df["_init_dt"] + pd.to_timedelta(df["_gfs_fhr1"].astype(float), unit="h")
    else:
        df["_gfs_fhr0"] = df["_fhr"].astype(int)
        df["_gfs_fhr1"] = df["_fhr"].astype(int)
        df["_gfs_w"] = 0.0
        df["_gfs_valid_dt0"] = df["_valid_dt"]
        df["_gfs_valid_dt1"] = df["_valid_dt"]

    if args.instrument == "surface_obs":
        out = compare_surface_obs(df, gfs_root=args.gfs_root, method=args.interp, chunk_size=int(args.chunk_size))
    elif args.instrument == "radiosonde":
        out = compare_isobaric(
            df,
            gfs_root=args.gfs_root,
            method=args.interp,
            wind_u_col="wind_u",
            wind_v_col="wind_v",
            chunk_size=int(args.chunk_size),
        )
    elif args.instrument == "aircraft":
        out = compare_isobaric(
            df,
            gfs_root=args.gfs_root,
            method=args.interp,
            wind_u_col="windU",
            wind_v_col="windV",
            chunk_size=int(args.chunk_size),
        )
    else:
        raise AssertionError(args.instrument)

    out["init_datetime_used"] = out["_init_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    if str(args.gfs_time_mode) == "obs_interp" and "_obs_dt" in out.columns:
        out["valid_datetime_used"] = out["_obs_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    else:
        out["valid_datetime_used"] = out["_valid_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["fhr_used"] = out["_fhr"].astype(int)
    out["gfs_time_mode"] = str(args.gfs_time_mode)
    if "_gfs_fhr0" in out.columns:
        out["gfs_fhr_lo"] = out["_gfs_fhr0"].astype(int)
        out["gfs_fhr_hi"] = out["_gfs_fhr1"].astype(int)
        out["gfs_time_weight_hi"] = pd.to_numeric(out["_gfs_w"], errors="coerce")
    if "_gfs_valid_dt0" in out.columns:
        out["gfs_valid_datetime_lo"] = out["_gfs_valid_dt0"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        out["gfs_valid_datetime_hi"] = out["_gfs_valid_dt1"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    # Keep the original per-row datetime around for debugging (often this is observation time).
    if "_obs_dt" in out.columns:
        out["obs_datetime_from_csv"] = out["_obs_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    out = out.drop(
        columns=[
            c
            for c in [
                "_init_dt",
                "_valid_dt",
                "_obs_dt",
                "_obs_fhr",
                "_fhr_raw",
                "_fhr_raw_source",
                "_gfs_fhr0",
                "_gfs_fhr1",
                "_gfs_w",
                "_gfs_valid_dt0",
                "_gfs_valid_dt1",
            ]
            if c in out.columns
        ]
    )

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv} (rows={len(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
