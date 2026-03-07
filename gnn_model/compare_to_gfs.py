#!/usr/bin/env python
"""Compare Ocelot conventional-observation predictions vs GFS.

Inputs
------
Ocelot CSV must contain at least:
- `datetime` (valid time, ISO8601 UTC)
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
conda run -n gnn-env python compare_to_gfs.py \
  --instrument surface_obs \
  --ocelot_csv predictions/pred_csv/obs-space/pred_surface_obs_target_init_2025030100.csv \
  --gfs_root /scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25 \
  --out_csv predictions/pred_csv/obs-space/pred_surface_obs_target_init_2025030100_vs_gfs.csv \
  --init_mode from_csv \
  --interp nearest

If your prediction CSV does not yet have `datetime`/`init_datetime`, rerun prediction
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
) -> pd.DataFrame:
    if "datetime" not in df.columns:
        raise ValueError("CSV must include a 'datetime' column")

    # NOTE:
    # In this repo's eval-mode prediction CSVs, `datetime` can represent the *per-observation timestamp*
    # inside the target sub-window (e.g., within [init, init+3h)). That is NOT necessarily the nominal
    # forecast valid time for a 3h/6h/... lead.
    #
    # For robust GFS matching we therefore prefer `lead_hours_nominal` when present, and only fall back
    # to (datetime - init_datetime) when the nominal lead column is missing.
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

    # Nominal valid time used for GFS matching.
    df["_valid_dt"] = df["_init_dt"] + pd.to_timedelta(df["_fhr"].astype(float), unit="h")

    # If the CSV datetime looks like per-observation times within a bucket, call that out explicitly.
    # This happens when (obs_dt - init_dt) spans a wide range within the same nominal fhr group.
    try:
        delta_h = (df["_obs_dt"] - df["_valid_dt"]).dt.total_seconds().abs() / 3600.0
        med = float(delta_h.median()) if len(delta_h) else 0.0
        p95 = float(delta_h.quantile(0.95)) if len(delta_h) else 0.0
        if np.isfinite(med) and np.isfinite(p95) and (med > 0.25 or p95 > 1.0):
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

    for (init_dt, fhr), idx in df.groupby(["_init_dt", "_fhr"]).groups.items():
        key = GfsKey(init=pd.Timestamp(init_dt).tz_convert("UTC"), fhr=int(fhr))
        path = _gfs_path(gfs_root, key)
        if not os.path.exists(path):
            print(f"[WARN] Missing GFS file: {path}")
            continue

        ds_u = ds_v = ds_t = ds_sp = None
        if need_u10:
            try:
                ds_u = _open_gfs_height10(path, "10u")
            except Exception as e:
                print(f"[WARN] Failed to open GFS 10u: {path} ({e})")
        if need_v10:
            try:
                ds_v = _open_gfs_height10(path, "10v")
            except Exception as e:
                print(f"[WARN] Failed to open GFS 10v: {path} ({e})")
        if need_t2m:
            try:
                ds_t = _open_gfs_height2(path, "2t")
            except Exception as e:
                print(f"[WARN] Failed to open GFS 2t: {path} ({e})")
        if need_sp:
            try:
                ds_sp = _open_gfs_surface(path, "sp")
            except Exception as e:
                print(f"[WARN] Failed to open GFS surface pressure (sp): {path} ({e})")

        ii = np.asarray(idx, dtype=np.int64)
        for jj in _iter_index_chunks(ii, chunk_size=chunk_size):
            if ds_u is not None:
                gfs_u[jj] = _interp_2d(ds_u, _pick_var_name(ds_u, "u10"), lat[jj], lon360[jj], method=method)
            if ds_v is not None:
                gfs_v[jj] = _interp_2d(ds_v, _pick_var_name(ds_v, "v10"), lat[jj], lon360[jj], method=method)
            if ds_t is not None:
                tvar = _pick_var_name(ds_t, "t2m")
                gfs_t2m_c[jj] = _interp_2d(ds_t, tvar, lat[jj], lon360[jj], method=method) - 273.15
            if ds_sp is not None:
                spvar = _pick_var_name(ds_sp, "sp")
                gfs_sp_hpa[jj] = _interp_2d(ds_sp, spvar, lat[jj], lon360[jj], method=method) / 100.0

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

    for (init_dt, fhr), idx in df.groupby(["_init_dt", "_fhr"]).groups.items():
        key = GfsKey(init=pd.Timestamp(init_dt).tz_convert("UTC"), fhr=int(fhr))
        path = _gfs_path(gfs_root, key)
        if not os.path.exists(path):
            print(f"[WARN] Missing GFS file: {path}")
            continue

        ds_u = _open_gfs_isobaric(path, "u")
        ds_v = _open_gfs_isobaric(path, "v")
        ds_t = _open_gfs_isobaric(path, "t")

        ii = np.asarray(idx, dtype=np.int64)
        for jj in _iter_index_chunks(ii, chunk_size=chunk_size):
            gfs_u[jj] = _interp_isobaric(ds_u, "u", p[jj], lat[jj], lon360[jj], method=method)
            gfs_v[jj] = _interp_isobaric(ds_v, "v", p[jj], lat[jj], lon360[jj], method=method)
            gfs_t_c[jj] = _interp_isobaric(ds_t, "t", p[jj], lat[jj], lon360[jj], method=method) - 273.15

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
    )

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
    out["valid_datetime_used"] = out["_valid_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    out["fhr_used"] = out["_fhr"].astype(int)

    # Keep the original per-row datetime around for debugging (often this is observation time).
    if "_obs_dt" in out.columns:
        out["obs_datetime_from_csv"] = out["_obs_dt"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    out = out.drop(columns=[c for c in ["_init_dt", "_valid_dt", "_obs_dt"] if c in out.columns])

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv} (rows={len(out)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
