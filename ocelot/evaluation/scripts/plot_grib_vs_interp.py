#!/usr/bin/env python
"""Plot a GRIB-native GFS map vs values interpolated to obs locations.

Author: Azadeh Gholoubi

This is a diagnostic to sanity-check:
- GFS file selection for a given (init, fhr)
- GRIB metadata time/step/valid_time
- spatial interpolation (including longitude seam handling)

Example:
    conda run -n gnn-env python evaluation/scripts/plot_grib_vs_interp.py \
    --csv predictions/Rand_TenYear_nl16/pred_csv/obs-space/pred_surface_obs_target_init_2025030100.csv \
    --gfs_root /scratch3/NCEPDEV/da/Mu-Chieh.Ko/JEDI-nudging/gfs-rt25 \
    --init 2025030100 \
    --fhr 3 \
    --short_name 10u \
    --out_png /tmp/gfs_10u_init2025030100_f003_grib_vs_interp.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from compare_to_gfs import (
    GfsKey,
    _gfs_path,
    _interp_2d,
    _lon_to_360,
    _open_gfs_height10,
    _open_gfs_height2,
    _open_gfs_surface,
    _pick_var_name,
)


_CARTOPY_FEATURES_WARNED = False


def _cartopy_natural_earth_cached(scale: str = "110m") -> bool:
    mode = os.environ.get("OCELOT_CARTOPY_FEATURES", "auto").strip().lower()
    if mode in {"0", "false", "no", "off"}:
        return False
    if mode in {"1", "true", "yes", "on"}:
        return True

    try:
        from cartopy import config as cartopy_config

        data_dir = cartopy_config.get("pre_existing_data_dir") or cartopy_config.get("data_dir")
        if not data_dir:
            return False
        base = Path(str(data_dir)).expanduser() / "shapefiles" / "natural_earth"
        needed = [
            base / "physical" / f"ne_{scale}_land.shp",
            base / "physical" / f"ne_{scale}_coastline.shp",
            base / "cultural" / f"ne_{scale}_admin_0_boundary_lines_land.shp",
        ]
        for shp in needed:
            if not (shp.exists() and shp.with_suffix(".dbf").exists() and shp.with_suffix(".shx").exists()):
                return False
        return True
    except Exception:
        return False


def _add_land(ax):
    global _CARTOPY_FEATURES_WARNED

    if not _cartopy_natural_earth_cached(scale="110m"):
        if not _CARTOPY_FEATURES_WARNED:
            _CARTOPY_FEATURES_WARNED = True
            print(
                "[WARN] Cartopy NaturalEarth shapefiles not found locally; skipping coastlines/land/borders "
                "to avoid network downloads on offline nodes. "
                "(Set OCELOT_CARTOPY_FEATURES=1 to force, or pre-populate the Cartopy data cache.)"
            )
        return

    try:
        import cartopy.feature as cfeature

        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.4)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.25)
    except Exception:
        return


def _parse_init_utc(s: str) -> pd.Timestamp:
    s = str(s).strip()
    if s.isdigit() and len(s) == 10:
        # YYYYMMDDHH
        return pd.Timestamp(f"{s[0:4]}-{s[4:6]}-{s[6:8]}T{s[8:10]}:00:00Z").tz_convert("UTC")
    ts = pd.Timestamp(s)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def _to_lon180(lon_deg: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_deg, dtype=np.float64)
    return (np.mod(lon + 180.0, 360.0) - 180.0).astype(np.float64)


def _robust_sym_limits(values: np.ndarray, q: float = 99.0) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    m = float(np.nanpercentile(np.abs(v), q))
    if not np.isfinite(m) or m == 0:
        m = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 1.0
    if m == 0:
        m = 1.0
    return -m, m


def _robust_limits(values: np.ndarray, q_lo: float = 1.0, q_hi: float = 99.0) -> tuple[float, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return -1.0, 1.0
    lo = float(np.nanpercentile(v, q_lo))
    hi = float(np.nanpercentile(v, q_hi))
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo = float(np.nanmin(v))
        hi = float(np.nanmax(v))
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    return lo, hi


def _open_gfs(path: str, short_name: str):
    if short_name in {"10u", "10v"}:
        return _open_gfs_height10(path, short_name)
    if short_name in {"2t"}:
        return _open_gfs_height2(path, short_name)
    if short_name in {"sp"}:
        return _open_gfs_surface(path, short_name)
    raise ValueError(f"Unsupported short_name={short_name!r}. Use one of: 10u, 10v, 2t, sp")


def _get_fhr_series(df: pd.DataFrame) -> pd.Series:
    if "fhr_used" in df.columns:
        return pd.to_numeric(df["fhr_used"], errors="coerce")
    if "lead_hours_nominal" in df.columns:
        return pd.to_numeric(df["lead_hours_nominal"], errors="coerce")
    if "lead_hours" in df.columns:
        return pd.to_numeric(df["lead_hours"], errors="coerce")
    raise ValueError("CSV must include one of: fhr_used, lead_hours_nominal, lead_hours")


def _pick_truth_and_mask_cols(columns: list[str], short_name: str) -> tuple[str | None, str | None]:
    cols = set(columns)
    if short_name == "10u":
        truth = "true_wind_u" if "true_wind_u" in cols else None
        mask = "mask_wind_u" if "mask_wind_u" in cols else None
        return truth, mask
    if short_name == "10v":
        truth = "true_wind_v" if "true_wind_v" in cols else None
        mask = "mask_wind_v" if "mask_wind_v" in cols else None
        return truth, mask
    if short_name == "2t":
        truth = "true_airTemperature" if "true_airTemperature" in cols else None
        mask = "mask_airTemperature" if "mask_airTemperature" in cols else None
        return truth, mask
    if short_name == "sp":
        if "true_pressureMeanSeaLevel_prepbufr" in cols:
            truth = "true_pressureMeanSeaLevel_prepbufr"
            mask = "mask_pressureMeanSeaLevel_prepbufr" if "mask_pressureMeanSeaLevel_prepbufr" in cols else None
            return truth, mask
        if "true_airPressure_prepbufr_event_1" in cols:
            truth = "true_airPressure_prepbufr_event_1"
            mask = "mask_airPressure_prepbufr_event_1" if "mask_airPressure_prepbufr_event_1" in cols else None
            return truth, mask
        if "true_airPressure" in cols:
            truth = "true_airPressure"
            mask = "mask_airPressure" if "mask_airPressure" in cols else None
            return truth, mask
        return None, None
    return None, None


def _metrics(a: np.ndarray, b: np.ndarray) -> dict:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    m = np.isfinite(aa) & np.isfinite(bb)
    n = int(np.count_nonzero(m))
    if n == 0:
        return {"n": 0, "rmse": float("nan"), "bias": float("nan"), "corr": float("nan")}
    d = aa[m] - bb[m]
    bias = float(np.mean(d))
    rmse = float(np.sqrt(np.mean(d * d)))
    corr = float("nan")
    if n >= 2:
        try:
            corr = float(np.corrcoef(aa[m], bb[m])[0, 1])
        except Exception:
            corr = float("nan")
    return {"n": n, "rmse": rmse, "bias": bias, "corr": corr}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="OCELOT prediction CSV (must include lat/lon + a lead-time column)")
    ap.add_argument("--gfs_root", required=True)
    ap.add_argument("--init", required=True, help="Init time: YYYYMMDDHH or any pandas-parsable datetime")
    ap.add_argument("--fhr", required=True, type=int, help="Forecast hour (integer, e.g. 3, 6, 9)")
    ap.add_argument("--short_name", required=True, help="GRIB shortName (e.g. 10u, 10v, 2t, sp)")
    ap.add_argument("--interp", default="nearest", choices=["nearest", "linear"], help="Spatial interpolation method")
    ap.add_argument("--max_points", type=int, default=200_000, help="Subsample obs points for plotting")
    ap.add_argument("--grid_stride", type=int, default=1, help="Stride for plotting the GRIB grid (1=full)")
    ap.add_argument("--point_size", type=float, default=4.0)
    ap.add_argument(
        "--t2m_range_C",
        default="-30,30",
        help="Color range for 2t/t2m plots in °C (default: -30,30; set to 'auto' for robust range).",
    )
    ap.add_argument("--truth_col", default=None, help="Truth column in the CSV (default: inferred from --short_name)")
    ap.add_argument("--mask_col", default=None, help="Mask column in the CSV (default: inferred from --short_name)")
    ap.add_argument("--out_png", required=True)
    args = ap.parse_args()

    init = _parse_init_utc(args.init)
    key = GfsKey(init=init, fhr=int(args.fhr))
    gfs_path = _gfs_path(args.gfs_root, key)
    if not os.path.exists(gfs_path):
        raise FileNotFoundError(f"Missing GFS file: {gfs_path}")

    df = pd.read_csv(args.csv)
    for col in ["lat", "lon"]:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    fhr_series = _get_fhr_series(df)
    df = df.loc[np.isfinite(fhr_series) & (np.round(fhr_series).astype(int) == int(args.fhr))].copy()
    if df.empty:
        raise ValueError(f"No rows matched fhr={args.fhr} in {args.csv}")

    if len(df) > int(args.max_points):
        df = df.sample(n=int(args.max_points), random_state=7)

    truth_col = str(args.truth_col).strip() if args.truth_col is not None else None
    mask_col = str(args.mask_col).strip() if args.mask_col is not None else None
    if not truth_col:
        truth_col, mask_col_infer = _pick_truth_and_mask_cols(list(df.columns), str(args.short_name))
        if mask_col is None:
            mask_col = mask_col_infer

    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=np.float64)
    lon_in = pd.to_numeric(df["lon"], errors="coerce").to_numpy(dtype=np.float64)
    truth = None
    mask = None
    if truth_col and truth_col in df.columns:
        truth = pd.to_numeric(df[truth_col], errors="coerce").to_numpy(dtype=np.float64)
    if mask_col and mask_col in df.columns:
        mask = pd.to_numeric(df[mask_col], errors="coerce").to_numpy(dtype=np.float64)

    ok = np.isfinite(lat) & np.isfinite(lon_in)
    if truth is not None:
        ok &= np.isfinite(truth)
    if mask is not None and np.isfinite(mask).any():
        ok &= (mask > 0.5)

    lat = lat[ok]
    lon_in = lon_in[ok]
    if truth is not None:
        truth = truth[ok]

    lon360 = _lon_to_360(lon_in)

    ds = _open_gfs(gfs_path, args.short_name)
    var = _pick_var_name(ds, {"10u": "u10", "10v": "v10", "2t": "t2m", "sp": "sp"}.get(args.short_name, ""))

    vals = _interp_2d(ds, var, lat, lon360, method=args.interp)

    units = None
    if args.short_name in {"10u", "10v"}:
        units = "m/s"
    elif args.short_name == "2t":
        units = "°C"
        vals = vals - 273.15
        if truth is not None and np.isfinite(truth).any():
            # Heuristic: if truth looks like Kelvin, convert.
            if float(np.nanmedian(truth)) > 100.0:
                print(f"[WARN] truth_col={truth_col!r} median looks like Kelvin; converting to °C")
                truth = truth - 273.15
    elif args.short_name == "sp":
        units = "hPa"
        vals = vals / 100.0
        if truth is not None and np.isfinite(truth).any():
            # If truth looks like Pa, convert to hPa.
            if float(np.nanmedian(truth)) > 2000.0:
                print(f"[WARN] truth_col={truth_col!r} median looks like Pa; converting to hPa")
                truth = truth / 100.0

    n = int(vals.size)
    n_nan = int(np.count_nonzero(~np.isfinite(vals)))
    print("GFS file:", gfs_path)
    for k in ["time", "step", "valid_time"]:
        if k in ds.coords:
            try:
                print(f"GRIB {k}:", str(ds.coords[k].values))
            except Exception:
                pass

    expected_valid = init + pd.Timedelta(hours=int(args.fhr))
    print("Expected valid_time:", expected_valid.strftime("%Y-%m-%dT%H:%M:%SZ"))

    if "datetime" in df.columns:
        dt = pd.to_datetime(df["datetime"], utc=True, errors="coerce")
        if dt.notna().any():
            print("CSV datetime min/max:", dt.min().strftime("%Y-%m-%dT%H:%M:%SZ"), dt.max().strftime("%Y-%m-%dT%H:%M:%SZ"))

    print(f"Interpolated {var}: N={n}  NaN={n_nan}")
    if np.isfinite(vals).any():
        print("Interpolated min/median/max:", float(np.nanmin(vals)), float(np.nanmedian(vals)), float(np.nanmax(vals)))

    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    field = np.asarray(ds[var].values, dtype=np.float64)
    glat = np.asarray(ds["latitude"].values, dtype=np.float64)
    glon = np.asarray(ds["longitude"].values, dtype=np.float64)

    if args.short_name == "2t":
        field = field - 273.15
    elif args.short_name == "sp":
        field = field / 100.0

    stride = max(1, int(args.grid_stride))
    field = field[::stride, ::stride]
    glat = glat[::stride]
    glon = glon[::stride]

    glon180 = _to_lon180(glon)
    order = np.argsort(glon180)
    glon180 = glon180[order]
    field = field[:, order]

    lon_plot = _to_lon180(lon360)

    pool = [field[np.isfinite(field)], vals[np.isfinite(vals)]]
    if truth is not None:
        pool.append(truth[np.isfinite(truth)])
    pooled = np.concatenate([p for p in pool if p.size], axis=0) if any(p.size for p in pool) else field[np.isfinite(field)]

    if args.short_name in {"10u", "10v"}:
        vmin, vmax = _robust_sym_limits(pooled)
    elif args.short_name == "2t":
        s = str(args.t2m_range_C).strip().lower()
        if s == "auto":
            vmin, vmax = _robust_limits(pooled, 1.0, 99.0)
        else:
            try:
                lo_s, hi_s = [x.strip() for x in s.split(",", 1)]
                vmin, vmax = float(lo_s), float(hi_s)
            except Exception as e:
                raise ValueError("--t2m_range_C must be 'auto' or 'lo,hi' (e.g. -30,30)") from e
    else:
        vmin, vmax = _robust_limits(pooled, 1.0, 99.0)

    m = None
    if truth is not None:
        m = _metrics(vals, truth)
        print(f"Truth col: {truth_col}")
        if mask_col:
            print(f"Mask col: {mask_col}")
        print(f"RMSE(GFSinterp−Truth)={m['rmse']:.3g}  Bias={m['bias']:.3g}  Corr={m['corr']:.3f}  N={m['n']}")

    fig, axes = plt.subplots(1, 3, figsize=(24, 6), subplot_kw={"projection": ccrs.PlateCarree()}, sharey=True)

    ax0 = axes[0]
    ax0.set_title(f"GRIB native {args.short_name} ({var})\ninit={init:%Y%m%d%H} f{int(args.fhr):03d}")
    m0 = ax0.pcolormesh(glon180, glat, field, shading="auto", cmap="coolwarm", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    _add_land(ax0)
    ax0.set_global()
    cb0 = fig.colorbar(m0, ax=ax0, orientation="vertical", pad=0.02, fraction=0.05)
    cb0.set_label(f"{var}{f' ({units})' if units else ''}")

    ax1 = axes[1]
    ax1.set_title(f"GFS interpolated to obs ({args.interp})\nN={n} NaN={n_nan}")
    sc = ax1.scatter(lon_plot, lat, c=vals, s=float(args.point_size), cmap="coolwarm", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    _add_land(ax1)
    ax1.set_global()
    cb1 = fig.colorbar(sc, ax=ax1, orientation="vertical", pad=0.02, fraction=0.05)
    cb1.set_label(f"{var}{f' ({units})' if units else ''}")

    ax2 = axes[2]
    if truth is None:
        ax2.set_title("Truth (not available in CSV)")
        _add_land(ax2)
        ax2.set_global()
    else:
        stats = f"RMSE={m['rmse']:.3g} Bias={m['bias']:.3g}" if m is not None and np.isfinite(m.get('rmse', np.nan)) else ""
        ax2.set_title(f"Truth at obs\n{stats}")
        sc2 = ax2.scatter(lon_plot, lat, c=truth, s=float(args.point_size), cmap="coolwarm", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        _add_land(ax2)
        ax2.set_global()
        cb2 = fig.colorbar(sc2, ax=ax2, orientation="vertical", pad=0.02, fraction=0.05)
        cb2.set_label(f"Truth{f' ({units})' if units else ''}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    plt.savefig(args.out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Wrote:", args.out_png)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
