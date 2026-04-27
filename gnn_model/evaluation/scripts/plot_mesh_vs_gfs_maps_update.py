#!/usr/bin/env python
"""Plot OCELOT mesh-grid predictions vs GFS forecast (and optionally GFS analysis).

Author: Azadeh Gholoubi

Reads a CSV produced by compare_mesh_to_gfs.py and generates:

  - 3-panel plot  (odd fhrs: no analysis available)
      OCELOT forecast | GFS forecast | OCELOT − GFS

  - 6-panel plot  (even 6h fhrs: GFS analysis present)
      Row 1:  OCELOT forecast | GFS forecast    | GFS analysis
      Row 2:  OCELOT − GFS   | OCELOT − analysis | GFS − analysis

The script auto-detects which layout to use based on whether the anl_* column
for the requested variable contains any non-NaN values.
"""

from __future__ import annotations

import argparse
import os
import re

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Cartopy / plotting helpers
# ---------------------------------------------------------------------------

def _require_plotting() -> None:
    try:
        import cartopy  # noqa: F401
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise SystemExit(f"Plotting requires cartopy and matplotlib: {e}")


def _ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


_CARTOPY_FEATURES_WARNED = False


def _add_land(ax) -> None:
    global _CARTOPY_FEATURES_WARNED
    import cartopy.feature as cfeature

    mode = os.environ.get("OCELOT_CARTOPY_FEATURES", "auto").strip().lower()
    if mode in {"0", "false", "no", "off"}:
        return
    try:
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)
        ax.add_feature(cfeature.LAND, facecolor="lightgray", alpha=0.3)
    except Exception as e:
        if not _CARTOPY_FEATURES_WARNED:
            _CARTOPY_FEATURES_WARNED = True
            print(
                f"[WARN] Could not add Cartopy land features (offline node?). "
                f"(Set OCELOT_CARTOPY_FEATURES=1 to force.) Error={type(e).__name__}: {e}"
            )
        try:
            ax.stock_img()
        except Exception:
            pass


def _to_lon180(lon_deg: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_deg, dtype=np.float64)
    return (np.mod(lon + 180.0, 360.0) - 180.0).astype(np.float64)


# ---------------------------------------------------------------------------
# Statistics / colour-limit helpers
# ---------------------------------------------------------------------------

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


def _robust_limits(data: np.ndarray, q_lo: float, q_hi: float) -> tuple[float, float]:
    d = data[np.isfinite(data)]
    if len(d) == 0:
        return -1.0, 1.0
    return float(np.percentile(d, q_lo)), float(np.percentile(d, q_hi))


def _robust_sym(data: np.ndarray, q_hi: float) -> tuple[float, float]:
    d = data[np.isfinite(data)]
    if len(d) == 0:
        return -1.0, 1.0
    v = float(np.percentile(np.abs(d), q_hi))
    return -v, v


# ---------------------------------------------------------------------------
# GFS file path (used for native-vs-interpolated diagnostic)
# ---------------------------------------------------------------------------

def _gfs_path(gfs_root: str, init_ymdh: str, fhr: int) -> str:
    ymd = str(init_ymdh)[:8]
    hh = str(init_ymdh)[8:10]
    return os.path.join(str(gfs_root), ymd, f"gfs.{ymd}.t{hh}z.pgrb2.0p25.f{int(fhr):03d}")


def _open_gfs_mean_sea_pressure(path: str):
    import cfgrib

    for short_name in ("prmsl", "msl"):
        try:
            return cfgrib.open_dataset(
                path, indexpath="",
                filter_by_keys={"typeOfLevel": "meanSea", "shortName": short_name},
            )
        except Exception:
            pass
    raise RuntimeError(f"Could not open GFS mean sea-level pressure from {path}")


def _open_gfs(path: str, var: str, *, level_hpa: int | None = None):
    import cfgrib

    if var == "u10":
        return cfgrib.open_dataset(
            path, indexpath="",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10, "shortName": "10u"},
        )
    if var == "v10":
        return cfgrib.open_dataset(
            path, indexpath="",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10, "shortName": "10v"},
        )
    if var == "t2m":
        return cfgrib.open_dataset(
            path, indexpath="",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t"},
        )
    if var == "sp":
        return _open_gfs_mean_sea_pressure(path)
    if var in {"u", "v", "temp"}:
        if level_hpa is None:
            raise ValueError("isobaric plotting requires level_hpa")
        short = {"u": "u", "v": "v", "temp": "t"}[var]
        return cfgrib.open_dataset(
            path, indexpath="",
            filter_by_keys={"typeOfLevel": "isobaricInhPa", "level": int(level_hpa), "shortName": short},
        )
    raise ValueError(f"Unsupported var={var!r}")


# ---------------------------------------------------------------------------
# init/fhr inference from CSV
# ---------------------------------------------------------------------------

_INIT_FHR_RE = re.compile(r"init_(?P<init>\d{10}).*?_f(?P<fhr>\d{3})")


def _infer_init_fhr(df: pd.DataFrame, csv_path: str) -> tuple[str, int]:
    if "init_time" in df.columns and "fhr" in df.columns:
        init = str(df["init_time"].iloc[0])
        fhr = int(pd.to_numeric(df["fhr"].iloc[0], errors="coerce"))
        if init.isdigit() and len(init) == 10 and np.isfinite(float(fhr)):
            return init, fhr
    m = _INIT_FHR_RE.search(os.path.basename(str(csv_path)))
    if m:
        return m.group("init"), int(m.group("fhr"))
    raise ValueError(
        "Could not infer init_time/fhr from CSV "
        "(need init_time+fhr columns or init_YYYYMMDDHH_fFFF in filename)"
    )


# ---------------------------------------------------------------------------
# Surface-pressure column helper
# ---------------------------------------------------------------------------

def _surface_pressure_pred_col(df: pd.DataFrame) -> str | None:
    for col in (
        "pred_airPressure_prepbufr_event_1",
        "pred_airPressure",
        "pred_pressureMeanSeaLevel_prepbufr",
    ):
        if col in df.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# 3-panel plot  (no analysis)
# ---------------------------------------------------------------------------

def plot_tripanel(
    lon: np.ndarray,
    lat: np.ndarray,
    pred: np.ndarray,
    gfs: np.ndarray,
    title: str,
    out_png: str,
    units: str | None,
    point_size: int,
) -> None:
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    diff = pred - gfs
    m = _metrics(pred, gfs)
    stats_line = f"N={m['n']}  RMSE={m['rmse']:.3g}  Bias={m['bias']:.3g}  Corr={m['corr']:.3f}"
    vmin, vmax = _robust_limits(np.concatenate([pred, gfs]), 1.0, 99.0)
    dmin, dmax = _robust_sym(diff, 99.0)

    fig, axes = plt.subplots(
        1, 3,
        figsize=(20, 5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        sharey=True,
    )
    panels = [
        ("OCELOT", pred, "turbo", None),
        ("GFS", gfs, "turbo", None),
        ("OCELOT − GFS", diff, "bwr", TwoSlopeNorm(vmin=dmin, vcenter=0.0, vmax=dmax)),
    ]

    for ax, (ttl, field, cmap, norm) in zip(axes, panels):
        ax.set_title(ttl, fontsize=14)
        if norm is None:
            sc = ax.scatter(lon, lat, c=field, s=point_size, cmap=cmap,
                            vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            cb = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
            cb.set_label(f"Value{f' ({units})' if units else ''}")
        else:
            sc = ax.scatter(lon, lat, c=field, s=point_size, cmap=cmap,
                            norm=norm, transform=ccrs.PlateCarree())
            cb = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
            cb.set_label(f"Δ{f' ({units})' if units else ''}")
        _add_land(ax)
        ax.set_global()

    fig.suptitle(f"{title}\n{stats_line}", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_png}")


# ---------------------------------------------------------------------------
# 6-panel plot  (with analysis)
# ---------------------------------------------------------------------------

def plot_sixpanel(
    lon: np.ndarray,
    lat: np.ndarray,
    pred: np.ndarray,
    gfs: np.ndarray,
    anl: np.ndarray,
    title: str,
    out_png: str,
    units: str | None,
    point_size: int,
) -> None:
    """2-row × 3-col layout:
      Row 1: OCELOT forecast  | GFS forecast    | GFS analysis
      Row 2: OCELOT − GFS     | OCELOT − analysis | GFS − analysis
    """
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    diff_og = pred - gfs          # OCELOT − GFS forecast
    diff_oa = pred - anl          # OCELOT − analysis
    diff_ga = gfs - anl           # GFS forecast − analysis

    m_og = _metrics(pred, gfs)
    m_oa = _metrics(pred, anl)
    m_ga = _metrics(gfs, anl)

    # Shared absolute colour limits across all three value panels
    vmin, vmax = _robust_limits(np.concatenate([pred, gfs, anl]), 1.0, 99.0)

    # Shared difference colour limits across all three diff panels
    all_diffs = np.concatenate([
        diff_og[np.isfinite(diff_og)],
        diff_oa[np.isfinite(diff_oa)],
        diff_ga[np.isfinite(diff_ga)],
    ])
    dmin, dmax = _robust_sym(all_diffs, 99.0)
    diff_norm = TwoSlopeNorm(vmin=dmin, vcenter=0.0, vmax=dmax)

    fig, axes = plt.subplots(
        2, 3,
        figsize=(20, 10),
        subplot_kw={"projection": ccrs.PlateCarree()},
        sharex=True,
        sharey=True,
    )

    val_label = f"Value{f' ({units})' if units else ''}"
    diff_label = f"Δ{f' ({units})' if units else ''}"

    # Row 0: absolute fields
    row0 = [
        ("OCELOT forecast", pred),
        ("GFS forecast", gfs),
        ("GFS analysis (truth)", anl),
    ]
    for ax, (ttl, field) in zip(axes[0], row0):
        ax.set_title(ttl, fontsize=13)
        sc = ax.scatter(lon, lat, c=field, s=point_size, cmap="turbo",
                        vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02).set_label(val_label)
        _add_land(ax)
        ax.set_global()

    # Row 1: difference fields
    row1 = [
        (f"OCELOT − GFS  (RMSE={m_og['rmse']:.3g}  Bias={m_og['bias']:.3g})", diff_og),
        (f"OCELOT − analysis  (RMSE={m_oa['rmse']:.3g}  Bias={m_oa['bias']:.3g})", diff_oa),
        (f"GFS − analysis  (RMSE={m_ga['rmse']:.3g}  Bias={m_ga['bias']:.3g})", diff_ga),
    ]
    for ax, (ttl, field) in zip(axes[1], row1):
        ax.set_title(ttl, fontsize=11)
        sc = ax.scatter(lon, lat, c=field, s=point_size, cmap="bwr",
                        norm=diff_norm, transform=ccrs.PlateCarree())
        fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02).set_label(diff_label)
        _add_land(ax)
        ax.set_global()

    fig.suptitle(f"{title}  |  N={m_og['n']}", fontsize=14, y=1.01)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_png}")


# ---------------------------------------------------------------------------
# Native GFS grid vs interpolated diagnostic
# ---------------------------------------------------------------------------

def plot_gfs_native_vs_mesh_interp(
    *,
    lon: np.ndarray,
    lat: np.ndarray,
    gfs_on_mesh: np.ndarray,
    var: str,
    units: str | None,
    init_ymdh: str,
    fhr: int,
    gfs_root: str,
    interp_method: str,
    out_png: str,
    point_size: int,
    grid_stride: int,
    level_hpa: int | None = None,
) -> None:
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gfs_p = _gfs_path(gfs_root, init_ymdh=init_ymdh, fhr=int(fhr))
    if not os.path.exists(gfs_p):
        raise FileNotFoundError(f"Missing GFS file: {gfs_p}")

    ds = _open_gfs(gfs_p, var=var, level_hpa=level_hpa)
    ds_var = {"u10": "u10", "v10": "v10", "t2m": "t2m", "sp": "prmsl",
              "u": "u", "v": "v", "temp": "t"}[var]
    if ds_var not in ds:
        keys = list(getattr(ds, "data_vars", {}).keys())
        if len(keys) != 1:
            raise KeyError(f"Variable {ds_var!r} not in GRIB; available: {keys}")
        ds_var = keys[0]

    field = np.asarray(ds[ds_var].values, dtype=np.float64)
    glat = np.asarray(ds["latitude"].values, dtype=np.float64)
    glon = np.asarray(ds["longitude"].values, dtype=np.float64)

    if var in {"t2m", "temp"}:
        field = field - 273.15
    elif var == "sp":
        field = field / 100.0

    stride = max(1, int(grid_stride))
    field = field[::stride, ::stride]
    glat = glat[::stride]
    glon = glon[::stride]

    glon180 = _to_lon180(glon)
    order = np.argsort(glon180)
    glon180 = glon180[order]
    field = field[:, order]

    pooled = np.concatenate([
        field[np.isfinite(field)],
        np.asarray(gfs_on_mesh, dtype=np.float64)[np.isfinite(gfs_on_mesh)],
    ])
    if var in {"u10", "v10"}:
        vmin, vmax = _robust_sym(pooled, 99.0)
        cmap = "coolwarm"
    else:
        vmin, vmax = _robust_limits(pooled, 1.0, 99.0)
        cmap = "turbo"

    fig, axes = plt.subplots(
        1, 2,
        figsize=(18, 5.5),
        subplot_kw={"projection": ccrs.PlateCarree()},
        sharey=True,
    )

    ax0 = axes[0]
    ax0.set_title(f"GFS native GRIB ({var})\ninit={init_ymdh} f{int(fhr):03d}")
    m0 = ax0.pcolormesh(glon180, glat, field, shading="auto", cmap=cmap,
                        vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    _add_land(ax0)
    ax0.set_global()
    fig.colorbar(m0, ax=ax0, orientation="vertical", pad=0.02, fraction=0.05).set_label(
        f"{var}{f' ({units})' if units else ''}"
    )

    ax1 = axes[1]
    ax1.set_title(f"GFS interpolated onto OCELOT mesh ({interp_method})")
    sc = ax1.scatter(lon, lat, c=gfs_on_mesh, s=point_size, cmap=cmap,
                     vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    _add_land(ax1)
    ax1.set_global()
    fig.colorbar(sc, ax=ax1, orientation="vertical", pad=0.02, fraction=0.05).set_label(
        f"{var}{f' ({units})' if units else ''}"
    )

    fig.suptitle("GFS native vs interpolated-to-mesh diagnostic", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_png}")


# ---------------------------------------------------------------------------
# Column mapping: var name → (pred_col, gfs_col, anl_col, units)
# ---------------------------------------------------------------------------

def _col_map(var: str, df: pd.DataFrame) -> tuple[str | None, str, str, str | None]:
    """Return (pred_col, gfs_col, anl_col, units) for the requested variable."""
    if var == "u10":
        return "pred_wind_u", "gfs_u10", "anl_u10", "m/s"
    if var == "v10":
        return "pred_wind_v", "gfs_v10", "anl_v10", "m/s"
    if var == "t2m":
        return "pred_airTemperature", "gfs_t2m_C", "anl_t2m_C", "°C"
    if var == "sp":
        pred_col = _surface_pressure_pred_col(df)
        return pred_col, "gfs_mslp_hPa", "anl_mslp_hPa", "hPa"
    if var == "u":
        return "pred_wind_u", "gfs_u", "anl_u", "m/s"
    if var == "v":
        return "pred_wind_v", "gfs_v", "anl_v", "m/s"
    # temp
    return "pred_airTemperature", "gfs_airTemperature_C", "anl_airTemperature_C", "°C"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description=(
            "Plot OCELOT mesh-grid predictions vs GFS forecast "
            "(and GFS analysis when available)."
        )
    )
    ap.add_argument(
        "--csv",
        required=True,
        help=(
            "CSV produced by compare_mesh_to_gfs.py containing pred_*, gfs_*, and "
            "optionally anl_* columns.  Common names: *_gfs_on_ocelot_mesh.csv"
        ),
    )
    ap.add_argument("--plot_dir", required=True)
    ap.add_argument("--var", required=True, choices=["u10", "v10", "t2m", "sp", "u", "v", "temp"])
    ap.add_argument("--point_size", type=int, default=5)
    ap.add_argument(
        "--gfs_root",
        default=None,
        help="If provided, also plot native GFS GRIB vs interpolated-to-mesh for diagnostics.",
    )
    ap.add_argument("--interp", default="nearest", choices=["nearest", "linear"])
    ap.add_argument("--grid_stride", type=int, default=2)
    args = ap.parse_args()

    _require_plotting()
    _ensure_dir(args.plot_dir)

    df = pd.read_csv(args.csv)
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(dtype=np.float64)
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=np.float64)

    pred_col, gfs_col, anl_col, units = _col_map(args.var, df)

    # Level tag for isobaric variables
    level_tag = ""
    level_hpa = None
    if "pressure_level_label" in df.columns and df["pressure_level_label"].notna().any():
        level_tag = str(df["pressure_level_label"].dropna().iloc[0])
    elif "pressure_hPa" in df.columns and pd.to_numeric(df["pressure_hPa"], errors="coerce").notna().any():
        level_hpa = int(round(float(pd.to_numeric(df["pressure_hPa"], errors="coerce").dropna().iloc[0])))
        level_tag = f"{level_hpa}hPa"
    if level_hpa is None and level_tag.endswith("hPa"):
        try:
            level_hpa = int(level_tag.replace("hPa", ""))
        except Exception:
            level_hpa = None

    if pred_col is None or pred_col not in df.columns or gfs_col not in df.columns:
        raise SystemExit(f"Missing columns for {args.var}: need {pred_col!r} and {gfs_col!r}")

    pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=np.float64)
    gfs = pd.to_numeric(df[gfs_col], errors="coerce").to_numpy(dtype=np.float64)

    # Check whether analysis data is present and non-NaN for this CSV
    has_analysis = (
        anl_col in df.columns
        and pd.to_numeric(df[anl_col], errors="coerce").notna().any()
    )
    if has_analysis:
        anl = pd.to_numeric(df[anl_col], errors="coerce").to_numpy(dtype=np.float64)
    else:
        anl = None

    lvl = f" • {level_tag}" if level_tag else ""
    base_title = f"mesh-grid {args.var}{lvl} • init={df['init_time'].iloc[0] if 'init_time' in df.columns else ''} fhr={df['fhr'].iloc[0] if 'fhr' in df.columns else ''}"
    level_suffix = f"_{'_' + level_tag if level_tag else ''}"

    if has_analysis:
        # 6-panel: need all three to be finite at the same points
        valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(pred) & np.isfinite(gfs) & np.isfinite(anl)
        _lon, _lat, _pred, _gfs, _anl = lon[valid], lat[valid], pred[valid], gfs[valid], anl[valid]

        title = f"{base_title} • vs GFS forecast + analysis"
        out_png = os.path.join(
            args.plot_dir,
            f"mesh_6panel_{args.var}{level_suffix}.png",
        )
        plot_sixpanel(_lon, _lat, _pred, _gfs, _anl, title, out_png,
                      units=units, point_size=int(args.point_size))
    else:
        # 3-panel: original behaviour
        valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(pred) & np.isfinite(gfs)
        _lon, _lat, _pred, _gfs = lon[valid], lat[valid], pred[valid], gfs[valid]

        title = f"{base_title} • OCELOT_on_mesh vs GFS_on_mesh"
        out_png = os.path.join(
            args.plot_dir,
            f"mesh_OCELOT_on_mesh_vs_GFS_on_mesh_{args.var}{level_suffix}.png",
        )
        plot_tripanel(_lon, _lat, _pred, _gfs, title, out_png,
                      units=units, point_size=int(args.point_size))

    # Optional native-GFS diagnostic (unchanged from original)
    if args.gfs_root:
        init_ymdh, fhr = _infer_init_fhr(df, csv_path=str(args.csv))
        out_png2 = os.path.join(
            args.plot_dir,
            f"mesh_GFS_native_vs_GFS_on_mesh_{args.var}{level_suffix}_init_{init_ymdh}_f{int(fhr):03d}.png",
        )
        plot_gfs_native_vs_mesh_interp(
            lon=_lon,
            lat=_lat,
            gfs_on_mesh=_gfs,
            var=str(args.var),
            units=units,
            init_ymdh=str(init_ymdh),
            fhr=int(fhr),
            gfs_root=str(args.gfs_root),
            interp_method=str(args.interp),
            out_png=str(out_png2),
            point_size=int(args.point_size),
            grid_stride=int(args.grid_stride),
            level_hpa=level_hpa,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
