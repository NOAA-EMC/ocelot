#!/usr/bin/env python
"""Plot mesh-grid OCELOT vs GFS comparisons.

Author: Azadeh Gholoubi

Input CSV is produced by compare_mesh_to_gfs.py and should contain:
- lat, lon
- pred_* columns
- gfs_* columns

Outputs a 3-panel scatter map on the *same* mesh points:
  [OCELOT] [GFS] [OCELOT - GFS]

If --gfs_root is provided, also writes a diagnostic 2-panel plot to visualize
the *native* GRIB GFS field (0.25°) alongside the GFS values interpolated to
the OCELOT mesh points:
    [GFS native GRIB] [GFS interpolated onto mesh]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re

import numpy as np
import pandas as pd


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


def _require_plotting():
    try:
        import cartopy.crs as ccrs  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Plotting dependencies (cartopy/matplotlib) are not available in this environment."
        ) from e


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


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


def _robust_sym(values: np.ndarray, q: float = 99.0) -> tuple[float, float]:
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


def _add_land(ax):
    """Add land/sea boundaries.

    Match the boundary styling used in `evaluations.py` (`_add_land_boundaries`).
    """

    global _CARTOPY_FEATURES_WARNED

    mode = os.environ.get("OCELOT_CARTOPY_FEATURES", "auto").strip().lower()
    if mode in {"0", "false", "no", "off"}:
        return

    try:
        import cartopy.feature as cfeature

        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)
        return
    except Exception as e:
        if mode in {"1", "true", "yes", "on"}:
            raise
        if not _CARTOPY_FEATURES_WARNED:
            _CARTOPY_FEATURES_WARNED = True
            print(
                "[WARN] Could not add Cartopy coastlines/borders; falling back to stock background. "
                "(Pre-populate Cartopy NaturalEarth cache, or set OCELOT_CARTOPY_FEATURES=1 to force.) "
                f"Error={type(e).__name__}: {e}"
            )
        try:
            ax.stock_img()
        except Exception:
            pass
        return


def _to_lon180(lon_deg: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon_deg, dtype=np.float64)
    return (np.mod(lon + 180.0, 360.0) - 180.0).astype(np.float64)


def _gfs_path(gfs_root: str, init_ymdh: str, fhr: int) -> str:
    init_ymdh = str(init_ymdh)
    ymd = init_ymdh[:8]
    hh = init_ymdh[8:10]
    return os.path.join(str(gfs_root), ymd, f"gfs.{ymd}.t{hh}z.pgrb2.0p25.f{int(fhr):03d}")


def _open_gfs(path: str, var: str):
    import cfgrib

    if var == "u10":
        return cfgrib.open_dataset(
            path,
            indexpath="",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10, "shortName": "10u"},
        )
    if var == "v10":
        return cfgrib.open_dataset(
            path,
            indexpath="",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 10, "shortName": "10v"},
        )
    if var == "t2m":
        return cfgrib.open_dataset(
            path,
            indexpath="",
            filter_by_keys={"typeOfLevel": "heightAboveGround", "level": 2, "shortName": "2t"},
        )
    if var == "sp":
        return cfgrib.open_dataset(
            path,
            indexpath="",
            filter_by_keys={"typeOfLevel": "surface", "shortName": "sp"},
        )
    raise ValueError(f"Unsupported var={var!r}")


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

    raise ValueError("Could not infer init_time/fhr from CSV (need init_time+fhr columns or init_YYYYMMDDHH_fFFF in filename)")


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
):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    gfs_path = _gfs_path(gfs_root, init_ymdh=init_ymdh, fhr=int(fhr))
    if not os.path.exists(gfs_path):
        raise FileNotFoundError(f"Missing GFS file: {gfs_path}")

    ds = _open_gfs(gfs_path, var=var)
    ds_var = {"u10": "u10", "v10": "v10", "t2m": "t2m", "sp": "sp"}[var]
    if ds_var not in ds:
        # cfgrib sometimes exposes a single unknown var name; fall back to that.
        keys = list(getattr(ds, "data_vars", {}).keys())
        if len(keys) != 1:
            raise KeyError(f"Variable {ds_var!r} not found in GRIB dataset; available vars={keys}")
        ds_var = keys[0]

    field = np.asarray(ds[ds_var].values, dtype=np.float64)
    glat = np.asarray(ds["latitude"].values, dtype=np.float64)
    glon = np.asarray(ds["longitude"].values, dtype=np.float64)

    if var == "t2m":
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

    fig, axes = plt.subplots(1, 2, figsize=(18, 5.5), subplot_kw={"projection": ccrs.PlateCarree()}, sharey=True)

    ax0 = axes[0]
    ax0.set_title(f"GFS native GRIB ({var})\ninit={init_ymdh} f{int(fhr):03d}")
    m0 = ax0.pcolormesh(glon180, glat, field, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    _add_land(ax0)
    ax0.set_global()
    cb0 = fig.colorbar(m0, ax=ax0, orientation="vertical", pad=0.02, fraction=0.05)
    cb0.set_label(f"{var}{f' ({units})' if units else ''}")

    ax1 = axes[1]
    ax1.set_title(f"GFS interpolated onto OCELOT mesh ({interp_method})")
    sc = ax1.scatter(lon, lat, c=gfs_on_mesh, s=point_size, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
    _add_land(ax1)
    ax1.set_global()
    cb1 = fig.colorbar(sc, ax=ax1, orientation="vertical", pad=0.02, fraction=0.05)
    cb1.set_label(f"{var}{f' ({units})' if units else ''}")

    fig.suptitle("GFS native vs interpolated-to-mesh diagnostic", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_png}")


def plot_tripanel(lon, lat, pred, gfs, title, out_png, units: str | None, point_size: int):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.colors import TwoSlopeNorm  # noqa: E402

    diff = pred - gfs
    m = _metrics(pred, gfs)
    stats_line = f"N={m['n']}  RMSE={m['rmse']:.3g}  Bias={m['bias']:.3g}  Corr={m['corr']:.3f}"
    vmin, vmax = _robust_limits(np.concatenate([pred, gfs]), 1.0, 99.0)
    dmin, dmax = _robust_sym(diff, 99.0)

    fig, axes = plt.subplots(
        1,
        3,
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
            sc = ax.scatter(lon, lat, c=field, s=point_size, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            cb = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
            cb.set_label(f"Value{f' ({units})' if units else ''}")
        else:
            sc = ax.scatter(lon, lat, c=field, s=point_size, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
            cb = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
            cb.set_label(f"Δ{f' ({units})' if units else ''}")

        _add_land(ax)
        ax.set_global()

    fig.suptitle(f"{title}\n{stats_line}", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_png}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv",
        required=True,
        help=(
            "CSV containing OCELOT predictions on mesh (pred_*) and GFS interpolated onto the same mesh (gfs_*). "
            "Common names: *_gfs_on_ocelot_mesh.csv (preferred) or legacy *_mesh_vs_gfs.csv"
        ),
    )
    ap.add_argument("--plot_dir", required=True)
    ap.add_argument("--var", required=True, choices=["u10", "v10", "t2m", "sp"])
    ap.add_argument("--point_size", type=int, default=5)
    ap.add_argument(
        "--gfs_root",
        default=None,
        help="If provided, also plot the native GFS GRIB field for the inferred init/fhr to diagnose interpolation.",
    )
    ap.add_argument("--interp", default="nearest", choices=["nearest", "linear"], help="Label for the interpolation method used")
    ap.add_argument("--grid_stride", type=int, default=2, help="Stride for plotting the GFS grid (1=full, 2=every other)")
    args = ap.parse_args()

    _require_plotting()
    _ensure_dir(args.plot_dir)

    df = pd.read_csv(args.csv)
    lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(dtype=np.float64)
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(dtype=np.float64)

    if args.var == "u10":
        pred_col, gfs_col, units = "pred_wind_u", "gfs_u10", "m/s"
    elif args.var == "v10":
        pred_col, gfs_col, units = "pred_wind_v", "gfs_v10", "m/s"
    elif args.var == "t2m":
        pred_col, gfs_col, units = "pred_airTemperature", "gfs_t2m_C", "°C"
    else:
        pred_col = "pred_airPressure_prepbufr_event_1" if "pred_airPressure_prepbufr_event_1" in df.columns else "pred_airPressure"
        gfs_col, units = "gfs_sp_hPa", "hPa"

    if pred_col not in df.columns or gfs_col not in df.columns:
        raise SystemExit(f"Missing columns for {args.var}: need {pred_col} and {gfs_col}")

    pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=np.float64)
    gfs = pd.to_numeric(df[gfs_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(pred) & np.isfinite(gfs)
    lon, lat, pred, gfs = lon[valid], lat[valid], pred[valid], gfs[valid]

    title = f"mesh-grid {args.var} • OCELOT_on_mesh vs GFS_on_mesh"
    out_png = os.path.join(args.plot_dir, f"mesh_OCELOT_on_mesh_vs_GFS_on_mesh_{args.var}.png")
    plot_tripanel(lon, lat, pred, gfs, title, out_png, units=units, point_size=int(args.point_size))

    if args.gfs_root:
        init_ymdh, fhr = _infer_init_fhr(df, csv_path=str(args.csv))
        out_png2 = os.path.join(args.plot_dir, f"mesh_GFS_native_vs_GFS_on_mesh_{args.var}_init_{init_ymdh}_f{int(fhr):03d}.png")
        plot_gfs_native_vs_mesh_interp(
            lon=lon,
            lat=lat,
            gfs_on_mesh=gfs,
            var=str(args.var),
            units=units,
            init_ymdh=str(init_ymdh),
            fhr=int(fhr),
            gfs_root=str(args.gfs_root),
            interp_method=str(args.interp),
            out_png=str(out_png2),
            point_size=int(args.point_size),
            grid_stride=int(args.grid_stride),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
