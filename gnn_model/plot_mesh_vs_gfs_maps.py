#!/usr/bin/env python
"""Plot mesh-grid OCELOT vs GFS comparisons.

Input CSV is produced by compare_mesh_to_gfs.py and should contain:
- lat, lon
- pred_* columns
- gfs_* columns

Outputs a 3-panel scatter map on the *same* mesh points:
  [OCELOT] [GFS] [OCELOT - GFS]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

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

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), subplot_kw={"projection": ccrs.PlateCarree()}, sharey=True)
    panels = [("OCELOT", pred, "turbo", None), ("GFS", gfs, "turbo", None), ("OCELOT − GFS", diff, "bwr", TwoSlopeNorm(vmin=dmin, vcenter=0.0, vmax=dmax))]

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
        pred_col, gfs_col, units = ("pred_airPressure_prepbufr_event_1" if "pred_airPressure_prepbufr_event_1" in df.columns else "pred_airPressure"), "gfs_sp_hPa", "hPa"

    if pred_col not in df.columns or gfs_col not in df.columns:
        raise SystemExit(f"Missing columns for {args.var}: need {pred_col} and {gfs_col}")

    pred = pd.to_numeric(df[pred_col], errors="coerce").to_numpy(dtype=np.float64)
    gfs = pd.to_numeric(df[gfs_col], errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(lon) & np.isfinite(lat) & np.isfinite(pred) & np.isfinite(gfs)
    lon, lat, pred, gfs = lon[valid], lat[valid], pred[valid], gfs[valid]

    title = f"mesh-grid {args.var} • OCELOT_on_mesh vs GFS_on_mesh"
    out_png = os.path.join(args.plot_dir, f"mesh_OCELOT_on_mesh_vs_GFS_on_mesh_{args.var}.png")
    plot_tripanel(lon, lat, pred, gfs, title, out_png, units=units, point_size=int(args.point_size))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
