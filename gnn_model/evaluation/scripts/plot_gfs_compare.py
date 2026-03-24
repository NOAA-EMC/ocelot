#!/usr/bin/env python
"""plot_gfs_compare.py

Single entrypoint for obs-space comparisons against GFS using *_vs_gfs.csv.

Author: Azadeh Gholoubi

Typical usage (RMSE + maps for multiple lead times):

    python evaluation/scripts/plot_gfs_compare.py \
    --init_time 2025030100 \
    --data_dir predictions/.../pred_csv/obs-space \
        --plot_dir evaluation/figures/gfs_compare/init_2025030100 \
    --instrument surface_obs \
    --vars wind_temperature_pressure \
    --chunksize 200000 \
    --maps --fhrs 3 6 9 12

"""

from __future__ import annotations

import argparse
import glob
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _unique_ints(xs: Iterable[int]) -> list[int]:
    out: list[int] = []
    seen: set[int] = set()
    for x in xs:
        i = int(x)
        if i in seen:
            continue
        seen.add(i)
        out.append(i)
    return out


@dataclass
class VarSpec:
    name: str
    pred: str
    truth: str
    gfs: str
    mask: str | None


@dataclass(frozen=True)
class MapSpec:
    name: str
    pred_col: str
    truth_col: str
    gfs_col: str
    mask_col: str | None
    units: str | None


def _find_one(path_glob: str) -> str:
    hits = sorted(glob.glob(path_glob))
    if not hits:
        raise FileNotFoundError(f"No files matched: {path_glob}")
    if len(hits) > 1:
        raise FileExistsError(f"Expected 1 file for {path_glob}, found {len(hits)}: {hits[:5]}")
    return hits[0]


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _require_plotting():
    try:
        import cartopy.crs as ccrs  # noqa: F401
        import matplotlib.pyplot as plt  # noqa: F401
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Plotting dependencies (cartopy/matplotlib) are not available in this environment. "
            "Install them or run on a node with the plotting stack available."
        ) from e


def _rmse_from_sums(sumsq: float, count: int) -> float:
    if count <= 0:
        return float("nan")
    return float(np.sqrt(sumsq / count))


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


def _robust_sym_limits(values: np.ndarray, q_lo: float = 1.0, q_hi: float = 99.0) -> tuple[float, float]:
    lo, hi = _robust_limits(values, q_lo=q_lo, q_hi=q_hi)
    m = max(abs(lo), abs(hi))
    if not np.isfinite(m) or m == 0:
        m = 1.0
    return -m, m


def _get_fhr_series(df: pd.DataFrame) -> pd.Series:
    if "fhr_used" in df.columns:
        return pd.to_numeric(df["fhr_used"], errors="coerce")
    if "lead_hours_nominal" in df.columns:
        return pd.to_numeric(df["lead_hours_nominal"], errors="coerce")
    raise ValueError("CSV must include fhr_used or lead_hours_nominal")


def _chunk_iter(csv_path: str, usecols: list[str], chunksize: int):
    return pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize)


def _reservoir_append(rng: np.random.Generator, cur: np.ndarray, new: np.ndarray, max_points: int) -> np.ndarray:
    if max_points <= 0:
        return cur
    if cur.size == 0:
        if new.size <= max_points:
            return new
        idx = rng.choice(new.size, size=max_points, replace=False)
        return new[idx]

    total = cur.size + new.size
    if total <= max_points:
        return np.concatenate([cur, new])

    out = cur.copy()
    start = cur.size
    for i in range(new.size):
        j = start + i
        k = int(rng.integers(0, j + 1))
        if k < out.size:
            out[k] = new[i]
    return out


def sample_values(
    csv_path: str,
    varspecs: list[VarSpec],
    chunksize: int,
    max_points: int = 200_000,
    seed: int = 7,
) -> dict[str, dict[str, np.ndarray]]:
    """Return a sampled set of (pred, truth, gfs) arrays per variable for extra plots."""

    preview = pd.read_csv(csv_path, nrows=1)
    base_cols = ["fhr_used", "lead_hours_nominal"]
    needed_cols: set[str] = set(c for c in base_cols if c in preview.columns)
    for vs in varspecs:
        needed_cols.update([vs.pred, vs.truth, vs.gfs])
        if vs.mask:
            needed_cols.add(vs.mask)

    usecols = [c for c in needed_cols if c in preview.columns]
    rng = np.random.default_rng(seed)

    out: dict[str, dict[str, np.ndarray]] = {
        vs.name: {"pred": np.array([], dtype=np.float64), "truth": np.array([], dtype=np.float64), "gfs": np.array([], dtype=np.float64)}
        for vs in varspecs
    }

    for chunk in _chunk_iter(csv_path, usecols=usecols, chunksize=chunksize):
        for vs in varspecs:
            if not all(col in chunk.columns for col in [vs.pred, vs.truth, vs.gfs]):
                continue
            pred = pd.to_numeric(chunk[vs.pred], errors="coerce")
            truth = pd.to_numeric(chunk[vs.truth], errors="coerce")
            gfs = pd.to_numeric(chunk[vs.gfs], errors="coerce")
            valid = pred.notna() & truth.notna() & gfs.notna()
            if vs.mask and vs.mask in chunk.columns:
                m = chunk[vs.mask].fillna(False).astype(bool)
                valid &= m
            if not valid.any():
                continue

            p = pred[valid].to_numpy(dtype=np.float64)
            t = truth[valid].to_numpy(dtype=np.float64)
            g = gfs[valid].to_numpy(dtype=np.float64)

            out[vs.name]["pred"] = _reservoir_append(rng, out[vs.name]["pred"], p, max_points)
            out[vs.name]["truth"] = _reservoir_append(rng, out[vs.name]["truth"], t, max_points)
            out[vs.name]["gfs"] = _reservoir_append(rng, out[vs.name]["gfs"], g, max_points)

    return out


def plot_extra_eval(
    samples: dict[str, dict[str, np.ndarray]],
    instrument: str,
    init_time: str,
    vars_label: str,
    plot_dir: str,
    units_by_name: dict[str, str | None],
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    for name, d in samples.items():
        pred = d["pred"]
        truth = d["truth"]
        gfs = d["gfs"]
        if pred.size == 0:
            continue

        units = units_by_name.get(name)
        units_sfx = f" ({units})" if units else ""

        diffs = [
            (pred - truth, "OCELOT − Truth"),
            (pred - gfs, "OCELOT − GFS"),
            (truth - gfs, "Truth − GFS"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=160)
        for ax, (dd, lbl) in zip(axes, diffs):
            dd = dd[np.isfinite(dd)]
            if dd.size == 0:
                ax.set_axis_off()
                continue
            xlo, xhi = _robust_limits(dd, 1.0, 99.0)
            ax.hist(dd, bins=80, range=(xlo, xhi), alpha=0.85)
            ax.grid(True, linestyle="--", alpha=0.35)
            ax.set_title(lbl)
            ax.set_xlabel(f"Δ{units_sfx}")
            m = {
                "rmse": float(np.sqrt(np.mean(dd * dd))) if dd.size else float("nan"),
                "bias": float(np.mean(dd)) if dd.size else float("nan"),
            }
            ax.text(0.02, 0.95, f"N={dd.size}\nRMSE={m['rmse']:.3g}\nBias={m['bias']:.3g}", transform=ax.transAxes, va="top")

        fig.suptitle(f"{instrument} {name} • diffs ({vars_label}) • init {init_time}", y=1.02)
        out_png = os.path.join(plot_dir, f"eval_hist_{instrument}_{name}_init_{init_time}.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"Wrote: {out_png}")

        pairs = [
            (pred, truth, "OCELOT vs Truth"),
            (gfs, truth, "GFS vs Truth"),
            (pred, gfs, "OCELOT vs GFS"),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5), dpi=160)
        for ax, (x, y, ttl) in zip(axes, pairs):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            m = np.isfinite(x) & np.isfinite(y)
            if not m.any():
                ax.set_axis_off()
                continue
            xx, yy = x[m], y[m]
            lo, hi = _robust_limits(np.concatenate([xx, yy]), 1.0, 99.0)
            hb = ax.hexbin(xx, yy, gridsize=70, bins="log", mincnt=1)
            ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_title(ttl)
            ax.set_xlabel(f"X{units_sfx}")
            ax.set_ylabel(f"Y{units_sfx}")
            ms = _metrics(xx, yy)
            ax.text(
                0.02,
                0.95,
                f"N={ms['n']}\nRMSE={ms['rmse']:.3g}\nBias={ms['bias']:.3g}\nCorr={ms['corr']:.3f}",
                transform=ax.transAxes,
                va="top",
            )
            fig.colorbar(hb, ax=ax, pad=0.01)

        fig.suptitle(f"{instrument} {name} • scatter ({vars_label}) • init {init_time}", y=1.02)
        out_png = os.path.join(plot_dir, f"eval_scatter_{instrument}_{name}_init_{init_time}.png")
        plt.tight_layout()
        plt.savefig(out_png, bbox_inches="tight")
        plt.close()
        print(f"Wrote: {out_png}")


def _masked_preview_stats(chunk: pd.DataFrame, vs: VarSpec) -> dict:
    if not all(c in chunk.columns for c in [vs.truth, vs.gfs]):
        return {"n": 0}

    truth = pd.to_numeric(chunk[vs.truth], errors="coerce")
    gfs = pd.to_numeric(chunk[vs.gfs], errors="coerce")
    valid = truth.notna() & gfs.notna()
    if vs.mask and vs.mask in chunk.columns:
        m = chunk[vs.mask].fillna(False).astype(bool)
        valid &= m
    if not valid.any():
        return {"n": 0}

    t = truth[valid].to_numpy(dtype=np.float64)
    g = gfs[valid].to_numpy(dtype=np.float64)
    if t.size < 1000:
        return {"n": int(t.size)}

    def _p(x):
        return float(np.nanpercentile(x, 1.0)), float(np.nanpercentile(x, 99.0))

    t1, t99 = _p(t)
    g1, g99 = _p(g)
    return {
        "n": int(t.size),
        "t_p1": t1,
        "t_p99": t99,
        "g_p1": g1,
        "g_p99": g99,
        "t_range": float(t99 - t1),
        "g_range": float(g99 - g1),
    }


def _sanity_check_units(csv_path: str, varspecs: list[VarSpec], chunksize: int) -> None:
    preview = pd.read_csv(csv_path, nrows=1)
    needed: set[str] = set()
    for vs in varspecs:
        needed.update([vs.truth, vs.gfs])
        if vs.mask:
            needed.add(vs.mask)

    try:
        usecols = [c for c in needed if c in preview.columns]
        if not usecols:
            return
        chunk = next(_chunk_iter(csv_path, usecols=usecols, chunksize=min(int(chunksize), 200_000)))
    except Exception:
        return

    for vs in varspecs:
        s = _masked_preview_stats(chunk, vs)
        if s.get("n", 0) < 1000:
            continue

        if vs.name in {"u10", "v10", "u", "v"}:
            t_range = s.get("t_range", 0.0)
            g_range = s.get("g_range", 0.0)
            t_p99 = s.get("t_p99", 0.0)
            if (t_range is not None and g_range is not None) and (t_range < 5.0 and g_range > 10.0) and (t_p99 <= 0.5):
                print(
                    f"[WARN] {vs.name}: Truth-vs-GFS ranges look inconsistent on masked-valid rows. "
                    f"truth[p1,p99]=({s.get('t_p1'):.3f},{s.get('t_p99'):.3f}) vs gfs[p1,p99]=({s.get('g_p1'):.3f},{s.get('g_p99'):.3f}). "
                    "This often indicates a unit/encoding mismatch (e.g., windDirection radians treated as degrees upstream), "
                    "not a GFS lat/lon mapping bug."
                )


def compute_rmse_by_fhr(
    csv_path: str,
    varspecs: list[VarSpec],
    chunksize: int,
) -> pd.DataFrame:
    preview = pd.read_csv(csv_path, nrows=1)
    base_cols = ["fhr_used", "lead_hours_nominal"]
    needed_cols: set[str] = set(base_cols)
    for vs in varspecs:
        needed_cols.update([vs.pred, vs.truth, vs.gfs])
        if vs.mask:
            needed_cols.add(vs.mask)

    usecols = [c for c in needed_cols if c in preview.columns]

    sumsq: dict[tuple[int, str], float] = {}
    count: dict[tuple[int, str], int] = {}

    for chunk in _chunk_iter(csv_path, usecols=usecols, chunksize=chunksize):
        fhr = _get_fhr_series(chunk)
        fhr_rounded = pd.to_numeric(fhr.round().astype("Int64"), errors="coerce")

        for vs in varspecs:
            if not all(col in chunk.columns for col in [vs.pred, vs.truth, vs.gfs]):
                continue

            pred = pd.to_numeric(chunk[vs.pred], errors="coerce")
            truth = pd.to_numeric(chunk[vs.truth], errors="coerce")
            gfs = pd.to_numeric(chunk[vs.gfs], errors="coerce")

            valid = fhr_rounded.notna() & pred.notna() & truth.notna() & gfs.notna()
            if vs.mask and vs.mask in chunk.columns:
                m = chunk[vs.mask].fillna(False).astype(bool)
                valid &= m

            if not valid.any():
                continue

            fhr_vals = fhr_rounded[valid].astype(int).to_numpy()
            pred_vals = pred[valid].to_numpy(dtype=np.float64)
            truth_vals = truth[valid].to_numpy(dtype=np.float64)
            gfs_vals = gfs[valid].to_numpy(dtype=np.float64)

            diffs = {
                f"{vs.name}_pred_minus_true": pred_vals - truth_vals,
                f"{vs.name}_pred_minus_gfs": pred_vals - gfs_vals,
                f"{vs.name}_true_minus_gfs": truth_vals - gfs_vals,
            }

            for metric_name, d in diffs.items():
                d2 = d * d
                for fh in np.unique(fhr_vals):
                    sel = fhr_vals == fh
                    k = (int(fh), metric_name)
                    sumsq[k] = float(sumsq.get(k, 0.0) + np.nansum(d2[sel]))
                    count[k] = int(count.get(k, 0) + int(np.count_nonzero(np.isfinite(d2[sel]))))

    rows: list[dict] = []
    fhrs = sorted({k[0] for k in sumsq.keys()})
    metric_names = sorted({k[1] for k in sumsq.keys()})

    for fh in fhrs:
        row = {"fhr": fh}
        for mn in metric_names:
            k = (fh, mn)
            row[mn.replace("minus", "-")] = _rmse_from_sums(sumsq.get(k, 0.0), count.get(k, 0))
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("fhr") if rows else pd.DataFrame(columns=["fhr"])
    return out


def plot_rmse_table(rmse_df: pd.DataFrame, out_png: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    if rmse_df.empty or "fhr" not in rmse_df.columns:
        print(f"[WARN] No RMSE data to plot for {title}")
        return

    x = rmse_df["fhr"].to_numpy()
    ycols = [c for c in rmse_df.columns if c != "fhr"]

    plt.figure(figsize=(10, 6), dpi=160)
    for c in ycols:
        plt.plot(x, rmse_df[c].to_numpy(), marker="o", linewidth=2, label=c)

    plt.xlabel("Forecast hour")
    plt.ylabel("RMSE")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()


def _mapspecs_for(instrument: str, which: str) -> list[MapSpec]:
    want_wind = which in ("wind", "wind_temperature", "wind_pressure", "wind_temperature_pressure")
    want_temp = which in ("temperature", "wind_temperature", "temperature_pressure", "wind_temperature_pressure")
    want_pressure = which in ("pressure", "wind_pressure", "temperature_pressure", "wind_temperature_pressure")

    out: list[MapSpec] = []
    if instrument == "surface_obs":
        if want_wind:
            out.extend(
                [
                    MapSpec("u10", "pred_wind_u", "true_wind_u", "gfs_u10", "mask_wind_u", "m/s"),
                    MapSpec("v10", "pred_wind_v", "true_wind_v", "gfs_v10", "mask_wind_v", "m/s"),
                ]
            )
        if want_temp:
            out.append(MapSpec("t2m", "pred_airTemperature", "true_airTemperature", "gfs_t2m_C", "mask_airTemperature", "°C"))
        if want_pressure:
            out.append(
                MapSpec(
                    "sp",
                    "pred_pressureMeanSeaLevel_prepbufr",
                    "true_pressureMeanSeaLevel_prepbufr",
                    "gfs_sp_hPa",
                    "mask_pressureMeanSeaLevel_prepbufr",
                    "hPa",
                )
            )
        return out

    if instrument in ("radiosonde", "aircraft"):
        wind_u = "wind_u" if instrument == "radiosonde" else "windU"
        wind_v = "wind_v" if instrument == "radiosonde" else "windV"
        if want_wind:
            out.extend(
                [
                    MapSpec("u", f"pred_{wind_u}", f"true_{wind_u}", "gfs_u", f"mask_{wind_u}", "m/s"),
                    MapSpec("v", f"pred_{wind_v}", f"true_{wind_v}", "gfs_v", f"mask_{wind_v}", "m/s"),
                ]
            )
        if want_temp:
            out.append(MapSpec("temp", "pred_airTemperature", "true_airTemperature", "gfs_airTemperature_C", "mask_airTemperature", "°C"))
        return out

    raise ValueError(f"Unsupported instrument: {instrument}")


def _iter_chunks(csv_path: str, usecols: list[str], chunksize: int):
    return pd.read_csv(csv_path, usecols=usecols, chunksize=chunksize)


def _collect_points_for_fhr(
    csv_path: str,
    fhr: int,
    spec: MapSpec,
    chunksize: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    preview = pd.read_csv(csv_path, nrows=1)
    needed = ["lon", "lat", spec.pred_col, spec.truth_col, spec.gfs_col]
    if spec.mask_col:
        needed.append(spec.mask_col)
    base_cols = [c for c in ["fhr_used", "lead_hours_nominal"] if c in preview.columns]
    usecols = [c for c in (needed + base_cols) if c in preview.columns]

    if not all(c in preview.columns for c in ["lon", "lat", spec.pred_col, spec.truth_col, spec.gfs_col]):
        missing = [c for c in ["lon", "lat", spec.pred_col, spec.truth_col, spec.gfs_col] if c not in preview.columns]
        raise ValueError(f"Missing required columns in {os.path.basename(csv_path)}: {missing}")

    lon_all: list[np.ndarray] = []
    lat_all: list[np.ndarray] = []
    pred_all: list[np.ndarray] = []
    truth_all: list[np.ndarray] = []
    gfs_all: list[np.ndarray] = []

    fhr_target = float(fhr)

    for chunk in _iter_chunks(csv_path, usecols=usecols, chunksize=chunksize):
        fhrs = _get_fhr_series(chunk)
        fhrs = pd.to_numeric(fhrs.round(), errors="coerce")
        sel = fhrs.notna() & np.isclose(fhrs.to_numpy(dtype=float, na_value=np.nan), fhr_target)
        if not sel.any():
            continue

        lon = pd.to_numeric(chunk.loc[sel, "lon"], errors="coerce")
        lat = pd.to_numeric(chunk.loc[sel, "lat"], errors="coerce")
        pred = pd.to_numeric(chunk.loc[sel, spec.pred_col], errors="coerce")
        truth = pd.to_numeric(chunk.loc[sel, spec.truth_col], errors="coerce")
        gfs = pd.to_numeric(chunk.loc[sel, spec.gfs_col], errors="coerce")

        valid = lon.notna() & lat.notna() & pred.notna() & truth.notna() & gfs.notna()
        if spec.mask_col and spec.mask_col in chunk.columns:
            m = chunk.loc[sel, spec.mask_col].fillna(False).astype(bool)
            valid &= m

        if not valid.any():
            continue

        lon_all.append(lon[valid].to_numpy(dtype=np.float64))
        lat_all.append(lat[valid].to_numpy(dtype=np.float64))
        pred_all.append(pred[valid].to_numpy(dtype=np.float64))
        truth_all.append(truth[valid].to_numpy(dtype=np.float64))
        gfs_all.append(gfs[valid].to_numpy(dtype=np.float64))

    if not lon_all:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64),
        )

    return (
        np.concatenate(lon_all),
        np.concatenate(lat_all),
        np.concatenate(pred_all),
        np.concatenate(truth_all),
        np.concatenate(gfs_all),
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


def plot_tripanel_map(
    lon: np.ndarray,
    lat: np.ndarray,
    pred: np.ndarray,
    truth: np.ndarray,
    gfs: np.ndarray,
    title: str,
    out_png: str,
    units: str | None,
    point_size: int,
):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402

    if lon.size == 0:
        print(f"[WARN] No valid points for {title}; skipping")
        return

    m_ot = _metrics(pred, truth)
    m_gt = _metrics(gfs, truth)
    stats_line = (
        f"N={m_ot['n']}  "
        f"RMSE(O−T)={m_ot['rmse']:.3g}  Bias(O−T)={m_ot['bias']:.3g}  "
        f"RMSE(G−T)={m_gt['rmse']:.3g}  Bias(G−T)={m_gt['bias']:.3g}"
    )

    vmin, vmax = _robust_limits(np.concatenate([pred, truth, gfs]), q_lo=1.0, q_hi=99.0)

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), subplot_kw={"projection": ccrs.PlateCarree()}, sharey=True)
    panel_titles = ["OCELOT", "Truth", "GFS"]
    fields = [pred, truth, gfs]

    for ax, ttl, field in zip(axes, panel_titles, fields):
        ax.set_title(ttl, fontsize=14)
        sc = ax.scatter(lon, lat, c=field, s=point_size, cmap="turbo", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        _add_land(ax)
        ax.set_global()
        cb = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
        cb.set_label(f"Value{f' ({units})' if units else ''}")

    fig.suptitle(f"{title}\n{stats_line}", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_png}")


def plot_tripanel_diff(
    lon: np.ndarray,
    lat: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    out_png: str,
    units: str | None,
    point_size: int,
):
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: E402
    from matplotlib.colors import TwoSlopeNorm  # noqa: E402

    if lon.size == 0:
        print(f"[WARN] No valid points for {title}; skipping")
        return

    diff = a - b
    vmin, vmax = _robust_limits(np.concatenate([a, b]), q_lo=1.0, q_hi=99.0)
    dmin, dmax = _robust_sym_limits(diff, q_lo=1.0, q_hi=99.0)

    m = _metrics(a, b)
    stats_line = f"N={m['n']}  RMSE={m['rmse']:.3g}  Bias={m['bias']:.3g}  Corr={m['corr']:.3f}"

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), subplot_kw={"projection": ccrs.PlateCarree()}, sharey=True)
    panels = [
        (label_a, a, "turbo", None, f"Value{f' ({units})' if units else ''}"),
        (label_b, b, "turbo", None, f"Value{f' ({units})' if units else ''}"),
        (f"{label_a} − {label_b}", diff, "bwr", TwoSlopeNorm(vmin=dmin, vcenter=0.0, vmax=dmax), f"Δ{f' ({units})' if units else ''}"),
    ]

    for ax, (ttl, field, cmap, norm, cb_label) in zip(axes, panels):
        ax.set_title(ttl, fontsize=14)
        if norm is None:
            sc = ax.scatter(lon, lat, c=field, s=point_size, cmap=cmap, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        else:
            sc = ax.scatter(lon, lat, c=field, s=point_size, cmap=cmap, norm=norm, transform=ccrs.PlateCarree())
        _add_land(ax)
        ax.set_global()
        cb = fig.colorbar(sc, ax=ax, orientation="vertical", pad=0.02)
        cb.set_label(cb_label)

    fig.suptitle(f"{title}\n{stats_line}", fontsize=14, y=1.05)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Wrote: {out_png}")


def _varspecs_for(instrument: str, which: str) -> tuple[list[VarSpec], dict[str, str | None], str]:
    want_wind = which in ("wind", "wind_temperature", "wind_pressure", "wind_temperature_pressure")
    want_temp = which in ("temperature", "wind_temperature", "temperature_pressure", "wind_temperature_pressure")
    want_pressure = which in ("pressure", "wind_pressure", "temperature_pressure", "wind_temperature_pressure")

    if instrument == "surface_obs":
        varspecs: list[VarSpec] = []
        if want_wind:
            varspecs.extend(
                [
                    VarSpec("u10", "pred_wind_u", "true_wind_u", "gfs_u10", "mask_wind_u"),
                    VarSpec("v10", "pred_wind_v", "true_wind_v", "gfs_v10", "mask_wind_v"),
                ]
            )
        if want_temp:
            varspecs.append(
                VarSpec(
                    "t2m",
                    "pred_airTemperature",
                    "true_airTemperature",
                    "gfs_t2m_C",
                    "mask_airTemperature",
                )
            )
        if want_pressure:
            varspecs.append(
                VarSpec(
                    "sp",
                    "pred_pressureMeanSeaLevel_prepbufr",
                    "true_pressureMeanSeaLevel_prepbufr",
                    "gfs_sp_hPa",
                    "mask_pressureMeanSeaLevel_prepbufr",
                )
            )
        units_by_name: dict[str, str | None] = {"u10": "m/s", "v10": "m/s", "t2m": "°C", "sp": "hPa"}
        title = f"surface_obs {which} vs GFS (init {{init_time}})"
        return varspecs, units_by_name, title

    if instrument == "radiosonde":
        varspecs = []
        if want_wind:
            varspecs.extend(
                [
                    VarSpec("u", "pred_wind_u", "true_wind_u", "gfs_u", "mask_wind_u"),
                    VarSpec("v", "pred_wind_v", "true_wind_v", "gfs_v", "mask_wind_v"),
                ]
            )
        if want_temp:
            varspecs.append(
                VarSpec(
                    "temp",
                    "pred_airTemperature",
                    "true_airTemperature",
                    "gfs_airTemperature_C",
                    "mask_airTemperature",
                )
            )
        units_by_name = {"u": "m/s", "v": "m/s", "temp": "°C"}
        title = f"radiosonde {which} vs GFS (init {{init_time}})"
        return varspecs, units_by_name, title

    if instrument == "aircraft":
        varspecs = []
        if want_wind:
            varspecs.extend(
                [
                    VarSpec("u", "pred_windU", "true_windU", "gfs_u", "mask_windU"),
                    VarSpec("v", "pred_windV", "true_windV", "gfs_v", "mask_windV"),
                ]
            )
        if want_temp:
            varspecs.append(
                VarSpec(
                    "temp",
                    "pred_airTemperature",
                    "true_airTemperature",
                    "gfs_airTemperature_C",
                    "mask_airTemperature",
                )
            )
        units_by_name = {"u": "m/s", "v": "m/s", "temp": "°C"}
        title = f"aircraft {which} vs GFS (init {{init_time}})"
        return varspecs, units_by_name, title

    raise ValueError(f"Unsupported instrument: {instrument}")


def _run_rmse(
    *,
    init_time: str,
    instrument: str,
    which: str,
    csv_path: str,
    plot_dir: str,
    chunksize: int,
    extra_eval: bool,
) -> None:
    varspecs, units_by_name, title_tmpl = _varspecs_for(instrument, which)
    title = title_tmpl.format(init_time=init_time)

    _sanity_check_units(csv_path, varspecs, chunksize=int(chunksize))

    if not varspecs:
        raise SystemExit(f"No variable specs selected for instrument={instrument!r}, vars={which!r}")

    rmse_df = compute_rmse_by_fhr(csv_path, varspecs=varspecs, chunksize=int(chunksize))

    suffix = "" if which == "wind" else f"_{which}"
    out_csv = os.path.join(plot_dir, f"rmse_vs_gfs_{instrument}{suffix}_init_{init_time}.csv")
    rmse_df.to_csv(out_csv, index=False)
    print(f"Wrote: {out_csv}")

    out_png = os.path.join(plot_dir, f"rmse_vs_gfs_{instrument}{suffix}_init_{init_time}.png")
    plot_rmse_table(rmse_df, out_png=out_png, title=title)
    print(f"Wrote: {out_png}")

    if not extra_eval:
        return

    # Extra evaluation plots (OCELOT vs Truth vs GFS) using sampled values.
    try:
        samples = sample_values(
            csv_path,
            varspecs=varspecs,
            chunksize=int(chunksize),
            max_points=200_000,
            seed=7,
        )
        plot_extra_eval(
            samples,
            instrument=instrument,
            init_time=init_time,
            vars_label=which,
            plot_dir=plot_dir,
            units_by_name=units_by_name,
        )
    except Exception as e:
        print(f"[WARN] Failed to make extra eval plots: {e}")


def _run_maps(
    *,
    init_time: str,
    instrument: str,
    which: str,
    csv_path: str,
    plot_dir: str,
    chunksize: int,
    fhrs: list[int],
    point_size: int,
) -> None:
    _require_plotting()

    specs = _mapspecs_for(instrument, which)
    if not specs:
        print(f"[WARN] No variables selected for instrument={instrument} vars={which}")
        return

    for fhr in fhrs:
        fhr_dir = os.path.join(plot_dir, f"fhr{int(fhr)}")
        os.makedirs(fhr_dir, exist_ok=True)

        for spec in specs:
            lon, lat, pred, truth, gfs = _collect_points_for_fhr(
                csv_path=csv_path,
                fhr=int(fhr),
                spec=spec,
                chunksize=int(chunksize),
            )

            title = f"{instrument} {spec.name} • init {init_time} • fhr {int(fhr):d}"
            out_png = os.path.join(
                fhr_dir,
                f"{instrument}_Ocelot_Truth_GFS_{spec.name}_init_{init_time}_fhr{int(fhr):03d}.png",
            )
            plot_tripanel_map(
                lon=lon,
                lat=lat,
                pred=pred,
                truth=truth,
                gfs=gfs,
                title=title,
                out_png=out_png,
                units=spec.units,
                point_size=int(point_size),
            )

            out_png_ot = os.path.join(
                fhr_dir,
                f"{instrument}_Ocelot_Truth_Diff_{spec.name}_init_{init_time}_fhr{int(fhr):03d}.png",
            )
            plot_tripanel_diff(
                lon=lon,
                lat=lat,
                a=pred,
                b=truth,
                label_a="OCELOT",
                label_b="Truth",
                title=title,
                out_png=out_png_ot,
                units=spec.units,
                point_size=int(point_size),
            )

            out_png_gt = os.path.join(
                fhr_dir,
                f"{instrument}_GFS_Truth_Diff_{spec.name}_init_{init_time}_fhr{int(fhr):03d}.png",
            )
            plot_tripanel_diff(
                lon=lon,
                lat=lat,
                a=gfs,
                b=truth,
                label_a="GFS",
                label_b="Truth",
                title=title,
                out_png=out_png_gt,
                units=spec.units,
                point_size=int(point_size),
            )


def main() -> int:
    ap = argparse.ArgumentParser(description="Combined GFS-compare plotting: RMSE curves + per-fhr maps")
    ap.add_argument("--init_time", required=True, help="YYYYMMDDHH")
    ap.add_argument("--data_dir", required=True, help="Directory containing *_vs_gfs.csv")
    ap.add_argument("--plot_dir", required=True, help="Base output directory")
    ap.add_argument("--instrument", default="surface_obs", choices=["surface_obs", "radiosonde", "aircraft"])
    ap.add_argument(
        "--vars",
        default="wind",
        choices=[
            "wind",
            "temperature",
            "pressure",
            "wind_temperature",
            "wind_pressure",
            "temperature_pressure",
            "wind_temperature_pressure",
        ],
    )
    ap.add_argument("--chunksize", type=int, default=200_000)

    rmse_group = ap.add_mutually_exclusive_group()
    rmse_group.add_argument(
        "--skip_rmse",
        action="store_true",
        help="Skip RMSE-vs-fhr outputs (use with --maps to generate maps only).",
    )
    rmse_group.add_argument(
        "--rmse",
        action="store_true",
        help="(Deprecated) RMSE runs by default; keep for compatibility.",
    )
    ap.add_argument("--maps", action="store_true", help="Generate per-fhr map panels under plot_dir/fhr*/")
    ap.add_argument("--fhr", type=int, default=None, help="Single forecast hour (maps)")
    ap.add_argument("--fhrs", type=int, nargs="*", default=None, help="Forecast hours list (maps)")
    ap.add_argument("--point_size", type=int, default=7)
    ap.add_argument(
        "--skip_extra_eval",
        action="store_true",
        help="Skip extra hist/scatter evaluation plots (RMSE task only).",
    )

    args = ap.parse_args()

    # Default: RMSE always runs unless explicitly skipped.
    want_rmse = not bool(args.skip_rmse)

    _ensure_dir(args.plot_dir)

    pattern = os.path.join(args.data_dir, f"*{args.instrument}*init_{args.init_time}*_vs_gfs.csv")
    csv_path = _find_one(pattern)
    print(f"Using: {csv_path}")

    if want_rmse:
        _run_rmse(
            init_time=str(args.init_time),
            instrument=str(args.instrument),
            which=str(args.vars),
            csv_path=csv_path,
            plot_dir=str(args.plot_dir),
            chunksize=int(args.chunksize),
            extra_eval=not bool(args.skip_extra_eval),
        )

    # If the user passed fhr(s), enable maps unless explicitly disabled.
    fhrs: list[int] = []
    if args.fhrs:
        fhrs.extend(list(args.fhrs))
    if args.fhr is not None:
        fhrs.append(int(args.fhr))
    fhrs = sorted(_unique_ints(fhrs))

    if args.maps or fhrs:
        if not fhrs:
            raise SystemExit("--maps requires --fhr or --fhrs")
        _run_maps(
            init_time=str(args.init_time),
            instrument=str(args.instrument),
            which=str(args.vars),
            csv_path=csv_path,
            plot_dir=str(args.plot_dir),
            chunksize=int(args.chunksize),
            fhrs=fhrs,
            point_size=int(args.point_size),
        )

    if (not want_rmse) and (not (args.maps or fhrs)):
        raise SystemExit("Nothing to do: set RMSE (default) and/or --maps with --fhr/--fhrs")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
