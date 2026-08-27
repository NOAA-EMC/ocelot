#!/usr/bin/env python
"""eval_plots.py

Figure functions for OCELOT evaluation, plus the data-preparation helpers they
share (feature discovery, QC masking, lead selection, colour limits).

Originally authored by Azadeh Gholoubi as part of evaluations.py; restructured
so that the single-lead and horizon-aggregate variants of every figure are one
function rather than two near-identical copies.

Nothing in this module touches the filesystem to *find* inputs -- callers hand
it an already-loaded DataFrame via a PlotCtx. See evaluations.py.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

try:
    import cartopy.crs as ccrs
except Exception:  # pragma: no cover
    ccrs = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
except Exception:  # pragma: no cover
    plt = None
    TwoSlopeNorm = None


# =============================================================================
# Constants carried over from the original module
# =============================================================================

# NOTE: the per-feature tuning tables (auto_absolute, tiny_threshold,
# calm_wind_threshold, qc_ranges) used to live here as module constants with no
# way to reach them from the config. They are now in plotting.yaml under
# `features:` and are read through PlotCtx.

PRESSURE_COL_CANDIDATES = [
    "pressure_hPa", "pressure_hpa", "pressureMeanSeaLevel", "airPressure", "pressure",
]
PRESSURE_LEVEL_CANDIDATES = ["pressure_level_idx", "pressure_level_index"]
PRESSURE_LABEL_CANDIDATES = ["pressure_level_label", "level_label"]

STANDARD_PRESSURE_LEVELS = [
    1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10,
]

# Instruments disagree on spelling: radiosonde writes wind_u/wind_v, aircraft
# writes windU/windV. features.auto_absolute carries both, which is why the old
# hardcoded snake_case lookup in _plot_wind silently skipped aircraft.
WIND_COMPONENT_ALIASES = (("wind_u", "wind_v"), ("windU", "windV"))


def wind_columns(df) -> tuple[str, str] | None:
    """Return the (u, v) feature names present in this frame, or None."""
    for u, v in WIND_COMPONENT_ALIASES:
        need = {f"true_{u}", f"true_{v}", f"pred_{u}", f"pred_{v}"}
        if need.issubset(df.columns):
            return u, v
    return None


def require_plotting() -> None:
    if ccrs is None or plt is None or TwoSlopeNorm is None:
        raise RuntimeError(
            "Plotting dependencies (cartopy/matplotlib) are unavailable. "
            "Run with mode=metrics or install cartopy+matplotlib."
        )


# =============================================================================
# Context object handed to every figure function
# =============================================================================

@dataclass
class PlotCtx:
    """Everything a figure function needs, with the CSV already read once."""

    df: pd.DataFrame
    instrument: str
    spec: Any                       # InstrumentSpec from evaluations.py
    fig_dir: str
    base_filename_tag: str          # e.g. "_init_2025030100_epoch_350"
    base_title_tag: str             # e.g. " - Init 2025030100 - Epoch 350"
    ar_step: int | None = None
    cfg: dict = field(default_factory=dict)
    shared_limits: dict = field(default_factory=dict)
    metrics_rows: list = field(default_factory=list)
    written_tables: set = field(default_factory=set)

    def opt(self, path: str):
        """Dotted lookup into the config, e.g. ctx.opt('render.dpi').

        Deliberately has no default argument. A per-call-site fallback would be
        a third place where a value can live, alongside plotting.yaml and the
        CLI; a missing key is an error instead.
        """
        node = self.cfg
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                raise KeyError(
                    f"Missing config key '{path}'. Add it to plotting.yaml."
                )
            node = node[part]
        return node


# =============================================================================
# Small numeric helpers
# =============================================================================

def np_(x) -> np.ndarray:
    return pd.to_numeric(x, errors="coerce").to_numpy()


def smape(p, t, eps: float = 1e-6):
    return 200.0 * np.abs(p - t) / (np.abs(p) + np.abs(t) + eps)


def shortest_arc_deg(a, b):
    """Absolute shortest angular difference in degrees, in [0, 180]."""
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def first_existing(df: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def is_surface_pressure_feature(name: str) -> bool:
    return name in {"airPressure", "pressureMeanSeaLevel_prepbufr"}


def print_sanity(name, t, p, tiny=None) -> None:
    ae = np.abs(p - t)
    sp_all = smape(p, t)
    med_ae = float(np.nanmedian(ae))
    p95_ae = float(np.nanpercentile(ae, 95))
    med_sp = float(np.nanmedian(sp_all))
    p95_sp = float(np.nanpercentile(sp_all, 95))
    dropped = 0
    if tiny is not None:
        mask_rel = np.abs(t) >= tiny
        dropped = int((~mask_rel).sum())
        if mask_rel.any():
            sp = sp_all[mask_rel]
            med_sp = float(np.nanmedian(sp))
            p95_sp = float(np.nanpercentile(sp, 95))
    print(
        f"{name:20s} | N={t.size:6d} | AbsErr med/95%={med_ae:6.2f}/{p95_ae:6.2f} "
        f"| sMAPE% med/95%={med_sp:6.1f}/{p95_sp:6.1f} | dropped<tiny={dropped}"
    )


# =============================================================================
# Feature discovery and channel selection
# =============================================================================

def discover_features(df: pd.DataFrame, num_channels: int = 9999) -> list[str]:
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    feats = [c[len("pred_"):] for c in pred_cols]
    return feats if feats else [f"ch{i}" for i in range(1, num_channels + 1)]


def select_features(feats: list[str], wanted: Iterable | None) -> list[str]:
    """Filter discovered features by name or by 1-based channel number.

    `wanted` entries may be feature names ('airTemperature'), channel names
    ('ch5'), or bare integers (5 -> the 5th discovered feature).
    """
    if not wanted:
        return feats

    by_name = {f.lower(): f for f in feats}
    out: list[str] = []
    missing: list[str] = []

    for w in wanted:
        if isinstance(w, bool):
            continue
        if isinstance(w, (int, np.integer)):
            idx = int(w) - 1
            if 0 <= idx < len(feats):
                out.append(feats[idx])
            else:
                missing.append(str(w))
            continue
        s = str(w).strip()
        if s.lower() in by_name:
            out.append(by_name[s.lower()])
            continue
        m = re.fullmatch(r"(?:ch|channel)[_-]?(\d+)", s, flags=re.IGNORECASE)
        if m:
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(feats):
                out.append(feats[idx])
                continue
        missing.append(s)

    if missing:
        print(f"  [WARN] Requested channels not found: {', '.join(missing)}")
        print(f"         Available: {', '.join(feats)}")

    seen: set[str] = set()
    return [f for f in out if not (f in seen or seen.add(f))]


# =============================================================================
# Quality control -- one implementation used by every figure family
# =============================================================================

def apply_qc(df: pd.DataFrame, instrument: str, feature: str, *,
             need_truth: bool = True, cfg: dict | None = None) -> np.ndarray:
    """Boolean row mask combining finiteness, QC masks and physical ranges."""
    pred_col = f"pred_{feature}"
    true_col = f"true_{feature}"

    n = len(df)
    valid = np.ones(n, dtype=bool)

    if pred_col not in df.columns:
        return np.zeros(n, dtype=bool)

    p = np_(df[pred_col])
    valid &= np.isfinite(p)

    t = None
    if need_truth:
        if true_col not in df.columns:
            return np.zeros(n, dtype=bool)
        t = np_(df[true_col])
        valid &= np.isfinite(t)

    for col in ("lon", "lat"):
        if col not in df.columns:
            return np.zeros(n, dtype=bool)
        valid &= np.isfinite(np_(df[col]))

    mask_col = f"mask_{feature}"
    if mask_col in df.columns:
        valid &= df[mask_col].fillna(False).astype(bool).to_numpy()

    if instrument in ("radiosonde", "aircraft"):
        pcol = first_existing(df, PRESSURE_COL_CANDIDATES)
        if pcol is not None:
            pressure = np_(df[pcol])
            valid &= np.isfinite(pressure) & (pressure >= 10) & (pressure <= 1100)

    qc_ranges = ((cfg or {}).get("features") or {}).get("qc_ranges") or {}
    if instrument == "surface_obs" and feature in qc_ranges:
        lo, hi = qc_ranges[feature]
        valid &= (p >= lo) & (p <= hi)
        if t is not None:
            valid &= (t >= lo) & (t <= hi)

    return valid


# =============================================================================
# Lead-time selection: sub-windows and horizons
# =============================================================================

def leads_present(df: pd.DataFrame) -> list[int]:
    if "lead_hours_nominal" not in df.columns:
        return []
    lead = pd.to_numeric(df["lead_hours_nominal"], errors="coerce").to_numpy(dtype=float)
    return sorted({int(round(x)) for x in lead[np.isfinite(lead)]})


def infer_step_hours(df: pd.DataFrame) -> int | None:
    """Smallest positive gap between distinct nominal leads."""
    lead = leads_present(df)
    if len(lead) < 2:
        return None
    diffs = np.diff(np.asarray(lead, dtype=float))
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    step = float(np.nanmin(diffs))
    return int(round(step)) if np.isfinite(step) and step > 0 else None


def select_leads(df: pd.DataFrame, leads: int | Sequence[int] | None) -> pd.DataFrame:
    """Row-filter to one lead or a set of leads.

    Unlike the original _filter_df_by_lead_hours_set, an empty intersection is
    NOT silently replaced by "all leads present". That fallback mislabelled AR
    files, because an AR2 file asked for leads 3/6/9/12 would quietly return
    its 27/30/33/36 rows tagged as a 0-12h horizon.
    """
    if leads is None or "lead_hours_nominal" not in df.columns:
        return df

    want = [leads] if isinstance(leads, (int, np.integer)) else list(leads)
    lead = pd.to_numeric(df["lead_hours_nominal"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(lead)

    mask = np.zeros(len(df), dtype=bool)
    for L in want:
        mask |= finite & np.isclose(lead, float(L))
    return df if mask.all() else df.loc[mask].copy()


def tile_horizons(leads: list[int], step_hours: int | None,
                  horizon_length: int) -> list[list[int]]:
    """Group nominal leads into contiguous blocks of `horizon_length` hours.

    Each lead L covers the window (L - step, L]. Leads are bucketed by which
    horizon_length-sized block their window START falls in, so leads 3/6/9/12
    with step 3 form one 0-12h block, and 27/30/33/36 form one 24-36h block.
    """
    if not leads:
        return []
    step = step_hours or 0
    if horizon_length <= 0:
        return [sorted(leads)]

    buckets: dict[int, list[int]] = {}
    for L in sorted(leads):
        start = max(int(L) - int(step), 0)
        k = start // int(horizon_length)
        buckets.setdefault(k, []).append(int(L))
    return [buckets[k] for k in sorted(buckets)]


def horizon_bounds(leads: list[int], step_hours: int | None) -> tuple[int, int]:
    if not leads:
        return 0, 0
    end_h = int(max(leads))
    if step_hours is None or step_hours <= 0:
        return 0, end_h
    return max(int(min(leads)) - int(step_hours), 0), end_h


def window_tags(ctx: PlotCtx, sel, kind: str, step_hours: int | None,
                filtered: bool = False) -> tuple[str, str]:
    """Build (filename_tag, title_tag) for a sub-window or horizon selection.

    Reproduces the tag strings emitted by the original code so that existing
    figure filenames are preserved.
    """
    fn = ctx.base_filename_tag
    tt = ctx.base_title_tag

    if kind == "none" or sel is None:
        # No lead metadata in the file (legacy mesh dumps). The forecast hour,
        # if any, is already in the base tag from the filename.
        return fn, tt

    if kind == "sub":
        L = int(sel)
        # Guard against doubling up when the filename already carried _fNNN.
        if f"_f{L:03d}" not in fn:
            fn += f"_f{L:03d}"
            tt += f" - F{L:03d}"
        if filtered:
            tt += f" - Nominal lead {L}h"
        if step_hours and step_hours > 0:
            s = max(L - int(step_hours), 0)
            fn += f"_window_{s:02d}h_{L:02d}h"
            tt += f" - Window {s:02d}-{L:02d}h after init"
        return fn, tt

    leads = [int(x) for x in sel]
    s, e = horizon_bounds(leads, step_hours)
    fn += f"_horizon_{s:02d}h_{e:02d}h"
    step_str = f"{int(step_hours)}h" if step_hours is not None else "?h"
    leads_str = ", ".join(str(L) for L in leads)
    tt += (
        f" - Horizon {s:02d}-{e:02d}h after init"
        f" (aggregated {step_str} windows; nominal leads {leads_str}h)"
    )
    return fn, tt


def resolve_selections(ctx: PlotCtx) -> list[tuple[Any, str, int | None]]:
    """Return [(selection, kind, step_hours), ...] for the configured windows."""
    df = ctx.df
    step = infer_step_hours(df)
    present = leads_present(df)
    out: list[tuple[Any, str, int | None]] = []

    if not present:
        # No lead metadata (e.g. legacy mesh-grid files): one unfiltered pass.
        return [(None, "none", None)]

    if ctx.opt("windows.subwindows"):
        wanted = ctx.opt("windows.subwindow_leads")
        subs = present if (wanted in (None, "auto")) else [
            int(x) for x in wanted if int(x) in present
        ]
        out.extend((L, "sub", step) for L in subs)

    if ctx.opt("windows.horizons"):
        bounds = ctx.opt("windows.horizon_bounds")
        length = int(ctx.opt("windows.horizon_length_hours"))
        if bounds in (None, "auto"):
            blocks = tile_horizons(present, step, length)
        else:
            blocks = []
            for lo, hi in bounds:
                blk = [L for L in present if lo < L <= hi]
                if blk:
                    blocks.append(blk)
        out.extend((b, "horizon", step) for b in blocks)

    return out or [(None, "none", None)]


def check_obs_window(df: pd.DataFrame, start_h: int, end_h: int, strict: bool) -> None:
    """Warn (or raise) when obs_time_unix falls outside the expected window."""
    if not {"obs_time_unix", "init_time_unix"}.issubset(df.columns):
        return
    try:
        obs = pd.to_numeric(df["obs_time_unix"], errors="coerce").to_numpy(dtype=float)
        init = pd.to_numeric(df["init_time_unix"], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(obs) & np.isfinite(init) & (obs >= 0) & (init >= 0)
        if not np.any(m):
            return
        dsec = obs[m] - init[m]
        dh = dsec / 3600.0
        print(
            f"  obs_time offsets (h after init): min={float(np.nanmin(dh)):.3f} "
            f"max={float(np.nanmax(dh)):.3f} (N={int(dh.size)})"
        )
        tol = 1.0
        outside = (dsec < start_h * 3600.0 - tol) | (dsec > end_h * 3600.0 + tol)
        n_out = int(np.sum(outside))
        if n_out:
            print(
                f"  [WARN] {n_out}/{int(dh.size)} obs times outside expected window "
                f"[{start_h:02d},{end_h:02d}]h after init"
            )
            if strict:
                raise ValueError("obs_time_unix outside expected window")
    except Exception as e:
        if strict:
            raise
        print(f"  [WARN] obs-time check failed: {e}")


# =============================================================================
# Colour limits
# =============================================================================

def robust_sym_limits(x, q: float = 99.0) -> tuple[float, float]:
    """Symmetric limits [-m, m] from the qth percentile of |x|."""
    x = np.asarray(x)
    if x.size == 0 or not np.isfinite(x).any():
        return -1.0, 1.0
    m = float(np.nanpercentile(np.abs(x), q))
    if not np.isfinite(m) or m == 0:
        m = float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else 1.0
    return (-m, m) if m else (-1.0, 1.0)


def _fixed_lookup(ctx: PlotCtx, feature: str, which: str):
    fixed = ctx.opt("limits.fixed") or {}
    inst = fixed.get(ctx.instrument)
    if not isinstance(inst, dict):
        return None
    feat = inst.get(feature)
    if isinstance(feat, dict) and which in feat:
        return feat[which]
    if which in inst and not isinstance(inst[which], dict):
        return inst[which]
    return None


def value_limits(ctx: PlotCtx, feature: str, *arrays) -> tuple[float, float]:
    """Resolve the shared colour limits for the truth and prediction panels."""
    key = (ctx.instrument, feature, "value")
    fixed = _fixed_lookup(ctx, feature, "value")
    if fixed:
        return float(fixed[0]), float(fixed[1])
    if ctx.opt("limits.share_across_ar_steps") and key in ctx.shared_limits:
        return ctx.shared_limits[key]

    stacked = np.concatenate([np.asarray(a).ravel() for a in arrays if a is not None])
    stacked = stacked[np.isfinite(stacked)]
    if stacked.size == 0:
        return 0.0, 1.0

    mode = str(ctx.opt("limits.mode")).lower()
    if mode == "robust":
        q = float(ctx.opt("limits.robust_percentile"))
        lo = float(np.nanpercentile(stacked, 100.0 - q))
        hi = float(np.nanpercentile(stacked, q))
        if lo == hi:
            lo, hi = float(np.nanmin(stacked)), float(np.nanmax(stacked))
    else:
        lo, hi = float(np.nanmin(stacked)), float(np.nanmax(stacked))

    if lo == hi:
        hi = lo + 1.0
    if ctx.opt("limits.share_across_ar_steps"):
        ctx.shared_limits[key] = (lo, hi)
    return lo, hi


def error_limits(ctx: PlotCtx, feature: str, diff) -> tuple[float, float]:
    fixed = _fixed_lookup(ctx, feature, "error")
    if fixed:
        return float(fixed[0]), float(fixed[1])
    key = (ctx.instrument, feature, "error")
    if ctx.opt("limits.share_across_ar_steps") and key in ctx.shared_limits:
        return ctx.shared_limits[key]
    lim = robust_sym_limits(diff, q=float(ctx.opt("limits.robust_percentile")))
    if ctx.opt("limits.share_across_ar_steps"):
        ctx.shared_limits[key] = lim
    return lim


# =============================================================================
# Shared axes construction
# =============================================================================

def _add_land(ax) -> None:
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)


def make_axes_triple(title: str, figsize=(20, 5), suptitle_y=1.02):
    """Three PlateCarree panels in a row, sized to match the diff figure.

    The old (20, 6) canvas left a wide band under the title: GeoAxes hold a
    fixed 2:1 aspect, so three of them across 20in are only ~3.3in tall and get
    centred vertically, stranding the extra height as whitespace. Matching the
    diff figure's height and pinning the suptitle just above the axes closes it.
    """
    fig, axes = plt.subplots(
        1, 3, figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()}, sharey=True,
    )
    fig.suptitle(title, fontsize=16, y=suptitle_y)
    return fig, axes


def _finish_geo(axes) -> None:
    for ax in axes:
        ax.set_global()
        _add_land(ax)
        ax.set_xlabel("Longitude")
    axes[0].set_ylabel("Latitude")


def _save(fig_dir: str, filename: str, ctx: PlotCtx, *, tight_bbox=True) -> None:
    os.makedirs(fig_dir, exist_ok=True)
    out = os.path.join(fig_dir, filename)
    dpi = int(ctx.opt("render.dpi"))
    if tight_bbox:
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
    else:
        plt.savefig(out, dpi=dpi)
    plt.close()
    print(f"  -> Saved: {out}")


def save_level_skill(ctx: PlotCtx, sub: pd.DataFrame, feature: str, fn_tag: str,
                     level_col: str | None, label_col: str | None) -> None:
    """Write the pressure-level skill table once per instrument/feature/window.

    Both plot_error_map and plot_profiles used to build this table from the
    same frame with the same defaults, producing byte-identical files. In the
    original they shared a filename and silently overwrote each other; now the
    first family to run writes it and the second is a no-op.
    """
    if level_col is None:
        return
    out = os.path.join(
        ctx.fig_dir, f"{ctx.instrument}_{feature}{fn_tag}_level_skill.csv")
    if out in ctx.written_tables:
        return

    lvl_df = metrics_by_pressure_level(sub, feature, level_col, label_col)
    if lvl_df is None or lvl_df.empty:
        return

    os.makedirs(ctx.fig_dir, exist_ok=True)
    lvl_df.to_csv(out, index=False)
    ctx.written_tables.add(out)
    print(f"  -> Saved pressure-level skill table: {out}")


def _record_metric(ctx: PlotCtx, feature: str, kind: str, sel, t, p,
                   level: str | None = None) -> None:
    if not ctx.opt("figures.metrics_table"):
        return
    d = p - t
    ctx.metrics_rows.append({
        "instrument": ctx.instrument,
        "feature": feature,
        "ar_step": ctx.ar_step,
        "window_kind": kind,
        "level": level,
        "lead_hours": (sel if isinstance(sel, (int, np.integer))
                       else (max(sel) if sel else None)),
        "n": int(t.size),
        "rmse": float(np.sqrt(np.nanmean(d ** 2))),
        "mae": float(np.nanmean(np.abs(d))),
        "bias": float(np.nanmean(d)),
    })


# =============================================================================
# Figure family 1: OCELOT / Target / signed Difference
# =============================================================================

def plot_diff(ctx: PlotCtx) -> None:
    """Three-panel prediction / truth / signed-difference maps.

    Replaces plot_ocelot_target_diff and plot_ocelot_target_diff_12h_horizon.

    For radiosonde and aircraft this stratifies by pressure level, exactly as
    plot_error_map does. A single mixed-level diff map is meaningless for a
    profile instrument: 1000 hPa through 10 hPa obs share the same station
    lat/lon, so points overplot in row order and the RMSE/bias badge is a
    vertical average over the whole column.
    """
    require_plotting()
    units = ctx.spec.units
    ps = int(ctx.opt("render.point_size"))
    cmap_v = ctx.opt("render.cmap_value")
    cmap_d = ctx.opt("render.cmap_diff")
    strict = bool(ctx.opt("windows.strict_obs_window"))
    min_pts = int(ctx.opt("render.min_points_per_level"))

    is_profile_inst = ctx.instrument in ("radiosonde", "aircraft")
    do_levels = bool(ctx.opt("figures.pressure_levels")) and is_profile_inst

    feats = select_features(
        discover_features(ctx.df, ctx.spec.n_channels), ctx.spec.channels
    )

    for sel, kind, step in resolve_selections(ctx):
        sub = select_leads(ctx.df, sel)
        if len(sub) == 0:
            print(f"[WARN] No rows for lead selection {sel}; skipping.")
            continue

        filtered = len(sub) != len(ctx.df)
        fn_tag, tt_tag = window_tags(ctx, sel, kind, step, filtered)

        if kind == "horizon":
            s, e = horizon_bounds([int(x) for x in sel], step)
            check_obs_window(sub, s, e, strict)

        panel_titles = (
            ["OCELOT (all leads)", "Target (all leads)", "Difference (all leads)"]
            if kind == "horizon" else ["OCELOT", "Target", "Difference"]
        )

        lvl_labels = None
        if do_levels:
            level_col = first_existing(sub, PRESSURE_LEVEL_CANDIDATES)
            label_col = first_existing(sub, PRESSURE_LABEL_CANDIDATES)
            lvl_labels = _level_series(sub, level_col, label_col)
            if lvl_labels is None:
                print("[WARN] Per-level diff requested but no pressure level column "
                      "found; falling back to a mixed-level figure.")

        for fname in feats:
            base_valid = apply_qc(sub, ctx.instrument, fname,
                                  need_truth=True, cfg=ctx.cfg)
            if not np.any(base_valid):
                print(f"  Info: no valid rows for '{fname}' ({kind}). Skipping.")
                continue

            t_all = np_(sub[f"true_{fname}"])
            p_all = np_(sub[f"pred_{fname}"])
            lon_all = np_(sub["lon"])
            lat_all = np_(sub["lat"])

            def render(tag_dir: str, tag_suffix: str, mask: np.ndarray,
                       min_points: int) -> None:
                if not np.any(mask) or int(mask.sum()) < int(min_points):
                    return

                t, p = t_all[mask], p_all[mask]
                lon, lat = lon_all[mask], lat_all[mask]
                print(f"  {fname}{tag_suffix}: {int(mask.sum())}/{len(sub)} obs retained")

                diff = p - t
                rmse = float(np.sqrt(np.nanmean(diff ** 2)))
                _record_metric(ctx, fname, kind, sel, t, p,
                               level=(tag_suffix.lstrip("_") or None))

                vmin, vmax = value_limits(ctx, fname, t, p)
                dmin, dmax = error_limits(ctx, fname, diff)
                norm = TwoSlopeNorm(vmin=dmin, vcenter=0.0, vmax=dmax)

                fig, axes = plt.subplots(
                    1, 3, figsize=(20, 5),
                    subplot_kw={"projection": ccrs.PlateCarree()}, sharey=True,
                )
                for ax, ttl in zip(axes, panel_titles):
                    ax.set_title(ttl, fontsize=14)
                fig.suptitle(f"{ctx.instrument} - {fname}{tt_tag}{tag_suffix}",
                             fontsize=16, y=1.02)

                unit_sfx = f" ({units})" if units else ""
                sc0 = axes[0].scatter(lon, lat, c=p, s=ps, cmap=cmap_v, vmin=vmin,
                                      vmax=vmax, transform=ccrs.PlateCarree())
                fig.colorbar(sc0, ax=axes[0], orientation="vertical",
                             pad=0.02).set_label(f"Value{unit_sfx}")

                sc1 = axes[1].scatter(lon, lat, c=t, s=ps, cmap=cmap_v, vmin=vmin,
                                      vmax=vmax, transform=ccrs.PlateCarree())
                fig.colorbar(sc1, ax=axes[1], orientation="vertical",
                             pad=0.02).set_label(f"Value{unit_sfx}")

                sc2 = axes[2].scatter(lon, lat, c=diff, s=ps, cmap=cmap_d, norm=norm,
                                      transform=ccrs.PlateCarree())
                fig.colorbar(sc2, ax=axes[2], orientation="vertical",
                             pad=0.02).set_label(f"Pred - True{unit_sfx}")

                stats_kw = dict(
                    transform=axes[2].transAxes, va="top", zorder=10,
                    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                              alpha=0.8, linewidth=0),
                )
                axes[2].text(
                    0.02, 0.98,
                    f"RMSE = {rmse:.2f}{f' {units}' if units else ''}",
                    ha="left", **stats_kw,
                )
                axes[2].text(
                    0.98, 0.98,
                    f"Bias = {float(np.nanmean(diff)):+.2f}",
                    ha="right", **stats_kw,
                )

                _finish_geo(axes)
                plt.tight_layout()
                safe = str(fname).replace(" ", "_")
                _save(tag_dir,
                      f"{ctx.instrument}_OCELOT_Target_Diff_{safe}"
                      f"{fn_tag}{tag_suffix}.png", ctx)

            if do_levels and lvl_labels is not None:
                for lvl in sorted_level_labels(lvl_labels):
                    m = base_valid & (lvl_labels.astype(str) == str(lvl)).to_numpy()
                    render(os.path.join(ctx.fig_dir, f"level_{lvl}"),
                           f"_{lvl}", m, min_pts)
            else:
                render(ctx.fig_dir, "", base_valid, 1)


# =============================================================================
# Figure family 2: Ground Truth / Prediction / error metric
# =============================================================================

def _resolve_metric(error_metric: str, fname: str, auto_absolute) -> str:
    if error_metric != "auto":
        return error_metric
    return "absolute" if fname in set(auto_absolute or ()) else "smape"


def _level_series(df: pd.DataFrame, level_col, label_col) -> pd.Series | None:
    if label_col is not None and label_col in df.columns:
        return df[label_col].astype(str)
    if level_col is not None and level_col in df.columns:
        lvl = pd.to_numeric(df[level_col], errors="coerce")
        labels = []
        for x in lvl.to_numpy(dtype=float, na_value=np.nan):
            if not np.isfinite(x):
                labels.append(np.nan)
                continue
            i = int(x)
            labels.append(
                f"{STANDARD_PRESSURE_LEVELS[i]}hPa"
                if 0 <= i < len(STANDARD_PRESSURE_LEVELS) else f"level_{i}"
            )
        return pd.Series(labels, index=df.index)
    if "pressure_hPa" in df.columns:
        p = pd.to_numeric(df["pressure_hPa"], errors="coerce")
        std = np.asarray(STANDARD_PRESSURE_LEVELS, dtype=float)
        out = []
        for x in p.to_numpy(dtype=float, na_value=np.nan):
            if not np.isfinite(x):
                out.append(np.nan)
                continue
            out.append(f"{int(std[int(np.argmin(np.abs(std - float(x))))])}hPa")
        return pd.Series(out, index=df.index)
    return None


def sorted_level_labels(series: pd.Series) -> list[str]:
    vals = [str(x) for x in series.dropna().unique().tolist()
            if str(x) and str(x) != "nan"]

    def parse_hpa(label: str):
        m = re.search(r"(\d{2,4})", str(label).lower().replace(" ", ""))
        try:
            return float(m.group(1)) if m else None
        except Exception:
            return None

    parsed = [(v, parse_hpa(v)) for v in vals]
    if any(p is not None for _, p in parsed):
        parsed.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0), x[0]))
        return [v for v, _ in parsed]
    return sorted(vals)


def plot_error_map(ctx: PlotCtx) -> None:
    """Truth / prediction / error maps, optionally stratified by pressure level.

    Replaces plot_instrument_maps for both the single-lead and horizon cases.
    """
    require_plotting()
    ps = int(ctx.opt("render.point_size"))
    cmap_e = ctx.opt("render.cmap_error")
    strict = bool(ctx.opt("windows.strict_obs_window"))
    min_pts = int(ctx.opt("render.min_points_per_level"))
    drop_small = bool(getattr(ctx.spec, "drop_small_truth", True))

    is_profile_inst = ctx.instrument in ("radiosonde", "aircraft")
    do_levels = bool(ctx.opt("figures.pressure_levels")) and is_profile_inst

    feats = select_features(
        discover_features(ctx.df, ctx.spec.n_channels), ctx.spec.channels
    )

    for sel, kind, step in resolve_selections(ctx):
        sub = select_leads(ctx.df, sel)
        if len(sub) == 0:
            continue
        fn_tag, tt_tag = window_tags(ctx, sel, kind, step, len(sub) != len(ctx.df))

        if kind == "horizon":
            s, e = horizon_bounds([int(x) for x in sel], step)
            check_obs_window(sub, s, e, strict)

        level_col = first_existing(sub, PRESSURE_LEVEL_CANDIDATES) if do_levels else None
        label_col = first_existing(sub, PRESSURE_LABEL_CANDIDATES) if do_levels else None
        lvl_labels = None
        if do_levels:
            lvl_labels = _level_series(sub, level_col, label_col)
            if lvl_labels is None:
                print("[WARN] Per-level maps requested but no pressure level column found.")

        for fname in feats:
            if f"true_{fname}" not in sub.columns:
                continue

            base_valid = apply_qc(sub, ctx.instrument, fname, need_truth=True, cfg=ctx.cfg)
            t_all = np_(sub[f"true_{fname}"])
            p_all = np_(sub[f"pred_{fname}"])
            lon_all = np_(sub["lon"])
            lat_all = np_(sub["lat"])

            metric = _resolve_metric(ctx.spec.error_metric, fname,
                                     ctx.opt("features.auto_absolute"))
            tiny = float((ctx.opt("features.tiny_threshold") or {}).get(fname, 0.0))

            save_level_skill(ctx, sub, fname, fn_tag, level_col, label_col)

            def render(tag_dir: str, tag_suffix: str, mask: np.ndarray, min_points: int):
                m = mask.copy()
                if drop_small and metric in ("percent", "smape"):
                    m &= np.abs(t_all) >= tiny
                if not np.any(m) or int(m.sum()) < int(min_points):
                    return

                t, p = t_all[m], p_all[m]
                lon, lat = lon_all[m], lat_all[m]
                print_sanity(fname, t, p, tiny if drop_small else None)
                _record_metric(ctx, fname, kind, sel, t, p,
                               level=(tag_suffix.lstrip("_") or None))

                vmin, vmax = value_limits(ctx, fname, t, p)
                fig, axes = make_axes_triple(
                    f"Instrument: {ctx.instrument} - {fname}{tt_tag}{tag_suffix}")

                sc1 = axes[0].scatter(lon, lat, c=t, cmap=cmap_e, s=ps, vmin=vmin,
                                      vmax=vmax, transform=ccrs.PlateCarree())
                fig.colorbar(sc1, ax=axes[0], orientation="horizontal",
                             pad=0.1).set_label("Value")
                axes[0].set_title("Ground Truth")

                sc2 = axes[1].scatter(lon, lat, c=p, cmap=cmap_e, s=ps, vmin=vmin,
                                      vmax=vmax, transform=ccrs.PlateCarree())
                fig.colorbar(sc2, ax=axes[1], orientation="horizontal",
                             pad=0.1).set_label("Value")
                axes[1].set_title("Prediction")

                if metric == "absolute":
                    err = np.abs(p - t)
                    lo, hi = np.nanpercentile(err, [1, 99])
                    err = np.clip(err, lo, hi)
                    label, cmap, norm = "Abs Error", cmap_e, None
                elif metric == "percent":
                    err = np.clip(100.0 * (p - t) / np.clip(np.abs(t), 1e-6, None),
                                  -200, 200)
                    mmax = float(np.nanmax(np.abs(err))) if np.isfinite(err).any() else 1.0
                    label, cmap = "% Error", "bwr"
                    norm = TwoSlopeNorm(vmin=-mmax, vcenter=0.0, vmax=mmax)
                else:
                    err = smape(p, t)
                    lo, hi = np.nanpercentile(err, [1, 99])
                    err = np.clip(err, lo, hi)
                    label, cmap, norm = "sMAPE (%)", cmap_e, None

                sc3 = axes[2].scatter(lon, lat, c=err, cmap=cmap, norm=norm, s=ps,
                                      transform=ccrs.PlateCarree())
                fig.colorbar(sc3, ax=axes[2], orientation="horizontal",
                             pad=0.1).set_label(label)
                axes[2].set_title(label)

                _finish_geo(axes)
                plt.tight_layout()
                safe = str(fname).replace(" ", "_")
                _save(tag_dir,
                      f"{ctx.instrument}_map_{safe}{fn_tag}{tag_suffix}_{metric}.png",
                      ctx)

            if is_profile_inst:
                if lvl_labels is None:
                    print(f"  Info: skipping {ctx.instrument} mixed-level map for "
                          f"'{fname}' (per-level maps disabled).")
            else:
                render(ctx.fig_dir, "", base_valid, 1)

            if do_levels and lvl_labels is not None:
                for lvl in sorted_level_labels(lvl_labels):
                    m = base_valid & (lvl_labels.astype(str) == str(lvl)).to_numpy()
                    render(os.path.join(ctx.fig_dir, f"level_{lvl}"),
                           f"_{lvl}", m, min_pts)

        if ctx.opt("figures.wind"):
            # Per-level panels get the threshold; whole-domain panels do not,
            # mirroring the scalar render() above.
            _plot_wind(ctx, sub, fn_tag, tt_tag,
                       lvl_labels if do_levels else None,
                       min_points=min_pts if do_levels else 1)


def _plot_wind(ctx: PlotCtx, sub: pd.DataFrame, fn_tag: str, tt_tag: str,
               lvl_labels: pd.Series | None, min_points: int = 1) -> None:
    cols = wind_columns(sub)
    if cols is None or not {"lon", "lat"}.issubset(sub.columns):
        return
    u_name, v_name = cols

    tu, tv = np_(sub[f"true_{u_name}"]), np_(sub[f"true_{v_name}"])
    pu, pv = np_(sub[f"pred_{u_name}"]), np_(sub[f"pred_{v_name}"])
    lon_all, lat_all = np_(sub["lon"]), np_(sub["lat"])
    ps = int(ctx.opt("render.point_size"))
    cmap_e = ctx.opt("render.cmap_error")
    calm = float(ctx.opt("features.calm_wind_threshold"))

    ok = (np.isfinite(tu) & np.isfinite(tv) & np.isfinite(pu) & np.isfinite(pv)
          & np.isfinite(lon_all) & np.isfinite(lat_all))

    def render(tag_dir: str, tag_suffix: str, mask: np.ndarray) -> None:
        m = ok & mask
        # Without this the wind panels ignored min_points_per_level and would
        # draw a global map from a handful of obs, while the scalar panels at
        # the same lead and level were correctly suppressed.
        if not np.any(m) or int(m.sum()) < int(min_points):
            return
        tu_i, tv_i, pu_i, pv_i = tu[m], tv[m], pu[m], pv[m]
        lon, lat = lon_all[m], lat_all[m]

        ts, pspd = np.hypot(tu_i, tv_i), np.hypot(pu_i, pv_i)
        tdir = (np.degrees(np.arctan2(-tu_i, -tv_i)) + 360.0) % 360.0
        pdir = (np.degrees(np.arctan2(-pu_i, -pv_i)) + 360.0) % 360.0
        ang = shortest_arc_deg(pdir, tdir)
        se = np.abs(pspd - ts)

        vmin, vmax = value_limits(ctx, "wind_speed", ts, pspd)
        fig, axes = make_axes_triple(
            f"Instrument: {ctx.instrument} - wind_speed{tt_tag}{tag_suffix}")
        sc1 = axes[0].scatter(lon, lat, c=ts, cmap=cmap_e, s=ps, vmin=vmin,
                              vmax=vmax, transform=ccrs.PlateCarree())
        fig.colorbar(sc1, ax=axes[0], orientation="horizontal", pad=0.1).set_label("Value")
        axes[0].set_title("Ground Truth")
        sc2 = axes[1].scatter(lon, lat, c=pspd, cmap=cmap_e, s=ps, vmin=vmin,
                              vmax=vmax, transform=ccrs.PlateCarree())
        fig.colorbar(sc2, ax=axes[1], orientation="horizontal", pad=0.1).set_label("Value")
        axes[1].set_title("Prediction")
        lo, hi = np.nanpercentile(se, [1, 99])
        sc3 = axes[2].scatter(lon, lat, c=np.clip(se, lo, hi), cmap=cmap_e, s=ps,
                              transform=ccrs.PlateCarree())
        fig.colorbar(sc3, ax=axes[2], orientation="horizontal",
                     pad=0.1).set_label("Abs Error (m/s)")
        axes[2].set_title("Abs Error (m/s)")
        _finish_geo(axes)
        plt.tight_layout()
        _save(tag_dir, f"{ctx.instrument}_map_wind_speed{fn_tag}{tag_suffix}.png", ctx)

        keep = ts >= calm
        if int(keep.sum()) < int(min_points):
            return
        lon_d, lat_d = lon[keep], lat[keep]
        td, pd_, ad = tdir[keep], pdir[keep], ang[keep]
        vmin = float(np.nanmin([td.min(), pd_.min()]))
        vmax = float(np.nanmax([td.max(), pd_.max()]))

        fig, axes = make_axes_triple(
            f"Instrument: {ctx.instrument} - wind_direction{tt_tag}{tag_suffix}")
        sc1 = axes[0].scatter(lon_d, lat_d, c=td, cmap=cmap_e, s=ps, vmin=vmin,
                              vmax=vmax, transform=ccrs.PlateCarree())
        fig.colorbar(sc1, ax=axes[0], orientation="horizontal", pad=0.1).set_label("Value")
        axes[0].set_title("Ground Truth")
        sc2 = axes[1].scatter(lon_d, lat_d, c=pd_, cmap=cmap_e, s=ps, vmin=vmin,
                              vmax=vmax, transform=ccrs.PlateCarree())
        fig.colorbar(sc2, ax=axes[1], orientation="horizontal", pad=0.1).set_label("Value")
        axes[1].set_title("Prediction")
        lo, hi = np.nanpercentile(ad, [1, 99])
        sc3 = axes[2].scatter(lon_d, lat_d, c=np.clip(ad, lo, hi), cmap=cmap_e, s=ps,
                              transform=ccrs.PlateCarree())
        fig.colorbar(sc3, ax=axes[2], orientation="horizontal",
                     pad=0.1).set_label("Abs Error (deg)")
        axes[2].set_title("Abs Error (deg)")
        _finish_geo(axes)
        plt.tight_layout()
        _save(tag_dir, f"{ctx.instrument}_map_wind_direction{fn_tag}{tag_suffix}.png", ctx)

    if ctx.instrument in ("radiosonde", "aircraft"):
        if lvl_labels is not None:
            for lvl in sorted_level_labels(lvl_labels):
                m = (lvl_labels.astype(str) == str(lvl)).to_numpy()
                render(os.path.join(ctx.fig_dir, f"level_{lvl}"), f"_{lvl}", m)
    else:
        render(ctx.fig_dir, "", np.ones(len(sub), dtype=bool))


# =============================================================================
# Figure family 3: prediction-only maps (no ground truth)
# =============================================================================

def plot_pred_only(ctx: PlotCtx) -> None:
    """Single-panel prediction maps. Replaces plot_mesh_maps."""
    require_plotting()
    ps = int(ctx.opt("render.point_size"))
    cmap_v = ctx.opt("render.cmap_value")
    units = ctx.spec.units

    feats = select_features(
        discover_features(ctx.df, ctx.spec.n_channels), ctx.spec.channels
    )

    for sel, kind, step in resolve_selections(ctx):
        sub = select_leads(ctx.df, sel)
        if len(sub) == 0:
            continue
        fn_tag, tt_tag = window_tags(ctx, sel, kind, step, len(sub) != len(ctx.df))

        for fname in feats:
            valid = apply_qc(sub, ctx.instrument, fname, need_truth=False, cfg=ctx.cfg)
            if not np.any(valid):
                print(f"  Info: no valid rows for '{fname}'. Skipping.")
                continue

            p = np_(sub[f"pred_{fname}"])[valid]
            lon = np_(sub["lon"])[valid]
            lat = np_(sub["lat"])[valid]
            print(f"  {fname}: {valid.sum()}/{len(sub)} obs retained")

            vmin, vmax = value_limits(ctx, fname, p)

            fig, ax = plt.subplots(
                1, 1, figsize=(8, 6),
                subplot_kw={"projection": ccrs.PlateCarree()},
            )
            ax.set_title(f"{ctx.instrument} - {fname}{tt_tag}", fontsize=14)
            sc = ax.scatter(lon, lat, c=p, s=ps, cmap=cmap_v, vmin=vmin, vmax=vmax,
                            transform=ccrs.PlateCarree())
            fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.1).set_label(
                f"Prediction{f' ({units})' if units else ''}")
            ax.set_global()
            _add_land(ax)
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.tight_layout()
            safe = str(fname).replace(" ", "_")
            _save(ctx.fig_dir, f"{ctx.instrument}_prediction_{safe}{fn_tag}.png", ctx)


# =============================================================================
# Pressure-level metric tables
# =============================================================================

def metrics_by_pressure_level(df, feat, level_col="pressure_level_idx",
                              label_col="pressure_level_label", agg="mean"):
    """Metrics stratified by categorical pressure level index (0-15)."""
    true_col, pred_col, mask_col = f"true_{feat}", f"pred_{feat}", f"mask_{feat}"
    if level_col not in df.columns or true_col not in df.columns or pred_col not in df.columns:
        return None

    level_idx = np_(df[level_col])
    t, y = np_(df[true_col]), np_(df[pred_col])

    valid = np.isfinite(level_idx) & np.isfinite(t) & np.isfinite(y) & (level_idx >= 0)
    if mask_col in df.columns:
        valid &= df[mask_col].fillna(False).astype(bool).to_numpy()

    level_idx = level_idx[valid].astype(int)
    t, y = t[valid], y[valid]
    if level_idx.size == 0:
        return None

    labels_series = df.loc[df.index[valid], label_col] if (
        label_col and label_col in df.columns) else None

    rows = []
    for lvl in range(len(STANDARD_PRESSURE_LEVELS)):
        mask = level_idx == lvl
        if not np.any(mask):
            continue
        tt, yy = t[mask], y[mask]
        diff = yy - tt
        t_agg = float(np.nanmedian(tt)) if agg == "median" else float(np.nanmean(tt))
        y_agg = float(np.nanmedian(yy)) if agg == "median" else float(np.nanmean(yy))

        if labels_series is not None:
            level_label = labels_series.iloc[int(np.where(mask)[0][0])]
        else:
            level_label = f"{STANDARD_PRESSURE_LEVELS[lvl]}hPa"

        true_var, pred_var = float(np.nanvar(tt)), float(np.nanvar(yy))
        rows.append({
            "pressure_level_idx": int(lvl),
            "pressure_level_label": level_label,
            "pressure_hPa": STANDARD_PRESSURE_LEVELS[lvl],
            "N": int(mask.sum()),
            "mean_true": t_agg,
            "mean_pred": y_agg,
            "std_true": float(np.nanstd(tt)),
            "std_pred": float(np.nanstd(yy)),
            "variance_ratio": pred_var / true_var if true_var > 0 else np.nan,
            "bias": float(np.nanmean(diff)),
            "RMSE": float(np.sqrt(np.nanmean(diff ** 2))),
            "MAE": float(np.nanmean(np.abs(diff))),
            "R2": float(np.corrcoef(tt, yy)[0, 1] ** 2) if tt.size > 1 else np.nan,
        })

    return pd.DataFrame(rows) if rows else None


def metrics_by_pressure_bins(df, feat, pcol="pressure_hPa",
                             bins_hpa=(1000, 850, 700, 500, 300, 200, 100, 50, 10),
                             agg="mean"):
    """Fallback metrics from binned continuous pressure."""
    true_col, pred_col, mask_col = f"true_{feat}", f"pred_{feat}", f"mask_{feat}"
    if pcol not in df.columns or true_col not in df.columns or pred_col not in df.columns:
        return None

    p, t, y = np_(df[pcol]), np_(df[true_col]), np_(df[pred_col])
    valid = np.isfinite(p) & np.isfinite(t) & np.isfinite(y)
    if mask_col in df.columns:
        valid &= df[mask_col].fillna(False).astype(bool).to_numpy()
    p, t, y = p[valid], t[valid], y[valid]
    if p.size == 0:
        return None

    rows = []
    edges = list(bins_hpa)
    for hi, lo in zip(edges[:-1], edges[1:]):
        layer = (p <= hi) & (p > lo)
        if not np.any(layer):
            continue
        tt, yy = t[layer], y[layer]
        diff = yy - tt
        rows.append({
            "p_hi_hPa": float(hi), "p_lo_hPa": float(lo),
            "p_mid_hPa": float(0.5 * (hi + lo)),
            "N": int(layer.sum()),
            "mean_true": float(np.nanmedian(tt)) if agg == "median" else float(np.nanmean(tt)),
            "mean_pred": float(np.nanmedian(yy)) if agg == "median" else float(np.nanmean(yy)),
            "bias": float(np.nanmean(diff)),
            "RMSE": float(np.sqrt(np.nanmean(diff ** 2))),
            "MAE": float(np.nanmean(np.abs(diff))),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Figure family 4: vertical profiles
# =============================================================================

def wrap_title(title: str, width: int = 52, max_lines: int = 3) -> str:
    """Wrap a title to a few short lines.

    Profile panels are only 7 inches wide, and figures are saved with
    bbox_inches="tight", so a long single-line title stretches the whole canvas
    to fit it. Explicit newlines in the input are preserved as breaks.
    """
    import textwrap

    out: list[str] = []
    for para in str(title).split("\n"):
        out.extend(textwrap.wrap(para, width=width) or [""])
    if len(out) > max_lines:
        out = out[:max_lines - 1] + [" ".join(out[max_lines - 1:])]
    return "\n".join(out)


def _profile_panel(x, p_hpa, labels, *, xlabel, title, out_name, ctx,
                   color=None, marker="o", second=None, vline=None, xlim=None,
                   legend_label=None):
    plt.figure(figsize=(7, 9))
    if second is not None:
        plt.plot(x, p_hpa, marker="o", markersize=8, linewidth=2, label=second[0])
        plt.plot(second[1], p_hpa, marker="s", markersize=8, linewidth=2, label=second[2])
    else:
        plt.plot(x, p_hpa, marker=marker, markersize=8, linewidth=2,
                 color=color, label=(legend_label or xlabel))
    if vline is not None:
        plt.axvline(x=vline, color="gray", linestyle="--", linewidth=1.5,
                    label="Perfect (100%)")
    plt.gca().invert_yaxis()
    plt.yscale("log")
    plt.yticks(p_hpa, labels)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel("Pressure Level", fontsize=12)
    plt.title(wrap_title(title), fontsize=12)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
    if xlim is not None:
        plt.xlim(*xlim)
    plt.legend(fontsize=11)
    _save(ctx.fig_dir, out_name, ctx)


def plot_profiles(ctx: PlotCtx) -> None:
    """Vertical profiles by categorical pressure level.

    Replaces plot_radiosonde_profiles_by_pressure_level and its _horizon twin.
    """
    require_plotting()
    min_samples = int(getattr(ctx.spec, "profile_min_samples", None)
                      or ctx.opt("render.profile_min_samples"))
    agg = "mean"
    strict = bool(ctx.opt("windows.strict_obs_window"))

    level_col = first_existing(ctx.df, PRESSURE_LEVEL_CANDIDATES)
    label_col = first_existing(ctx.df, PRESSURE_LABEL_CANDIDATES)
    if level_col is None:
        print("[WARN] No pressure_level_idx column; cannot build profiles.")
        return

    feats = select_features(discover_features(ctx.df), ctx.spec.channels)

    for sel, kind, step in resolve_selections(ctx):
        sub = select_leads(ctx.df, sel)
        if len(sub) == 0:
            continue
        fn_tag, tt_tag = window_tags(ctx, sel, kind, step, len(sub) != len(ctx.df))

        if kind == "horizon":
            s, e = horizon_bounds([int(x) for x in sel], step)
            check_obs_window(sub, s, e, strict)

        for feat in feats:
            level_df = metrics_by_pressure_level(sub, feat, level_col, label_col, agg)
            if level_df is None or level_df.empty:
                continue

            # NOTE: the original wrote this file with a hard-coded "radiosonde_"
            # prefix even for aircraft, so aircraft tables overwrote radiosonde
            # ones. Now keyed by the actual instrument, and written once.
            save_level_skill(ctx, sub, feat, fn_tag, level_col, label_col)

            p_hpa = level_df["pressure_hPa"].to_numpy()
            counts = level_df["N"].to_numpy()
            keep = np.isfinite(p_hpa) & (p_hpa > 0) & (counts >= min_samples)

            if not np.any(keep):
                print(f"  [WARN] No levels with >= {min_samples} samples for {feat}.")
                continue

            excluded = level_df[np.isfinite(p_hpa) & (p_hpa > 0) & (counts < min_samples)]
            if len(excluded):
                info = ", ".join(f"{r['pressure_level_label']} (N={r['N']})"
                                 for _, r in excluded.iterrows())
                print(f"  [INFO] Excluding {len(excluded)} sparse level(s): {info}")

            p_hpa = p_hpa[keep]
            lf = level_df[keep].reset_index(drop=True)
            if len(p_hpa) < 2:
                print(f"  [WARN] Fewer than 2 levels remain for {feat}.")
                continue
            if p_hpa.max() / p_hpa.min() < 1.5:
                print(f"  [WARN] Pressure range too narrow for {feat} "
                      f"({p_hpa.min():.0f}-{p_hpa.max():.0f} hPa).")
                continue

            labels = lf["pressure_level_label"].to_numpy()
            head = f"{ctx.instrument} - {feat}"

            try:
                _profile_panel(
                    lf["mean_true"].to_numpy(), p_hpa, labels,
                    xlabel=f"{feat} ({agg})",
                    title=f"{head} - True vs Pred{tt_tag}\n(by pressure level)",
                    out_name=f"{ctx.instrument}_{feat}_true_vs_pred_by_level{fn_tag}.png",
                    ctx=ctx,
                    second=("True (level avg)", lf["mean_pred"].to_numpy(),
                            "Pred (level avg)"),
                )
            except Exception as e:
                plt.close()
                print(f"  [ERROR] True-vs-Pred profile failed for {feat}: {e}")

            try:
                _profile_panel(
                    lf["RMSE"].to_numpy(), p_hpa, labels, xlabel="RMSE",
                    title=f"{head} - RMSE{tt_tag}\n(by pressure level)",
                    out_name=f"{ctx.instrument}_{feat}_rmse_by_level{fn_tag}.png",
                    ctx=ctx, color="red",
                )
            except Exception as e:
                plt.close()
                print(f"  [ERROR] RMSE profile failed for {feat}: {e}")

            try:
                vr = lf["variance_ratio"].to_numpy()
                finite = vr[np.isfinite(vr)]
                xlim = (0, max(120, float(np.max(finite)) * 105)) if finite.size else (0, 120)
                # NOTE: the original named this file with the epoch only, so
                # every lead overwrote the previous one. Now uses the full tag.
                _profile_panel(
                    vr * 100, p_hpa, labels,
                    xlabel="Prediction Variance / True Variance (%)",
                    legend_label="Variance Ratio",
                    title=f"{head} - Variance Ratio{tt_tag}\n(by pressure level)",
                    out_name=f"{ctx.instrument}_{feat}_variance_ratio_by_level{fn_tag}.png",
                    ctx=ctx, color="green", vline=100, xlim=xlim,
                )
            except Exception as e:
                plt.close()
                print(f"  [ERROR] Variance-ratio profile failed for {feat}: {e}")


# =============================================================================
# Figure family 5 (new): AR error growth
# =============================================================================

def plot_ar_growth(metrics: pd.DataFrame, fig_dir: str, cfg: dict) -> None:
    """RMSE and bias against lead hour, one line per AR step.

    Built from the metrics rows accumulated during plotting, so it costs no
    extra file reads. This is the diagnostic the old code had no equivalent of.
    """
    require_plotting()
    if metrics is None or metrics.empty:
        print("[ar_growth] No metrics collected; nothing to plot.")
        return

    dpi = int((cfg.get("render") or {}).get("dpi", 150))
    df = metrics[metrics["window_kind"] == "sub"].copy()
    if df.empty:
        df = metrics.copy()
    df = df[np.isfinite(pd.to_numeric(df["lead_hours"], errors="coerce"))]
    if df.empty:
        print("[ar_growth] No lead-resolved metrics; nothing to plot.")
        return

    # Profile instruments contribute one row per pressure level. Pool them into
    # a column total rather than drawing a multi-valued line: RMSE pools as
    # sqrt(sum(n*rmse^2)/sum(n)) and bias as sum(n*bias)/sum(n), both exact.
    if "level" in df.columns and df["level"].notna().any():
        keys = ["instrument", "feature", "ar_step", "lead_hours"]
        df["_sq"] = df["n"] * df["rmse"] ** 2
        df["_b"] = df["n"] * df["bias"]
        df = (df.groupby(keys, dropna=False, as_index=False)
                .agg(n=("n", "sum"), _sq=("_sq", "sum"), _b=("_b", "sum")))
        denom = df["n"].replace(0, np.nan)
        df["rmse"] = np.sqrt(df["_sq"] / denom)
        df["bias"] = df["_b"] / denom
        df = df.drop(columns=["_sq", "_b"])

    pooled_levels = bool("level" in metrics.columns and metrics["level"].notna().any())
    os.makedirs(fig_dir, exist_ok=True)

    for (inst, feat), g in df.groupby(["instrument", "feature"]):
        if g["lead_hours"].nunique() < 2:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
        for ar, gg in g.groupby("ar_step", dropna=False):
            gg = gg.sort_values("lead_hours")
            lbl = "single" if pd.isna(ar) else f"AR{int(ar)}"
            axes[0].plot(gg["lead_hours"], gg["rmse"], marker="o", label=lbl)
            axes[1].plot(gg["lead_hours"], gg["bias"], marker="o", label=lbl)

        axes[0].set_ylabel("RMSE")
        axes[1].set_ylabel("Bias (pred - true)")
        axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1)
        for ax in axes:
            ax.set_xlabel("Lead time (hours after init)")
            ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
            ax.legend(fontsize=9)
        fig.suptitle(f"{inst} - {feat} - error growth across AR rollout", fontsize=15)
        if pooled_levels:
            axes[0].text(0.02, 0.02, "pooled across pressure levels",
                         transform=axes[0].transAxes, fontsize=9, color="gray")
        plt.tight_layout()
        safe = str(feat).replace(" ", "_")
        out = os.path.join(fig_dir, f"{inst}_{safe}_ar_growth.png")
        plt.savefig(out, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved: {out}")


# Registry of figure families, consumed by evaluations.py
FIGURE_FNS = {
    "diff": plot_diff,
    "error_map": plot_error_map,
    "profiles": plot_profiles,
    "pred_only": plot_pred_only,
}
