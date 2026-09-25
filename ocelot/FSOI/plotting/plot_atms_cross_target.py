#!/usr/bin/env python
"""
Manuscript Figure 3: ATMS channel impact by verification target.

Produces a single composite PNG (and PDF) with three vertically stacked panels
sharing the ATMS channel x-axis:

    (a) Radiosonde target — 4 variables x 16 pressure levels
    (b) Aircraft target   — 3 variables x flight-level pressures
    (c) Surface obs target — one row per variable (T, Td, u, v, ps) when the
        per-variable FSOI run is available; legacy aggregate-only runs collapse
        to a single strip.

Panel heights are proportional to the number of target rows in each panel so
the surface strip is naturally small.  Each panel has its own symmetric
colorbar because the three targets differ in magnitude by ~6x, so a shared
scale would wash out the surface panel and hide the sign/pattern that is the
whole point of the figure.

Data source: the same seasonal FSOI CSVs used by
``analyze_cross_target_channels.py``.  The three target dictionaries in that
module are re-used so the figure stays in sync with the underlying analysis.

Usage:
    python FSOI/plotting/plot_atms_cross_target.py
    python FSOI/plotting/plot_atms_cross_target.py --output FSOI/fsoi_outputs/paper_figures
    python FSOI/plotting/plot_atms_cross_target.py --instrument ssmis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_FSOI = Path(__file__).resolve().parent.parent
if str(_FSOI) not in sys.path:
    sys.path.insert(0, str(_FSOI))

from analyze_cross_target_channels import (  # noqa: E402
    TARGETS,
    TARGET_DISPLAY,
    load_and_average,
)
from plot_instrument_channel_heatmaps import (  # noqa: E402
    DISPLAY_NAMES,
    _prepare_aggregate,
    _ordered_rows,
    _channel_label,
)


# Manuscript-friendly typography.  Kept as module constants so the whole file
# stays consistent and one edit changes every panel.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    "axes.linewidth": 0.6,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,   # editable text in Illustrator/InDesign
    "ps.fonttype": 42,
})

MIN_FONT = 10.5
FONT_TITLE = 16
FONT_SUPTITLE = 16
FONT_PANEL_TAG = 15   # (a), (b), (c) labels
FONT_AXIS_LABEL = 15
FONT_TICK = 12.5
FONT_GROUP = 18   # variable group symbol on the y-axis
FONT_CBAR_LABEL = 12
FONT_CBAR_TICK = 11.5

HEATMAP_CMAP = "RdBu_r"
GRID_COLOR = "white"
SEP_COLOR = "#2b2b2b"
SPINE_COLOR = "#5a5a5a"

# Label for the legacy single-strip case: older surface runs lacked per-variable
# FSOI stratification, so the panel shows one aggregate row instead of one per
# variable.  Per-variable runs use the real variable names instead.
SURFACE_AGGREGATE_LABEL = "Aggregate surface-target FSOI"

# Pretty display names for target variables.
VAR_PRETTY = {
    "temperature":          "Temperature",
    "dewpoint_temperature": "Dewpoint",
    "specific_humidity":    "Humidity",
    "u_wind":               "Zonal wind",
    "v_wind":               "Meridional wind",
    "surface_pressure":     "MSLP",
}
VAR_SHORT = {
    "temperature":          "$T$",
    "dewpoint_temperature": "$T_d$",
    "specific_humidity":    "$q$",
    "u_wind":               "$u$",
    "v_wind":               "$v$",
    "surface_pressure":     "$p_{msl}$",
}


def _pretty_var(name: str, short: bool = False) -> str:
    key = str(name)
    if short:
        return VAR_SHORT.get(key, key)
    return VAR_PRETTY.get(key, key.replace("_", " ").capitalize())


def _ensure_target_variable_column(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """Mirror of analyze_cross_target_channels._ensure_target_variable_column.

    Adds a synthetic ``target_variable`` / ``pressure_hpa`` for surface_obs so
    ``_prepare_aggregate`` can build a pivot with one row.
    """
    df = df.copy()
    if "target_variable" not in df.columns:
        df["target_variable"] = f"{target_name}_agg"
    if "pressure_hpa" not in df.columns:
        df["pressure_hpa"] = 0.0
    else:
        df["pressure_hpa"] = df["pressure_hpa"].fillna(0.0)
    if "p_hpa" not in df.columns:
        df["p_hpa"] = df["pressure_hpa"]
    else:
        df["p_hpa"] = df["p_hpa"].fillna(0.0)
    return df


def _pivot_for_instrument(
    df: pd.DataFrame,
    target_name: str,
    instrument: str,
) -> tuple[pd.DataFrame, list[dict], list[str]]:
    """Return (pivot values, per-row metadata, x-tick labels) for one instrument."""
    df = _ensure_target_variable_column(df, target_name)

    impact_col = "sum_impact_scaled" if "sum_impact_scaled" in df.columns else "mean_impact"
    if impact_col not in df.columns and "mean_impact" in df.columns:
        df = df.copy()
        df[impact_col] = df["mean_impact"]

    prep = _prepare_aggregate(df, basis="total")
    # ``_prepare_aggregate`` returns 4 items; keep only what we need.
    agg, value_col = prep[0], prep[1]

    inst_lower = instrument.lower()
    matched = [i for i in agg["instrument"].unique() if str(i).lower() == inst_lower]
    if not matched:
        raise ValueError(f"Instrument {instrument!r} not found in {target_name} data")
    inst_df = agg[agg["instrument"] == matched[0]].copy()

    inst_df = _ordered_rows(inst_df)
    # Meteorological convention: pressure decreases upward, so within each variable
    # block 1000 hPa sits at the bottom of the panel and 10 hPa at the top. Rows are
    # drawn top to bottom, so the sort key is ascending pressure.
    inst_df["_p_sort"] = inst_df["pressure_hpa"].astype(float)
    inst_df = inst_df.sort_values(["_var_rank", "_p_sort", "channel_display"])
    channel_order = sorted(inst_df["channel_display"].dropna().unique())
    channel_labels = {ch: _channel_label(matched[0], ch) for ch in channel_order}

    pivot = inst_df.pivot_table(
        index=["_var_rank", "_p_sort", "target_variable", "pressure_hpa", "row_label"],
        columns="channel_display",
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(columns=channel_order, fill_value=0.0)

    # Per-row metadata drives the styled y-axis (variable groups + pressure).
    row_meta = [
        {"var": str(idx[2]), "p": float(idx[3]), "aggregate": False}
        for idx in pivot.index
    ]
    x_labels = [channel_labels[ch] for ch in pivot.columns]

    # Surface obs: when per-variable FSOI stratification is available the panel
    # shows one row per target variable (T, Td, u, v, ps), like the other
    # targets.  Legacy aggregate-only runs (no target_variable column) yield a
    # single row and collapse to one labelled strip for backward compatibility.
    if target_name == "surface_obs":
        if pivot.shape[0] <= 1:
            aggregate_row = pivot.iloc[0].values if pivot.shape[0] == 1 else pivot.sum(axis=0).values
            pivot = pd.DataFrame(aggregate_row.reshape(1, -1), columns=pivot.columns)
            row_meta = [{"var": SURFACE_AGGREGATE_LABEL, "p": None, "aggregate": True}]
        else:
            # Surface targets carry no meaningful pressure coordinate.
            for m in row_meta:
                m["p"] = None

    return pivot, row_meta, x_labels


def _format_pressure(p: float) -> str:
    """Compact hPa tick label (e.g. 1000, 850, 12.5)."""
    if p is None or not np.isfinite(p):
        return ""
    return f"{int(round(p))}" if float(p).is_integer() else f"{p:g}"


def _pressure_tick_labels(row_meta: list[dict]) -> list[str]:
    """Show readable pressure ticks while keeping all fonts >= 10.5 pt."""
    if len(row_meta) <= 36:
        return [_format_pressure(m["p"]) for m in row_meta]
    # Dense panels: label interior levels only, at least two rows apart, so no
    # label collides with its neighbour or with the next variable block's edge.
    major = {850, 500, 250, 100, 30}
    return [
        _format_pressure(m["p"]) if m.get("p") is not None and int(round(float(m["p"]))) in major else ""
        for m in row_meta
    ]


def _draw_variable_groups(ax, row_meta: list[dict]) -> None:
    """Draw variable-group separators and a bold symbol per contiguous block.

    The symbol sits just left of the pressure ticks using a blended transform
    (x in axes fraction, y in data coordinates), so it lines up with the group
    centre regardless of how many pressure levels it spans.
    """
    n = len(row_meta)
    trans = ax.get_yaxis_transform()  # x: axes fraction, y: data coords
    start = 0
    for i in range(1, n + 1):
        if i == n or row_meta[i]["var"] != row_meta[start]["var"]:
            mid = (start + i - 1) / 2.0
            sym = _pretty_var(row_meta[start]["var"], short=True)
            ax.text(
                -0.11, mid, sym,
                transform=trans, ha="center", va="center",
                fontsize=FONT_GROUP, fontweight="bold", color="#1a1a1a",
                clip_on=False,
            )
            if i < n:
                ax.axhline(i - 0.5, color=SEP_COLOR, linewidth=1.1)
            start = i


def _draw_panel(
    ax,
    values: np.ndarray,
    row_meta: list[dict],
    x_labels: list[str],
    title: str,
    y_axis_label: str,
    show_xtick_labels: bool,
    cbar_label: str,
) -> None:
    n_rows = values.shape[0]
    n_cols = values.shape[1]
    pressure_mode = any(m.get("p") is not None for m in row_meta)

    # Robust, symmetric diverging scale centred on zero.  A high percentile
    # keeps a single outlier cell from washing out the whole panel; the
    # colorbar uses "extend" arrows so clipped extremes stay visible.
    finite = np.abs(values[np.isfinite(values)])
    finite = finite[finite > 0]
    if finite.size:
        vmax = float(np.percentile(finite, 99))
    else:
        vmax = 1.0
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = float(np.nanmax(np.abs(values))) or 1.0
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)

    im = ax.imshow(
        values,
        cmap=HEATMAP_CMAP,
        aspect="auto",
        norm=norm,
        interpolation="nearest",
    )

    # Thin cell gridlines for a clean, modern heatmap look.
    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)
    ax.grid(which="minor", color=GRID_COLOR, linewidth=0.6)
    ax.tick_params(which="minor", length=0)

    # X ticks (channels).
    ax.set_xticks(np.arange(n_cols))
    if show_xtick_labels:
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=FONT_TICK)
        ax.set_xlabel("ATMS channel", fontsize=FONT_AXIS_LABEL, labelpad=4)
    else:
        ax.set_xticklabels([""] * n_cols)

    # Y ticks.
    ax.set_yticks(np.arange(n_rows))
    if pressure_mode:
        ax.set_yticklabels(_pressure_tick_labels(row_meta), fontsize=FONT_TICK)
        _draw_variable_groups(ax, row_meta)
    else:
        labels = [
            m["var"] if m.get("aggregate") else _pretty_var(m["var"])
            for m in row_meta
        ]
        ax.set_yticklabels(labels, fontsize=FONT_TICK)
    ax.set_ylabel(y_axis_label, fontsize=FONT_AXIS_LABEL)
    ax.yaxis.set_label_coords(-0.165 if pressure_mode else -0.19, 0.5)

    ax.set_title(title, fontsize=FONT_TITLE, pad=6, loc="left", fontweight="bold")
    ax.tick_params(axis="both", length=2)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_COLOR)
        spine.set_linewidth(0.6)

    # Per-panel colorbar (targets differ ~6x in magnitude, so a shared scale
    # would flatten the weaker panels).
    cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012, aspect=26, extend="both")
    cbar.set_label(cbar_label, fontsize=FONT_CBAR_LABEL)
    cbar.ax.tick_params(labelsize=FONT_CBAR_TICK, length=2)
    cbar.outline.set_linewidth(0.5)
    cbar.outline.set_edgecolor(SPINE_COLOR)
    fmt = ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((-2, 3))
    cbar.ax.yaxis.set_major_formatter(fmt)
    cbar.ax.yaxis.get_offset_text().set_fontsize(FONT_CBAR_TICK)


def build_figure(
    dfs: dict[str, pd.DataFrame],
    instrument: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("png", "pdf"),
    dpi: int = 600,
) -> list[Path]:
    inst_display = DISPLAY_NAMES.get(instrument.lower(), instrument.upper())

    panels = []  # (target_name, pivot, row_meta, x_labels)
    for target_name in ("radiosonde", "aircraft", "surface_obs"):
        df = dfs.get(target_name, pd.DataFrame())
        if df.empty:
            print(f"[WARN] No data for {target_name}; skipping panel")
            continue
        try:
            pivot, row_meta, x_labels = _pivot_for_instrument(df, target_name, instrument)
        except ValueError as err:
            print(f"[WARN] {err}")
            continue
        panels.append((target_name, pivot, row_meta, x_labels))

    if not panels:
        raise RuntimeError(f"No panels could be built for instrument {instrument}")

    # Height ratios reflect the number of target rows in each panel so the
    # surface strip stays subordinate.  Clamp so single-row panels remain
    # readable and very tall panels do not dominate the figure.
    def _height_weight(n_rows: int) -> float:
        return float(np.clip(n_rows, 4, 34))

    height_ratios = [_height_weight(len(p[2])) for p in panels]

    fig_height = sum(0.18 * hr for hr in height_ratios) + 0.78 * len(panels) + 1.2
    fig_width = max(11.8, 0.48 * len(panels[0][3]) + 4.5)

    fig, axes = plt.subplots(
        nrows=len(panels),
        ncols=1,
        sharex=False,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.16},
    )
    if len(panels) == 1:
        axes = [axes]

    panel_tags = ["(a)", "(b)", "(c)"]
    cbar_label = "FSOI  (blue = beneficial)"

    for i, (ax, (target_name, pivot, row_meta, x_labels)) in enumerate(zip(axes, panels)):
        if target_name == "surface_obs":
            is_aggregate = bool(row_meta) and row_meta[0].get("aggregate", False)
            suffix = "  (aggregate)" if is_aggregate else ""
            title = f"{panel_tags[i]}  {inst_display} impact on {TARGET_DISPLAY[target_name]} target{suffix}"
            y_axis_label = "Surface\ntarget variable"
        else:
            title = f"{panel_tags[i]}  {inst_display} impact on {TARGET_DISPLAY[target_name]} target"
            y_axis_label = "Pressure (hPa)"
        _draw_panel(
            ax=ax,
            values=pivot.values,
            row_meta=row_meta,
            x_labels=x_labels,
            title=title,
            y_axis_label=y_axis_label,
            show_xtick_labels=(i == len(panels) - 1),
            cbar_label=cbar_label,
        )

    # Title block in fixed inches above the first panel title, so the spacing
    # does not depend on the figure height.
    inch = 1.0 / fig_height
    fig.suptitle(
        f"{inst_display} channel FSOI by verification target",
        fontsize=FONT_SUPTITLE,
        fontweight="bold",
        y=1.0 - 0.10 * inch, va="top",
    )
    fig.text(
        0.5, 1.0 - 0.45 * inch,
        "Seasonal mean forecast sensitivity to observation impact   "
        "(blue = beneficial,  red = detrimental)",
        ha="center", va="top", fontsize=FONT_CBAR_LABEL + 0.5, color="#444444",
    )
    # Manual margins avoid the tight_layout warning that inset colorbars trigger.
    fig.subplots_adjust(left=0.155, right=0.925, top=1.0 - 1.15 * inch, bottom=0.05)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    stem = f"figure3_{instrument.lower()}_cross_target"
    for fmt in formats:
        out = output_dir / f"{stem}.{fmt}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {out}")
        saved.append(out)
    plt.close(fig)
    return saved


def load_all_targets() -> dict[str, pd.DataFrame]:
    dfs = {}
    for target_name, cfg in TARGETS.items():
        print(f"\nLoading {target_name} channel data:")
        df = load_and_average(cfg["csv_dirs"], cfg["seasons"])
        if df.empty:
            print(f"  WARNING: no data found for {target_name}")
        dfs[target_name] = df
    return dfs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instrument",
        default="atms",
        help="Instrument to plot (default: atms)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_FSOI / "fsoi_outputs" / "paper_figures",
        help="Output directory (default: FSOI/fsoi_outputs/paper_figures)",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf"],
        help="Output formats (default: png pdf)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Raster output resolution for PNG and other bitmap formats.",
    )
    args = parser.parse_args()

    dfs = load_all_targets()
    build_figure(
        dfs=dfs,
        instrument=args.instrument,
        output_dir=args.output,
        formats=tuple(args.formats),
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
