#!/usr/bin/env python
"""
Manuscript Figure 1 — ATMS channel impact by verification target.

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
    python FSOI/plot_fig1_atms_cross_target.py
    python FSOI/plot_fig1_atms_cross_target.py --output FSOI/fsoi_outputs/paper_figures
    python FSOI/plot_fig1_atms_cross_target.py --instrument ssmis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_FSOI = Path(__file__).resolve().parent
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
FONT_TITLE       = 12
FONT_PANEL_TAG   = 12   # (a), (b), (c) labels
FONT_AXIS_LABEL  = 10
FONT_TICK        = 8
FONT_CBAR_LABEL  = 9
FONT_CBAR_TICK   = 8

# Label for the legacy single-strip case: older surface runs lacked per-variable
# FSOI stratification, so the panel shows one aggregate row instead of one per
# variable.  Per-variable runs use the real variable names instead.
SURFACE_AGGREGATE_LABEL = "Aggregate surface-target FSOI"


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
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (pivot values, row labels, x-tick labels) for one instrument."""
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
    channel_order = sorted(inst_df["channel_display"].dropna().unique())
    channel_labels = {ch: _channel_label(matched[0], ch) for ch in channel_order}

    pivot = inst_df.pivot_table(
        index=["_var_rank", "_p_sort", "target_variable", "pressure_hpa", "row_label"],
        columns="channel_display",
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(columns=channel_order, fill_value=0.0)

    row_labels = [idx[4] for idx in pivot.index]
    x_labels = [channel_labels[ch] for ch in pivot.columns]

    # Surface obs: when per-variable FSOI stratification is available the panel
    # shows one row per target variable (T, Td, u, v, ps), like the other
    # targets.  Legacy aggregate-only runs (no target_variable column) yield a
    # single row and collapse to one labelled strip for backward compatibility.
    if target_name == "surface_obs":
        if pivot.shape[0] <= 1:
            aggregate_row = pivot.iloc[0].values if pivot.shape[0] == 1 else pivot.sum(axis=0).values
            pivot = pd.DataFrame(aggregate_row.reshape(1, -1), columns=pivot.columns)
            row_labels = [SURFACE_AGGREGATE_LABEL]
        else:
            # Surface targets have no pressure; drop the meaningless "@ 0hPa" suffix.
            row_labels = [lbl.split(" @ ")[0] for lbl in row_labels]

    return pivot, row_labels, x_labels


def _draw_panel(
    ax,
    values: np.ndarray,
    row_labels: list[str],
    x_labels: list[str],
    title: str,
    y_axis_label: str,
    show_xtick_labels: bool,
    cbar_label: str,
) -> None:
    vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0

    im = ax.imshow(
        values,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    # X ticks (channels).
    ax.set_xticks(np.arange(len(x_labels)))
    if show_xtick_labels:
        ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=FONT_TICK)
        ax.set_xlabel("ATMS channels", fontsize=FONT_AXIS_LABEL)
    else:
        ax.set_xticklabels([""] * len(x_labels))

    # Y ticks (target variable @ pressure).
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=FONT_TICK)
    ax.set_ylabel(y_axis_label, fontsize=FONT_AXIS_LABEL)

    ax.set_title(title, fontsize=FONT_TITLE, pad=4)
    ax.tick_params(axis="both", length=2)

    # Separators between target variables.
    if len(row_labels) > 1:
        var_prefixes = [lab.split(" @ ")[0] for lab in row_labels]
        for i in range(1, len(var_prefixes)):
            if var_prefixes[i] != var_prefixes[i - 1]:
                ax.axhline(i - 0.5, color="black", linewidth=0.4, alpha=0.35)

    # Colorbar per panel with a fixed aspect so the surface strip does not get
    # a comically wide bar.
    cbar = plt.colorbar(im, ax=ax, fraction=0.018, pad=0.012, aspect=25)
    cbar.set_label(cbar_label, fontsize=FONT_CBAR_LABEL)
    cbar.ax.tick_params(labelsize=FONT_CBAR_TICK)


def build_figure(
    dfs: dict[str, pd.DataFrame],
    instrument: str,
    output_dir: Path,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[Path]:
    inst_display = DISPLAY_NAMES.get(instrument.lower(), instrument.upper())

    panels = []  # (target_name, pivot, row_labels, x_labels)
    for target_name in ("radiosonde", "aircraft", "surface_obs"):
        df = dfs.get(target_name, pd.DataFrame())
        if df.empty:
            print(f"[WARN] No data for {target_name}; skipping panel")
            continue
        try:
            pivot, row_labels, x_labels = _pivot_for_instrument(df, target_name, instrument)
        except ValueError as err:
            print(f"[WARN] {err}")
            continue
        panels.append((target_name, pivot, row_labels, x_labels))

    if not panels:
        raise RuntimeError(f"No panels could be built for instrument {instrument}")

    # Height ratios reflect the number of target rows in each panel so the
    # surface strip stays subordinate.  Clamp so single-row panels remain
    # readable and very tall panels do not dominate the figure.
    def _height_weight(n_rows: int) -> float:
        return float(np.clip(n_rows, 3, 32))

    height_ratios = [_height_weight(len(p[2])) for p in panels]

    fig_height = sum(0.16 * hr for hr in height_ratios) + 0.55 * len(panels) + 0.9
    fig_width = max(10.5, 0.42 * len(panels[0][3]) + 4.0)

    fig, axes = plt.subplots(
        nrows=len(panels),
        ncols=1,
        sharex=False,
        figsize=(fig_width, fig_height),
        gridspec_kw={"height_ratios": height_ratios, "hspace": 0.18},
    )
    if len(panels) == 1:
        axes = [axes]

    panel_tags = ["(a)", "(b)", "(c)"]
    cbar_label = "FSOI  (negative = beneficial)"

    for i, (ax, (target_name, pivot, row_labels, x_labels)) in enumerate(zip(axes, panels)):
        if target_name == "surface_obs":
            is_aggregate = (len(row_labels) == 1 and row_labels[0] == SURFACE_AGGREGATE_LABEL)
            suffix = " (aggregate)" if is_aggregate else ""
            title = f"{panel_tags[i]}  {inst_display} — {TARGET_DISPLAY[target_name]} target{suffix}"
            y_axis_label = "Surface obs target" if is_aggregate else "Surface obs target\n(variable)"
        else:
            title = f"{panel_tags[i]}  {inst_display} — {TARGET_DISPLAY[target_name]} target"
            y_axis_label = f"{TARGET_DISPLAY[target_name]} target\n(variable @ pressure)"
        _draw_panel(
            ax=ax,
            values=pivot.values,
            row_labels=row_labels,
            x_labels=x_labels,
            title=title,
            y_axis_label=y_axis_label,
            show_xtick_labels=(i == len(panels) - 1),
            cbar_label=cbar_label,
        )

    fig.suptitle(
        f"{inst_display} channel FSOI by verification target  "
        "(seasonal mean;  blue = beneficial,  red = detrimental)",
        fontsize=FONT_TITLE,
        y=0.995,
    )
    # Manual margins avoid the tight_layout warning that inset colorbars trigger.
    fig.subplots_adjust(left=0.14, right=0.92, top=0.965, bottom=0.055)

    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    stem = f"fig1_{instrument.lower()}_cross_target"
    for fmt in formats:
        out = output_dir / f"{stem}.{fmt}"
        fig.savefig(out, dpi=300, bbox_inches="tight")
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
    args = parser.parse_args()

    dfs = load_all_targets()
    build_figure(
        dfs=dfs,
        instrument=args.instrument,
        output_dir=args.output,
        formats=tuple(args.formats),
    )


if __name__ == "__main__":
    main()
