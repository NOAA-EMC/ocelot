#!/usr/bin/env python
"""Manuscript Figure 5, sized and formatted for JGR.

Same content as plot_atms_cross_target.py, whose data preparation and
panel drawing are imported rather than copied. Only the typography and the
figure geometry change: the module's font constants are lowered to the AGU
minimum and the figure is built at final print size, so nothing is shrunk by
the page layout.

The in-figure title block is dropped because the caption already carries it,
including that blue denotes beneficial attribution and that panels use
different scales. That also buys back the vertical room the smaller panels need.

AGU graphics requirements: two-column width 105-170 mm, height at most 228 mm,
Arial or Helvetica text no smaller than 8 pt at final print size, raster at
300-600 ppi. Saved as vector PDF plus 600 ppi TIFF and PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Imported first: this module sets its own rcParams at import time.
import plot_atms_cross_target as base
from plot_atms_cross_target import TARGET_DISPLAY, load_all_targets
from plot_instrument_channel_heatmaps import DISPLAY_NAMES

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The 6.5 in text column inside 1 in margins; still inside AGU's
# 105-170 mm range for a two-column figure.
WIDTH_IN = 165.1 / 25.4
MAX_HEIGHT_IN = 228.0 / 25.4
BASE_PT = 8.0

# The panel drawing reads these as module globals at call time, so lowering them
# here re-typesets the whole figure without touching the original script.
base.FONT_TITLE = BASE_PT + 1.0
base.FONT_AXIS_LABEL = BASE_PT
base.FONT_TICK = BASE_PT
base.FONT_GROUP = BASE_PT + 1.0
base.FONT_CBAR_LABEL = BASE_PT
base.FONT_CBAR_TICK = BASE_PT + 1.0   # its exponent renders at 0.7x, so keep it at 9

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": BASE_PT,
    "axes.linewidth": 0.5,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def build(dfs, instrument: str, out_dir: Path, dpi: int) -> list[Path]:
    inst_display = DISPLAY_NAMES.get(instrument.lower(), instrument.upper())

    panels = []
    for target_name in ("radiosonde", "aircraft", "surface_obs"):
        df = dfs.get(target_name)
        if df is None or df.empty:
            print(f"[WARN] No data for {target_name}; skipping panel")
            continue
        try:
            pivot, row_meta, x_labels = base._pivot_for_instrument(df, target_name, instrument)
        except ValueError as err:
            print(f"[WARN] {err}")
            continue
        panels.append((target_name, pivot, row_meta, x_labels))
    if not panels:
        raise RuntimeError(f"No panels could be built for instrument {instrument}")

    ratios = [float(np.clip(len(p[2]), 4, 34)) for p in panels]
    height = 0.078 * sum(ratios) + 0.34 * len(panels) + 0.80
    if height > MAX_HEIGHT_IN:
        raise RuntimeError(f"{height:.2f} in exceeds the {MAX_HEIGHT_IN:.2f} in page limit")

    fig, axes = plt.subplots(nrows=len(panels), ncols=1, figsize=(WIDTH_IN, height),
                             gridspec_kw={"height_ratios": ratios, "hspace": 0.13})
    if len(panels) == 1:
        axes = [axes]

    tags = ["(a)", "(b)", "(c)"]
    for i, (ax, (target_name, pivot, row_meta, x_labels)) in enumerate(zip(axes, panels)):
        if target_name == "surface_obs":
            aggregate = bool(row_meta) and row_meta[0].get("aggregate", False)
            title = (f"{tags[i]} {inst_display} impact on {TARGET_DISPLAY[target_name]} target"
                     + ("  (aggregate)" if aggregate else ""))
            # No axis label: the ticks already name the variables, the caption says
            # panel (c) resolves them, and a rotated two-line label collides with
            # "Meridional wind" at this width.
            y_label = ""
        else:
            title = f"{tags[i]} {inst_display} impact on {TARGET_DISPLAY[target_name]} target"
            y_label = "Pressure (hPa)"
        base._draw_panel(ax=ax, values=pivot.values, row_meta=row_meta, x_labels=x_labels,
                         title=title, y_axis_label=y_label,
                         show_xtick_labels=(i == len(panels) - 1),
                         cbar_label="FSOI (blue = beneficial)")
    fig.subplots_adjust(left=0.150, right=0.935, top=0.975, bottom=0.075)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"figure5_{instrument.lower()}_cross_target_jgr"
    written = []
    for path, kw in [(out_dir / f"{stem}.pdf", {}),
                     (out_dir / f"{stem}.png", {"dpi": dpi}),
                     (out_dir / f"{stem}.tif", {"dpi": dpi, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    print(f"figure is {WIDTH_IN * 25.4:.0f} x {height * 25.4:.0f} mm")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--instrument", default="atms")
    p.add_argument("--output", type=Path, default=Path("FSOI/fsoi_outputs/paper_figures_v2"))
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    for path in build(load_all_targets(), a.instrument, a.output, a.dpi):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
