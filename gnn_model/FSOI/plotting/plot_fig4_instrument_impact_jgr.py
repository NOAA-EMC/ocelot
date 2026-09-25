#!/usr/bin/env python
"""Manuscript Figure 4, sized and formatted for JGR.

Same content as figure_impact in plot_paper_figures_v2.py, redrawn at final
print size with source names in their published forms.

The pooling is repeated here rather than imported because the version in
plot_paper_figures_v2.py does not drop first-of-month cycles, whose background
is formed without a preceding cycle and which are scored nowhere in the paper.
Including them shifts every entry (aircraft -0.3679 to -0.3712, ATMS under
radiosonde verification -0.0050 to -0.0064) and so disagrees with Table 3. With
the exclusion the figure reproduces Table 3 to four decimal places.

AGU graphics requirements: two-column width 105-170 mm, height at most 228 mm,
Arial or Helvetica text no smaller than 8 pt at final print size, raster at
300-600 ppi. Saved as vector PDF plus 600 ppi TIFF and PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The 6.5 in text column inside 1 in margins; still inside AGU's
# 105-170 mm range for a two-column figure.
WIDTH_IN = 165.1 / 25.4
BASE_PT = 8.0
INK = "#1a1a1a"
BLUE, RED = "#2166ac", "#b2182b"      # beneficial, detrimental
ROOT = Path(__file__).resolve().parent.parent / "fsoi_outputs" / "seasonal_inclusion_weighted_final"
DISPLAY = {"amsua": "AMSU-A", "atms": "ATMS", "avhrr": "AVHRR", "ssmis": "SSMIS",
           "ascat": "ASCAT", "seviri_asr": "SEVIRI ASR", "aircraft": "Aircraft",
           "radiosonde": "Radiosonde", "surface_obs": "Surface"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "font.size": BASE_PT,
    "axes.titlesize": BASE_PT + 1,
    "axes.labelsize": BASE_PT,
    "xtick.labelsize": BASE_PT,
    "ytick.labelsize": BASE_PT,
    "axes.linewidth": 0.5,
    "axes.edgecolor": INK,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def name(key: str) -> str:
    return DISPLAY.get(str(key), str(key).replace("_", " "))


def pooled_scored(name: str = "fsoi_combined_by_instrument.csv",
                  keys: list[str] | None = None) -> pd.Series:
    """Impact relative to J, pooled over the cycles the paper scores.

    Cycles within a month, then months within a target. Cycles valid on the
    first of the month are dropped, as everywhere else in the paper.

    Shared with the Figure S2 script so both use one definition.
    """
    keys = keys or ["instrument"]
    frames = []
    for path in sorted(ROOT.glob(f"*/csv/{name}")):
        target, month = path.parts[-3].rsplit("_", 1)
        d = pd.read_csv(path)
        d = d[~d.curr_bin.astype(str).str[9:11].eq("01")]
        d["rel"] = d.sum_impact_ht / d.ea
        d["target"], d["month"] = target, month
        frames.append(d.groupby(["target", "month", "curr_bin"] + keys,
                                as_index=False)["rel"].sum())
    d = pd.concat(frames)
    return (d.groupby(["target", "month"] + keys)["rel"].mean()
             .groupby(["target"] + keys).mean())


def build(out_dir: Path, dpi: int) -> list[Path]:
    s = pooled_scored().unstack(0)
    cross = s.copy()
    for inst in cross.index:
        if inst in cross.columns:
            cross.loc[inst, inst] = np.nan
    order = s.mean(axis=1).sort_values().index
    s, cross = s.loc[order], cross.loc[order].mean(axis=1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.85),
                                   gridspec_kw={"width_ratios": [1.30, 1.0], "wspace": 0.42})

    v = float(np.abs(s.values).max())
    im = ax1.imshow(s.values, cmap="RdBu_r", vmin=-v, vmax=v, aspect="auto")
    ax1.set_xticks(range(s.shape[1]), [name(c) for c in s.columns])
    ax1.set_yticks(range(s.shape[0]), [name(i) for i in s.index])
    ax1.set_xlabel("Verification target", labelpad=2)
    ax1.set_title("(a) Impact per cycle, relative to $J$", loc="left", pad=4)
    for i in range(s.shape[0]):
        for j in range(s.shape[1]):
            val = s.values[i, j]
            ax1.text(j, i, f"{val:+.3f}", ha="center", va="center", fontsize=BASE_PT,
                     color="white" if abs(val) > 0.45 * v else INK)
    ax1.tick_params(pad=1.5)
    cb = fig.colorbar(im, ax=ax1, shrink=0.92, pad=0.02, aspect=14)
    cb.set_label("Relative impact", size=BASE_PT, labelpad=2)
    cb.ax.tick_params(labelsize=BASE_PT, width=0.5, length=2.0, pad=1.0)
    cb.outline.set_linewidth(0.5)

    ax2.barh(range(len(cross)), cross.values, height=0.7,
             color=[BLUE if x < 0 else RED for x in cross.values], zorder=2)
    ax2.set_yticks(range(len(cross)), [name(i) for i in cross.index])
    ax2.axvline(0, color="0.3", lw=0.6, zorder=3)
    ax2.set_xscale("symlog", linthresh=1e-3)
    ax2.set_xlabel("Cross-target impact (own network excluded)", labelpad=2)
    ax2.set_title("(b) Comparable between instruments", loc="left", pad=4)
    ax2.grid(axis="x", color="#e2e2e2", lw=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(pad=1.5)
    # The symlog labels are 10^-2 and 10^-3; superscripts render at 0.7x, so the
    # axis runs at 9 pt to keep the exponents above AGU's 6 pt floor.
    ax2.tick_params(axis="x", labelsize=BASE_PT + 1.0)

    fig.subplots_adjust(left=0.105, right=0.995, top=0.90, bottom=0.185)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "figure4_instrument_impact_jgr"
    written = []
    for path, kw in [(out_dir / f"{stem}.pdf", {}),
                     (out_dir / f"{stem}.png", {"dpi": dpi}),
                     (out_dir / f"{stem}.tif", {"dpi": dpi, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    s.assign(cross_target=cross).to_csv(out_dir / f"{stem}_source.csv")
    written.append(out_dir / f"{stem}_source.csv")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("FSOI/fsoi_outputs/paper_figures_v2"))
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    for path in build(a.output, a.dpi):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
