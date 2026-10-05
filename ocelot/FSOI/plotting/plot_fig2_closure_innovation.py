#!/usr/bin/env python
"""Manuscript Figure 2, sized and formatted for JGR.

Panel (a) shows the four monthly median closure ratios of each verification
network, drawn as individual points so the span is not read as a confidence
interval. Panel (b) shows the background innovation RMS of each source.

AGU graphics requirements: two-column width 105-170 mm, height at most 228 mm,
Arial or Helvetica text no smaller than 8 pt at final print size, raster at
300-600 ppi. Saved as vector PDF plus 600 ppi TIFF and PNG.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# The 6.5 in text column inside 1 in margins; still inside AGU's
# 105-170 mm range for a two-column figure.
FULL_WIDTH_IN = 165.1 / 25.4
BASE_PT = 8.0                         # AGU minimum text size at final size

MONTHS = [("jan", "January", "o"), ("apr", "April", "s"),
          ("jul", "July", "^"), ("oct", "October", "D")]
NETWORKS = [("aircraft", "Aircraft"), ("radiosonde", "Radiosonde"), ("surface_obs", "Surface")]
DISPLAY = {"amsua": "AMSU-A", "atms": "ATMS", "avhrr": "AVHRR", "ssmis": "SSMIS",
           "radiosonde": "Radiosonde", "surface_obs": "Surface", "aircraft": "Aircraft",
           "ascat": "ASCAT", "seviri_asr": "SEVIRI ASR"}
CONVENTIONAL = {"radiosonde", "surface_obs", "aircraft"}

INK = "#1a1a1a"
SAT = "#2c5f8a"       # satellite sources
CONV = "#b5651d"      # conventional sources; differs from SAT in greyscale too
RULE = "#8c8c8c"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "font.size": BASE_PT,
    "axes.labelsize": BASE_PT,
    "axes.titlesize": BASE_PT + 1,
    "xtick.labelsize": BASE_PT,
    "ytick.labelsize": BASE_PT,
    "legend.fontsize": BASE_PT,
    "axes.linewidth": 0.6,
    "axes.edgecolor": INK,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,          # embed TrueType rather than Type 3
    "ps.fonttype": 42,
    "svg.fonttype": "none",
})


def build(root: Path, out_dir: Path, dpi: int) -> list[Path]:
    closure = pd.read_csv(root / "closure_cycles_combined.csv")
    innov = pd.read_csv(root / "figureB_validation_source.csv")

    medians = (closure.groupby(["target", "month"]).closure_ratio.median()
               .rename("median_closure_ratio").reset_index())

    fig, axes = plt.subplots(
        1, 2, figsize=(FULL_WIDTH_IN, 2.95),
        gridspec_kw={"width_ratios": [1.0, 1.12], "wspace": 0.30})

    # ---------------------------------------------------------------- panel a
    ax = axes[0]
    for row, (key, label) in enumerate(NETWORKS):
        vals = medians[medians.target == key].set_index("month").median_closure_ratio
        ax.plot([vals.min(), vals.max()], [row, row], color=RULE, lw=1.0, zorder=2,
                solid_capstyle="round")
        for mkey, mlabel, marker in MONTHS:
            ax.plot(vals[mkey], row, marker=marker, ms=4.2, mfc="white", mec=SAT,
                    mew=1.0, ls="none", zorder=3)
    ax.axvline(1.0, color=INK, lw=0.7, ls=(0, (4, 2)), zorder=1)
    ax.set_yticks(range(len(NETWORKS)))
    ax.set_yticklabels([lbl for _, lbl in NETWORKS])
    ax.set_ylim(len(NETWORKS) - 0.5, -0.75)
    ax.set_xlim(0.96, 1.06)
    ax.set_xticks([0.96, 0.98, 1.00, 1.02, 1.04, 1.06])
    ax.set_xlabel("Monthly median closure ratio")
    ax.grid(axis="x", color="#e2e2e2", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker=mk, ls="none", ms=4.2, mfc="white",
                              mec=SAT, mew=1.0, label=lbl) for _, lbl, mk in MONTHS],
              loc="upper right", ncol=2, frameon=False, handletextpad=0.25,
              columnspacing=0.9, labelspacing=0.35, borderaxespad=0.4)
    ax.set_title("(a)", loc="left", fontweight="bold", pad=6)

    # ---------------------------------------------------------------- panel b
    ax = axes[1]
    bars = (innov.assign(rms=innov["after fix"])
            .sort_values("rms", ascending=False).reset_index(drop=True))
    colors = [CONV if i in CONVENTIONAL else SAT for i in bars.instrument]
    ax.barh(range(len(bars)), bars.rms, color=colors, height=0.68, zorder=2)
    for y, v in enumerate(bars.rms):
        ax.text(v + 0.008, y, f"{v:.3f}", va="center", ha="left", fontsize=BASE_PT)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([DISPLAY.get(i, i) for i in bars.instrument])
    ax.set_ylim(len(bars) - 0.5, -0.5)
    ax.set_xlim(0, max(bars.rms) * 1.16)
    ax.set_xlabel("Background innovation RMS (normalized units)")
    ax.grid(axis="x", color="#e2e2e2", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[Line2D([], [], marker="s", ls="none", ms=5, color=CONV,
                              label="Conventional"),
                       Line2D([], [], marker="s", ls="none", ms=5, color=SAT,
                              label="Satellite")],
              loc="lower right", frameon=False, handletextpad=0.3)
    ax.set_title("(b)", loc="left", fontweight="bold", pad=6)

    fig.subplots_adjust(left=0.108, right=0.985, top=0.93, bottom=0.15)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "figure2_closure_innovation"
    written = []
    for path, kw in [(out_dir / f"{stem}.pdf", {}),
                     (out_dir / f"{stem}.png", {"dpi": dpi}),
                     (out_dir / f"{stem}.tif", {"dpi": dpi, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)

    src = medians.assign(panel="a").rename(columns={"median_closure_ratio": "value"})
    src = pd.concat([
        src[["panel", "target", "month", "value"]],
        bars.assign(panel="b", month="all", target=bars.instrument,
                    value=bars.rms)[["panel", "target", "month", "value"]]])
    src.to_csv(out_dir / f"{stem}_source.csv", index=False)
    written.append(out_dir / f"{stem}_source.csv")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("FSOI/fsoi_outputs/paper_figures_v2"))
    p.add_argument("--output", type=Path, default=Path("FSOI/fsoi_outputs/paper_figures_v2"))
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    for path in build(a.root, a.output, a.dpi):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
