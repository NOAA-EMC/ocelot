#!/usr/bin/env python
"""Supporting Information Figure S2, sized and formatted for JGR.

Same content as figure_channels in plot_paper_figures_v2.py, redrawn at final
print size with source names in their published forms.

The pooling comes from plot_fig4_instrument_impact_jgr, which drops first-of-month
cycles; the version in plot_paper_figures_v2.py does not, and including those
cycles disagrees with the channel means quoted in section 3.6.

AGU graphics requirements: two-column width 105-170 mm, height at most 228 mm,
Arial or Helvetica text no smaller than 8 pt at final print size, raster at
300-600 ppi. Saved as vector PDF plus 600 ppi TIFF and PNG.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from plot_fig4_instrument_impact_jgr import BLUE, DISPLAY, RED, pooled_scored

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# The 6.5 in text column inside 1 in margins; still inside AGU's
# 105-170 mm range for a two-column figure.
WIDTH_IN = 165.1 / 25.4
BASE_PT = 8.0
INK = "#1a1a1a"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.fallback": "stixsans",
    "font.size": BASE_PT,
    "axes.labelsize": BASE_PT,
    "axes.titlesize": BASE_PT + 1,
    "xtick.labelsize": BASE_PT,
    "ytick.labelsize": BASE_PT,
    "legend.fontsize": BASE_PT,
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


def build(out_dir: Path, dpi: int) -> list[Path]:
    s = pooled_scored("fsoi_combined_by_channel.csv", ["instrument", "channel"])
    s = s.groupby(["instrument", "channel"]).mean()
    atms = s.loc["atms"].sort_index()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(WIDTH_IN, 2.55),
                                   gridspec_kw={"width_ratios": [1.55, 1.0], "wspace": 0.30})

    ax1.bar(atms.index.astype(int), atms.values, width=0.72,
            color=[BLUE if v < 0 else RED for v in atms.values], zorder=2)
    ax1.axhline(0, color="0.3", lw=0.6, zorder=3)
    ax1.set_xlabel("ATMS channel", labelpad=2)
    ax1.set_ylabel("Relative impact, pooled over targets", labelpad=2)
    ax1.set_title("(a) ATMS channel impact", loc="left", pad=4)
    ax1.set_xticks(atms.index.astype(int))
    ax1.grid(axis="y", color="#e2e2e2", lw=0.5, zorder=0)
    ax1.set_axisbelow(True)
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.tick_params(pad=1.5)
    # The axis offset carries a superscript, which renders at 0.7x.
    ax1.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
    ax1.yaxis.get_offset_text().set_size(BASE_PT + 1.0)

    counts = pd.DataFrame({
        "beneficial": s.groupby("instrument").apply(lambda g: int((g <= 0).sum())),
        "detrimental": s.groupby("instrument").apply(lambda g: int((g > 0).sum())),
    })
    counts = counts.loc[counts.sum(axis=1).sort_values().index]
    ax2.barh(range(len(counts)), counts["beneficial"], height=0.7, color=BLUE,
             label="Beneficial", zorder=2)
    ax2.barh(range(len(counts)), counts["detrimental"], left=counts["beneficial"],
             height=0.7, color=RED, label="Detrimental", zorder=2)
    ax2.set_yticks(range(len(counts)),
                   [DISPLAY.get(i, str(i).replace("_", " ")) for i in counts.index])
    ax2.set_xlabel("Number of channels", labelpad=2)
    ax2.set_title("(b) Channel sign census", loc="left", pad=4)
    ax2.legend(frameon=False, loc="lower right", handlelength=1.2, handletextpad=0.4)
    ax2.grid(axis="x", color="#e2e2e2", lw=0.5, zorder=0)
    ax2.set_axisbelow(True)
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.tick_params(pad=1.5)

    fig.subplots_adjust(left=0.105, right=0.995, top=0.865, bottom=0.165)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "figureS2_channel_impact_jgr"
    written = []
    for path, kw in [(out_dir / f"{stem}.pdf", {}),
                     (out_dir / f"{stem}.png", {"dpi": dpi}),
                     (out_dir / f"{stem}.tif", {"dpi": dpi, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    s.to_frame("relative_impact").to_csv(out_dir / f"{stem}_source.csv")
    counts.to_csv(out_dir / f"{stem}_sign_counts.csv")
    written += [out_dir / f"{stem}_source.csv", out_dir / f"{stem}_sign_counts.csv"]
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
