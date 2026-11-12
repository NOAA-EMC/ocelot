#!/usr/bin/env python
"""Manuscript Figure 1, sized and formatted for JGR: Machine Learning and Computation.

Same schematic and the same three panels as plot_fig1_framework.py: (a) how J is
frozen before any impact is seen, (b) the endpoint paths through the frozen
model, (c) the estimator and its two checks. The content is unchanged, so the
caption still describes it exactly.

Redrawn at final print size. The original was built on an 11.6 in canvas and
scaled to the 170 mm column, which put its smallest text at 4.4 pt. Here the
canvas is the printed size, the type sits at the AGU floor, and the vertical
scale was opened up so the boxes still breathe around 8 pt text.

Panel tags are "(a)"-style to match the caption and Figures 2-6.

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
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

# The 6.5 in text column inside 1 in margins; still inside AGU's
# 105-170 mm range for a two-column figure.
WIDTH_IN = 165.1 / 25.4
HEIGHT_IN = 5.05          # opens up the vertical scale so 8 pt text fits the boxes

INK = "#1a1a1a"
MUTED = "#6b6b6b"
RULE = "#c8c8c8"
EDGE = "#b4b4b4"
BOX_FILL = "#fcfcfc"
MODEL_FILL = "#f4f1ea"
MODEL_EDGE = "#b9a887"
# Categorical slots 1-4 of the validated palette; worst adjacent CVD separation
# deutan dE 9.2. Each mode is also named, so colour is never the only cue.
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    # Arial has no nabla, so the operators fall back. STIX Sans is a sans-serif
    # face that sits beside Arial; matplotlib's default fallback is Computer
    # Modern, whose serifs read as a different font mid-equation.
    "mathtext.fallback": "stixsans",
    "text.color": INK,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Sub- and superscripts render at 0.7x, and AGU's floor for those is 6 pt, so
# every string carrying one sits at 9 pt or above rather than at the 8 pt body size.
F_PANEL = 9.5    # panel tag and panel title
F_TITLE = 9.0    # box titles, several of which carry a subscript
F_SUB = 8.0      # box subtitles, none of which carry sub- or superscripts
F_EQ = 9.5       # equations


def header(ax, x, y, tag, title, width=95.5):
    """Panel letter and title over a hairline rule; no enclosing box."""
    ax.text(x, y, tag, ha="left", va="center", fontsize=F_PANEL,
            fontweight="bold", color=INK)
    ax.text(x + 4.6, y, title, ha="left", va="center", fontsize=F_PANEL, color=INK)
    ax.plot([x, x + width], [y - 2.6, y - 2.6], color=RULE, lw=0.7)


def box(ax, x, y, w, h, title, sub=None, edge=EDGE, fill=BOX_FILL, accent=None, lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.6",
                                facecolor=fill, edgecolor=edge, linewidth=lw))
    if accent:  # a thin colour bar carries identity without tinting the whole box
        ax.add_patch(FancyBboxPatch((x, y), 0.8, h, boxstyle="square,pad=0",
                                    facecolor=accent, edgecolor="none"))
    cy = y + h / 2 + (1.9 if sub else 0)
    ax.text(x + w / 2, cy, title, ha="center", va="center", fontsize=F_TITLE,
            fontweight="bold", color=INK)
    if sub:
        ax.text(x + w / 2, cy - 4.0, sub, ha="center", va="center", fontsize=F_SUB,
                color=MUTED)


def arrow(ax, x1, y1, x2, y2, color=MUTED, lw=0.9):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                 mutation_scale=7, linewidth=lw, color=color,
                                 shrinkA=0, shrinkB=0))


def build(ax):
    # ---- (a) the frozen metric -------------------------------------------
    header(ax, 2.5, 96.0, "(a)", "Verification metric, frozen before any impact is computed")
    steps = [("Coverage audit", "four months, no model run", 2.5),
             ("Frozen groups", "≥ 90% of cycles", 35.0),
             ("Objective $J$", "648 equal-area cells", 67.5)]
    for name, sub, x in steps:
        box(ax, x, 80.5, 30.5, 10.6, name, sub)
    arrow(ax, 33.4, 85.8, 34.7, 85.8, color=INK, lw=1.0)
    arrow(ax, 65.9, 85.8, 67.2, 85.8, color=INK, lw=1.0)

    # ---- (b) endpoint paths ----------------------------------------------
    header(ax, 2.5, 72.5, "(b)", "Endpoint paths through the frozen model")

    box(ax, 2.5, 57.5, 25.0, 9.6, "Control  $x_c$", "all observations", accent=BLUE)

    box(ax, 2.5, 35.0, 25.0, 17.5, "", None)
    ax.text(15.0, 49.6, "Denied endpoint  $x_0$", ha="center", va="center",
            fontsize=F_TITLE, fontweight="bold", color=INK)
    for label, color, yy in (("Background replacement", ORANGE, 43.0),
                             ("Structural denial", VIOLET, 37.6)):
        ax.add_patch(FancyBboxPatch((4.4, yy + 1.0), 2.2, 1.0,
                                    boxstyle="square,pad=0",
                                    facecolor=color, edgecolor=color, linewidth=0.8))
        ax.text(7.8, yy + 1.5, label, ha="left", va="center", fontsize=F_SUB, color=INK)

    box(ax, 36.0, 35.0, 27.0, 32.1, "", None, edge=MODEL_EDGE, fill=MODEL_FILL, lw=0.9)
    ax.text(49.5, 63.6, "Frozen OCELOT", ha="center", va="center",
            fontsize=F_TITLE + 0.5, fontweight="bold", color=INK)
    for name, yy in (("instrument encoders", 56.8),
                     ("latent mesh processor", 51.0),
                     ("instrument decoders", 45.2)):
        ax.text(49.5, yy, name, ha="center", va="center", fontsize=F_SUB, color=INK)
    for yy in (53.9, 48.1):
        arrow(ax, 49.5, yy + 0.9, 49.5, yy - 0.9, color=MODEL_EDGE, lw=0.8)
    ax.plot([38.5, 60.5], [41.8, 41.8], color=MODEL_EDGE, lw=0.6)
    ax.text(49.5, 38.6, "decoders conditioned on target\nlocation, level and valid time",
            ha="center", va="center", fontsize=F_SUB, color="#8a6d3b", linespacing=1.5)

    arrow(ax, 27.9, 62.0, 35.6, 58.4, color=BLUE, lw=1.0)
    for yy, color in ((43.0, ORANGE), (37.6, VIOLET)):
        arrow(ax, 27.9, yy + 1.5, 35.6, 46.0, color=color, lw=1.0)

    box(ax, 71.0, 54.0, 26.5, 10.0, "$J$  and  $\\nabla J$", "value and gradient")
    box(ax, 71.0, 36.0, 26.5, 10.0, "$J$  only", "no denied-endpoint input", accent=VIOLET)
    arrow(ax, 63.4, 57.0, 70.6, 59.0)
    arrow(ax, 63.4, 43.0, 70.6, 41.0, color=VIOLET)

    # ---- (c) estimator and checks ----------------------------------------
    header(ax, 2.5, 27.5, "(c)", "Estimator, and the two checks on it")
    eqs = [("Two-endpoint FSOI",
            "$I = \\frac{1}{2}\\, d^{\\mathsf{T}} (g_c + g_0)$", BLUE, 2.5),
           ("Closure",
            "$\\rho = I \\,/\\, \\Delta J$", ORANGE, 35.0),
           ("Path integration",
            "$\\int_0^1 \\nabla J(x_0 + t d)^{\\mathsf{T}} d \\;\\, \\mathrm{d}t$", AQUA, 67.5)]
    for name, eq, color, x in eqs:
        box(ax, x, 10.5, 30.5, 13.0, "", None, accent=color)
        ax.text(x + 15.6, 19.9, name, ha="center", va="center", fontsize=F_TITLE,
                fontweight="bold", color=INK)
        ax.text(x + 15.6, 14.6, eq, ha="center", va="center", fontsize=F_EQ, color=INK)
    ax.text(50.0, 4.5, "$d = x_c - x_0$;   positive $I$ or positive $\\Delta J$ means the "
                       "observation increased the frozen forecast-error metric",
            ha="center", va="center", fontsize=F_TITLE, color=MUTED)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("FSOI/fsoi_outputs/paper_figures_v2"))
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()

    fig = plt.figure(figsize=(WIDTH_IN, HEIGHT_IN))
    # Full-bleed axes and a fixed data range: the figure is saved at exactly this
    # size, so a tight bounding box cannot shrink the type after the fact.
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    build(ax)

    a.output.mkdir(parents=True, exist_ok=True)
    stem = "figure1_framework_jgr"
    for path, kw in [(a.output / f"{stem}.pdf", {}),
                     (a.output / f"{stem}.png", {"dpi": a.dpi}),
                     (a.output / f"{stem}.tif", {"dpi": a.dpi,
                                                 "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(path, **kw)
        print(f"Saved: {path}")
    plt.close(fig)
    print(f"figure is {WIDTH_IN * 25.4:.0f} x {HEIGHT_IN * 25.4:.0f} mm")


if __name__ == "__main__":
    main()
