#!/usr/bin/env python
"""Manuscript Figure 3, sized and formatted for JGR.

Composite Simpson path integration against the endpoint estimate, for the cycles
that carry the deepest refinement. The figure is drawn at final print size, so
nothing is shrunk by the page layout, and the marks and rules are scaled to
match.

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

FSOI = Path(__file__).resolve().parent.parent
# Radiosonde-metric runs and the surface-metric SEVIRI runs. Closure is a
# dimensionless ratio, so cycles scored against different frozen metrics can
# share an axis; the caption says which rows use which metric.
DEFAULT_ROOTS = [FSOI / "fsoi_outputs" / "ose_frozen_metric",
                 FSOI / "fsoi_outputs" / "ose_frozen_metric_surface"]
DEFAULT_OUT = FSOI / "fsoi_outputs" / "paper_figures_v2"

# Categorical slots 1-4 of the validated palette; every adjacent pair clears the
# CVD floor (worst deutan dE 9.2); the red/blue/green of
# the previous figure put orange next to green, which fails protan at dE 3.2.
COLORS = {"aircraft": "#2a78d6", "amsua": "#eb6834", "atms": "#1baf7a",
          "seviri_asr": "#4a3aa7"}
# Marker shape repeats the instrument identity, so colour is never the only cue.
MARKERS = {"aircraft": "o", "amsua": "s", "atms": "^", "seviri_asr": "D"}
LABELS = {"aircraft": "Aircraft", "amsua": "AMSU-A", "atms": "ATMS",
          "seviri_asr": "SEVIRI ASR"}
ORDER = ["aircraft", "amsua", "atms", "seviri_asr"]
# Cycles valid on the 1st of the month are not scored anywhere in the paper.
EXCLUDE_FIRST_DAY = True
# Alternate (colour, marker) for a second or third cycle of one instrument in
# panel (a); shade and marker together keep repeated curves distinguishable.
VARIANTS = {"seviri_asr": [("#4a3aa7", "D"), ("#8f84dc", "P"), ("#241a5e", "X")],
            "atms": [("#1baf7a", "^"), ("#0d6e4c", "v"), ("#6fd3ab", "<")],
            "amsua": [("#eb6834", "s"), ("#a8401a", "p"), ("#f4a07f", "h")],
            "aircraft": [("#2a78d6", "o"), ("#164f96", "8"), ("#7fb0ec", "H")]}
INK, MUTED, GRID = "#12130f", "#52514e", "#dbe3ed"

# The 6.5 in text column inside 1 in margins; still inside AGU's
# 105-170 mm range for a two-column figure.
WIDTH_IN = 165.1 / 25.4
BASE_PT = 8.0
INK, GRID = "#12130f", "#dbe3ed"

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
    "legend.fontsize": BASE_PT,
    "axes.linewidth": 0.5,
    "axes.edgecolor": MUTED,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "text.color": INK,
    "axes.labelcolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def _floats(cell) -> np.ndarray:
    return np.array([float(v) for v in str(cell).replace(";", ",").split(",") if v.strip()])


def _date(curr_bin: str) -> str:
    """bin2025070412 -> 04 Jul. A month alone is ambiguous once a run carries
    several path cycles."""
    b = str(curr_bin)
    try:
        return pd.Timestamp(f"{b[3:7]}-{b[7:9]}-{b[9:11]}").strftime("%d %b")
    except Exception:
        return b


def load_path_cycles(roots) -> pd.DataFrame:
    frames = []
    csvs = [c for root in roots for c in sorted(Path(root).glob("*/evaluation/ose_results.csv"))]
    for csv in csvs:
        d = pd.read_csv(csv)
        if "path_integration_enabled" not in d.columns:
            continue  # the run never requested path integration, or its pair was excluded
        d = d[d.path_integration_enabled.astype(str).str.strip().str.lower().eq("true")]
        if len(d):
            frames.append(d.assign(run=csv.parent.parent.name))
    if not frames:
        raise SystemExit(f"no path-integration cycles under {[str(r) for r in roots]}")
    d = pd.concat(frames, ignore_index=True)
    if EXCLUDE_FIRST_DAY:
        d = d[~d.curr_bin.astype(str).str[9:11].eq("01")]
    d["label"] = [f"{LABELS.get(i, i)}  {_date(b)}" for i, b in zip(d.denied_instruments, d.curr_bin)]
    d["_rank"] = [ORDER.index(i) if i in ORDER else len(ORDER) for i in d.denied_instruments]
    return d.sort_values(["_rank", "curr_bin"]).reset_index(drop=True)


def select_examples(d: pd.DataFrame, n_curved: int = 3) -> pd.DataFrame:
    """The n_curved most curved cycles, plus the straightest cycle from an
    instrument not already shown (or else the straightest remaining cycle), so
    the panel contrasts curvature with its absence."""
    miss = (d.matched_closure_ratio - 1.0).abs()
    curved = d.loc[miss.sort_values(ascending=False).index[:n_curved]]
    rest = d[~d.denied_instruments.isin(curved.denied_instruments)]
    rest = rest if len(rest) else d.drop(curved.index)
    straight = rest.loc[[(rest.matched_closure_ratio - 1.0).abs().idxmin()]]
    return pd.concat([straight, curved])


def panel_derivative(ax, d) -> None:
    ax.grid(True, color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.axhline(0, color="#94a3b8", linewidth=0.5)
    used = {}
    for _, r in select_examples(d).iterrows():
        inst = r.denied_instruments
        k = used.get(inst, 0)
        used[inst] = k + 1
        variants = VARIANTS.get(inst, [(COLORS.get(inst, MUTED), MARKERS.get(inst, "o"))])
        colour, marker = variants[min(k, len(variants) - 1)]
        t, f = _floats(r.path_integration_t_values), _floats(r.path_directional_derivatives)
        peak = np.max(np.abs(f))
        f = f / peak if peak > 0 else f
        ax.plot(t, f, "-", color=colour, linewidth=1.0, marker=marker, markersize=3.2,
                markerfacecolor="white", markeredgecolor=colour, markeredgewidth=0.8,
                zorder=3, label=f"{r.label}  two-point {float(r.matched_closure_ratio):.2f}")
        ax.plot([t[0], t[-1]], [f[0], f[-1]], "--", color=colour, linewidth=0.7,
                alpha=0.6, zorder=2)
    ax.set_xlabel("Path fraction $t$ (denied $\\to$ control)", labelpad=2)
    ax.set_ylabel("Directional derivative (scaled by curve peak)", labelpad=2)
    ax.set_title("(a) Curvature along the path", loc="left", color=INK, pad=4)
    ax.legend(frameon=False, loc="lower right", handlelength=1.6, handletextpad=0.4,
              labelspacing=0.3, borderaxespad=0.3)
    ax.tick_params(pad=1.5)


def panel_closure(ax, d) -> None:
    y = np.arange(len(d))[::-1]
    ax.grid(True, axis="x", color=GRID, linewidth=0.4)
    ax.set_axisbelow(True)
    ax.axvline(1.0, color="#94a3b8", linewidth=0.7, zorder=1)
    ax.axvline(0.0, color="#cbd5e1", linewidth=0.5, linestyle="--", zorder=1)
    for yi, (_, r) in zip(y, d.iterrows()):
        inst = r.denied_instruments
        colour = COLORS.get(inst, MUTED)
        a, b = float(r.matched_closure_ratio), float(r.path_closure_ratio)
        ax.plot([a, b], [yi, yi], "-", color=colour, linewidth=1.1, alpha=0.55, zorder=2)
        ax.plot(a, yi, MARKERS.get(inst, "o"), markersize=4.0, markerfacecolor="white",
                markeredgecolor=colour, markeredgewidth=1.0, zorder=3)
        ax.plot(b, yi, MARKERS.get(inst, "o"), markersize=4.0, color=colour, zorder=4)
    ax.set_yticks(y, d.label.tolist())
    ax.set_xlim(min(-0.2, float(d.matched_closure_ratio.min()) - 0.25),
                max(2.0, float(d.matched_closure_ratio.max()) + 0.25))
    ax.set_ylim(-0.7, len(d) - 0.3)
    ax.set_xlabel("Closure ratio (estimate / realized $\\Delta J$)", labelpad=2)
    ax.set_title("(b) Closure ratio, all cycles", loc="left", color=INK, pad=4)
    ax.text(1.0, len(d) - 0.55, " ideal", color=MUTED, fontsize=BASE_PT, va="center")
    ax.tick_params(pad=1.5)


def build(roots, out_dir: Path, dpi: int) -> list[Path]:
    d = load_path_cycles(roots)
    print(f"{len(d)} path-integration cycles")
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_IN, 0.195 * len(d) + 1.25),
                             gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.52})
    panel_derivative(axes[0], d)
    panel_closure(axes[1], d)
    fig.subplots_adjust(left=0.085, right=0.995, top=0.93, bottom=0.115)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "figure3_path_integration_jgr"
    written = []
    for path, kw in [(out_dir / f"{stem}.pdf", {}),
                     (out_dir / f"{stem}.png", {"dpi": dpi}),
                     (out_dir / f"{stem}.tif", {"dpi": dpi, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    cols = ["label", "denied_instruments", "curr_bin", "matched_closure_ratio",
            "path_closure_ratio"]
    d[cols].to_csv(out_dir / f"{stem}_source.csv", index=False)
    written.append(out_dir / f"{stem}_source.csv")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, nargs="+", default=DEFAULT_ROOTS)
    p.add_argument("--output", type=Path, default=Path("FSOI/fsoi_outputs/paper_figures_v2"))
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    for path in build(a.root, a.output, a.dpi):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
