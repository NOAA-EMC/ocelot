#!/usr/bin/env python
"""Supporting Information Figure S3, sized and formatted for JGR.

Cycle-level validation errors: estimated against realized change for combined
and conditional replacement, and the cumulative distributions of the closure
ratio and the relative error.

Reads the two closure tables directly. Those already hold only the scored
cycles, 694 combined and 913 conditional, so no further filtering is applied.

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
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# The 6.5 in text column inside 1 in margins; still inside AGU's
# 105-170 mm range for a two-column figure.
WIDTH_IN = 165.1 / 25.4
BASE_PT = 8.0
INK = "#1a1a1a"

NET = {"aircraft": "#2a78d6", "radiosonde": "#eb6834", "surface_obs": "#1baf7a"}
INST = {"aircraft": "#2a78d6", "atms": "#eb6834", "amsua": "#1baf7a", "seviri_asr": "#4a3aa7"}
DISPLAY = {"aircraft": "Aircraft", "radiosonde": "Radiosonde", "surface_obs": "Surface",
           "atms": "ATMS", "amsua": "AMSU-A", "seviri_asr": "SEVIRI ASR"}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.fallback": "stixsans",
    "font.size": BASE_PT,
    "axes.labelsize": BASE_PT + 1.0,     # these labels carry subscripts
    "axes.titlesize": BASE_PT + 1.0,
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


def name(key) -> str:
    return DISPLAY.get(str(key), str(key).replace("_", " "))


def build(src: Path, out_dir: Path, dpi: int) -> list[Path]:
    C = pd.read_csv(src / "closure_cycles_combined.csv")
    COND = pd.read_csv(src / "closure_cycles_conditional.csv")
    COND = COND[COND.matched_signal_valid.astype(str).str.lower().eq("true")]
    C["est"], C["real"] = C.fsoi_raw_sampled, C.delta_j_actual
    COND["est"], COND["real"] = COND.matched_fsoi, COND.delta_j_actual
    for f in (C, COND):
        f["rel_err"] = (f.est - f.real).abs() / f.real.abs()
    C["rho"], COND["rho"] = C.closure_ratio, COND.matched_closure_ratio
    C["grp"], COND["grp"] = C.target, COND.instrument

    fig, axes = plt.subplots(2, 2, figsize=(WIDTH_IN, 4.70))

    for ax, (d, colours, title) in zip(axes[0], [
            (C, NET, "(a) Combined replacement, all sources"),
            (COND, INST, "(b) Conditional replacement, one instrument")]):
        lim = max(d.est.abs().max(), d.real.abs().max()) * 1.08
        ax.plot([-lim, lim], [-lim, lim], color="#999999", lw=0.7, zorder=1)
        for g, sub in d.groupby("grp"):
            ax.scatter(sub.real, sub.est, s=5, alpha=0.65, linewidths=0,
                       color=colours.get(g, "#777777"),
                       label=f"{name(g)} (n={len(sub)})", zorder=2)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color="#cccccc", lw=0.5)
        ax.axvline(0, color="#cccccc", lw=0.5)
        ax.set_xlabel("Realized change $\\Delta J$", labelpad=2)
        ax.set_ylabel("Estimated $I$", labelpad=2)
        ax.set_title(title, fontweight="bold", loc="left", pad=4)
        ax.legend(fontsize=BASE_PT, frameon=False, loc="upper left", handletextpad=0.2,
                  labelspacing=0.25, borderaxespad=0.3)
        ax.grid(alpha=0.25, lw=0.4)
        ax.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2), useMathText=True)
        ax.xaxis.get_offset_text().set_size(BASE_PT + 1.0)
        ax.yaxis.get_offset_text().set_size(BASE_PT + 1.0)
        ax.tick_params(pad=1.5)

    ax = axes[1][0]
    for d, colours, style in ((C, NET, "-"), (COND, INST, "--")):
        for g, sub in d.groupby("grp"):
            x = np.sort(sub.rho.values)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), style, lw=0.9,
                    color=colours.get(g, "#777777"), label=name(g))
    ax.axvline(1.0, color="#444444", lw=0.6)
    ax.axvspan(0.97, 1.05, color="#dddddd", alpha=0.5, zorder=0)
    ax.set_xlim(0, 2)
    ax.set_xlabel("Closure ratio", labelpad=2)
    ax.set_ylabel("Cumulative fraction of cycles", labelpad=2)
    ax.set_title("(c) Closure ratio: combined solid, conditional dashed",
                 fontweight="bold", loc="left", pad=32)
    # A single column hugging the right edge: past a ratio of about 1.2 every
    # curve has reached one, so only that strip is free of data. Two columns
    # reached back into the rising part of the distributions.
    ax.legend(fontsize=BASE_PT, frameon=False, ncol=1, loc="lower right",
              handlelength=1.3, handletextpad=0.3, labelspacing=0.2,
              borderaxespad=0.3)
    ax.grid(alpha=0.25, lw=0.4)
    ax.tick_params(pad=1.5)

    ax = axes[1][1]
    for d, colours, style in ((C, NET, "-"), (COND, INST, "--")):
        for g, sub in d.groupby("grp"):
            x = np.sort(sub.rel_err.replace(0, np.nan).dropna().values)
            ax.plot(x, np.arange(1, len(x) + 1) / len(x), style, lw=0.9,
                    color=colours.get(g, "#777777"))
    ax.set_xscale("log")
    ax.set_xlabel("Relative error $|I-\\Delta J| \\, / \\, |\\Delta J|$", labelpad=2)
    ax.set_ylabel("Cumulative fraction of cycles", labelpad=2)
    ax.set_title("(d) Relative error", fontweight="bold", loc="left", pad=32)
    ax.grid(alpha=0.25, lw=0.4, which="both")
    ax.tick_params(pad=1.5)
    # The decade labels are 10^n; superscripts render at 0.7x, so this axis runs
    # at 9 pt to keep the exponents above AGU's 6 pt floor.
    ax.tick_params(axis="x", labelsize=BASE_PT + 1.0)

    # A header line names the three quantities so each row can stay short.
    lines, summary = ["median / p90 / sign disagreements"], []
    for label, d in (("Combined", C), ("Conditional", COND)):
        dis = int((d.sign_agreement.eq(0).sum()) if "sign_agreement" in d else
                  (~d.matched_sign_agree.astype(str).str.lower().eq("true")).sum())
        lines.append(f"{label}: {d.rel_err.median():.3f} / "
                     f"{d.rel_err.quantile(.9):.3f} / {dis} of {len(d)}")
        summary.append({"set": label, "n": len(d), "median_rel_err": d.rel_err.median(),
                        "p90_rel_err": d.rel_err.quantile(.9), "sign_disagreements": dis})
    # Above the axes: this block is wider than either free corner, so anywhere
    # inside the panel it sat on the distributions.
    ax.text(0.5, 1.015, "\n".join(lines), transform=ax.transAxes, va="bottom",
            ha="center", fontsize=BASE_PT, color="#333333", linespacing=1.3)

    fig.subplots_adjust(left=0.085, right=0.995, top=0.905, bottom=0.095,
                        hspace=0.62, wspace=0.28)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "figureS3_error_distribution_jgr"
    written = []
    for path, kw in [(out_dir / f"{stem}.pdf", {}),
                     (out_dir / f"{stem}.png", {"dpi": dpi}),
                     (out_dir / f"{stem}.tif", {"dpi": dpi, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    pd.DataFrame(summary).to_csv(out_dir / f"{stem}_source.csv", index=False)
    written.append(out_dir / f"{stem}_source.csv")

    out = COND[(COND.rho < 0) | (COND.rho > 2)]
    print(f"conditional cycles outside [0, 2]: {len(out)} of {len(COND)} "
          f"({int((COND.rho < 0).sum())} below zero, {int((COND.rho > 2).sum())} above two)")
    print("  by instrument:", {name(k): int(v) for k, v in out.grp.value_counts().items()})
    print(f"combined cycles outside [0, 2]: {int(((C.rho < 0) | (C.rho > 2)).sum())} of {len(C)}")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--src", type=Path, default=Path("FSOI/fsoi_outputs/paper_figures_v2"))
    p.add_argument("--output", type=Path, default=Path("FSOI/fsoi_outputs/paper_figures_v2"))
    p.add_argument("--dpi", type=int, default=600)
    a = p.parse_args()
    for path in build(a.src, a.output, a.dpi):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
