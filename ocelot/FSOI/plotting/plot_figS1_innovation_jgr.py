#!/usr/bin/env python
"""Supporting Information Figure S1, sized and formatted for JGR.

The streaming loader keeps memory flat over the per-sample CSVs, which are too
large to hold at once, and the figure is drawn at final print size rather than
on an 18 in canvas that the page then scaled to 38%, which had put its smallest
text at 2.6 pt.

The in-figure title block is dropped because the caption carries it, including
that hexbin density uses sampled rows while panel statistics use all valid rows.

AGU graphics requirements: two-column width 105-170 mm, height at most 228 mm,
Arial or Helvetica text no smaller than 8 pt at final print size, raster at
300-600 ppi. Saved as vector PDF plus 600 ppi TIFF and PNG.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import MaxNLocator, ScalarFormatter  # noqa: E402

from plot_instrument_channel_heatmaps import DISPLAY_NAMES  # noqa: E402

SEASONS = ["jan2025", "apr2025", "jul2025", "oct2025"]
INSTRUMENT_ORDER = [
    "radiosonde",
    "aircraft",
    "surface_obs",
    "ssmis",
    "amsua",
    "atms",
    "ascat",
    "seviri_asr",
    "avhrr",
]
CONVENTIONAL = {"radiosonde", "aircraft", "surface_obs"}
SENTINEL_LO = -12.0
SENTINEL_HI = -7.0

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


def _display(inst: str) -> str:
    return DISPLAY_NAMES.get(inst, inst.upper())


def _scatter_paths(root: Path, target: str) -> list[Path]:
    paths = []
    for season in SEASONS:
        path = root / f"{target}_{season}" / "csv" / "scatter_samples.csv"
        if path.is_file():
            paths.append(path)
        else:
            print(f"[SKIP] Missing {path}")
    return paths


def _init_stats() -> dict:
    return {
        "n": 0,
        "sum_innov": 0.0,
        "sum_abs_innov": 0.0,
        "sum_innov2": 0.0,
        "sum_fsoi": 0.0,
        "sum_abs_fsoi": 0.0,
        "sum_fsoi2": 0.0,
        "n_pos_fsoi": 0,
    }


def _update_stats(stats: dict, innov: np.ndarray, fsoi: np.ndarray) -> None:
    n = int(innov.size)
    if n == 0:
        return
    stats["n"] += n
    stats["sum_innov"] += float(np.sum(innov))
    stats["sum_abs_innov"] += float(np.sum(np.abs(innov)))
    stats["sum_innov2"] += float(np.sum(innov * innov))
    stats["sum_fsoi"] += float(np.sum(fsoi))
    stats["sum_abs_fsoi"] += float(np.sum(np.abs(fsoi)))
    stats["sum_fsoi2"] += float(np.sum(fsoi * fsoi))
    stats["n_pos_fsoi"] += int(np.sum(fsoi > 0.0))


def _downsample(existing: pd.DataFrame, incoming: pd.DataFrame, cap: int, seed: int) -> pd.DataFrame:
    if incoming.empty:
        return existing
    combined = incoming if existing.empty else pd.concat([existing, incoming], ignore_index=True)
    if len(combined) <= cap:
        return combined
    return combined.sample(n=cap, random_state=seed).reset_index(drop=True)


def _first_of_month_pairs(csv_dir: Path) -> set:
    """pair_idx values whose cycle is valid on the first of the month.

    Those cycles are scored nowhere in the paper, and scatter_samples.csv carries
    pair_idx but not the cycle time, so the mapping comes from a sibling CSV.
    """
    meta = pd.read_csv(csv_dir / "fsoi_by_channel.csv", usecols=["pair_idx", "curr_bin"])
    meta = meta.drop_duplicates()
    return set(meta.loc[meta.curr_bin.astype(str).str[9:11].eq("01"), "pair_idx"])


def load_scored_samples(paths, max_points_per_instrument: int, chunksize: int, seed: int):
    """load_samples, but skipping first-of-month cycles.

    The published figure excludes them; including them inflates every count by
    about 1.3% and would disagree with Table S2.
    """
    samples: dict = {}
    stats: dict = {}
    cols = ["instrument", "innovation", "fsoi"]

    for file_index, path in enumerate(paths):
        drop = _first_of_month_pairs(path.parent)
        print(f"[READ] {path}  (dropping pair_idx {sorted(drop)})")
        for chunk_index, chunk in enumerate(
                pd.read_csv(path, usecols=cols + ["pair_idx"], chunksize=chunksize)):
            chunk = chunk[~chunk["pair_idx"].isin(drop)]
            chunk = chunk[cols].replace([np.inf, -np.inf], np.nan).dropna(subset=cols)
            if chunk.empty:
                continue
            chunk["instrument"] = chunk["instrument"].astype(str).str.lower()
            chunk["innovation"] = pd.to_numeric(chunk["innovation"], errors="coerce")
            chunk["fsoi"] = pd.to_numeric(chunk["fsoi"], errors="coerce")
            chunk = chunk.dropna(subset=["innovation", "fsoi"])
            if chunk.empty:
                continue
            innov = chunk["innovation"].to_numpy(dtype=float)
            chunk = chunk.loc[~((innov >= SENTINEL_LO) & (innov <= SENTINEL_HI))]
            if chunk.empty:
                continue

            for inst, group in chunk.groupby("instrument", sort=False):
                if inst not in stats:
                    stats[inst] = _init_stats()
                    samples[inst] = pd.DataFrame(columns=cols)
                _update_stats(stats[inst], group["innovation"].to_numpy(dtype=float),
                              group["fsoi"].to_numpy(dtype=float))
                n_take = min(len(group), max(500, max_points_per_instrument // 60))
                draw = group.sample(n=n_take,
                                    random_state=seed + file_index * 1009 + chunk_index)
                samples[inst] = _downsample(samples[inst], draw[cols],
                                            cap=max_points_per_instrument,
                                            seed=seed + file_index * 1009 + chunk_index + 17)

    sample_df = (pd.concat(samples.values(), ignore_index=True) if samples
                 else pd.DataFrame(columns=cols))
    records = []
    for inst, st in stats.items():
        n = st["n"]
        if n == 0:
            continue
        abs_fsoi = sample_df.loc[sample_df.instrument == inst, "fsoi"].abs().to_numpy(float)
        records.append({
            "instrument": inst, "n_full": n,
            "n_plotted": int((sample_df.instrument == inst).sum()),
            "innovation_mean": st["sum_innov"] / n,
            "innovation_abs_mean": st["sum_abs_innov"] / n,
            "innovation_rms": math.sqrt(st["sum_innov2"] / n),
            "fsoi_mean": st["sum_fsoi"] / n,
            "fsoi_abs_mean": st["sum_abs_fsoi"] / n,
            "fsoi_rms": math.sqrt(st["sum_fsoi2"] / n),
            "fsoi_positive_fraction": st["n_pos_fsoi"] / n,
            "sample_abs_fsoi_p95": float(np.nanpercentile(abs_fsoi, 95)) if abs_fsoi.size else np.nan,
            "sample_abs_fsoi_p99": float(np.nanpercentile(abs_fsoi, 99)) if abs_fsoi.size else np.nan,
        })
    return sample_df, pd.DataFrame(records)


def build(sample_df, stats_df, out_dir: Path, dpi: int, x_limit: float) -> list[Path]:
    instruments = [i for i in INSTRUMENT_ORDER if i in set(sample_df["instrument"])]
    ncols = 3
    nrows = int(math.ceil(len(instruments) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(WIDTH_IN, 1.20 * nrows + 1.30),
                             squeeze=False)
    hb = None
    for i, inst in enumerate(instruments):
        ax = axes[i // ncols][i % ncols]
        sub = sample_df[sample_df["instrument"] == inst].copy()
        sub = sub[(sub["innovation"] >= -x_limit) & (sub["innovation"] <= x_limit)]

        y_abs = sub["fsoi"].abs().to_numpy(dtype=float)
        ymax = float(np.nanpercentile(y_abs, 99.3)) if y_abs.size else 1.0
        if not np.isfinite(ymax) or ymax <= 0:
            ymax = float(np.nanmax(y_abs)) if y_abs.size else 1.0
        ymax = max(ymax, 1e-12)
        sub = sub[(sub["fsoi"] >= -1.15 * ymax) & (sub["fsoi"] <= 1.15 * ymax)]

        hb = ax.hexbin(sub["innovation"], sub["fsoi"], gridsize=48, bins="log",
                       mincnt=1, cmap="viridis", linewidths=0.0)
        ax.axhline(0, color="#6b6b6b", linewidth=0.5)
        ax.axvline(0, color="#6b6b6b", linewidth=0.5, linestyle="--")
        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(-1.15 * ymax, 1.15 * ymax)
        ax.grid(True, color="#e1e1e1", linewidth=0.4, alpha=0.65)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.yaxis.set_major_locator(MaxNLocator(5))

        colour = "#b2182b" if inst in CONVENTIONAL else "#2166ac"
        ax.set_title(_display(inst), fontsize=BASE_PT + 1.0, fontweight="bold",
                     color=colour, pad=16)   # clears the sample-size line
        ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
        # The offset carries a superscript, which renders at 0.7x; 9 pt keeps it
        # above AGU's 6 pt floor for superscripts.
        ax.yaxis.get_offset_text().set_size(BASE_PT + 1.0)
        ax.tick_params(labelsize=BASE_PT, pad=1.5)

        # Statistics sit above the axes. Inside the panel they masked the whole
        # negative-FSOI half of every distribution.
        row = stats_df[stats_df["instrument"] == inst].iloc[0]
        # The sample size alone, right-aligned so it clears the axis exponent at
        # the top left. Table S3 carries the RMS and the 99th percentile for
        # every source, and the caption points the reader there.
        ax.text(1.0, 1.04, f"n={int(row['n_full']):,}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=BASE_PT,
                color="#333333")

        if i % ncols == 0:
            ax.set_ylabel("Per-sample FSOI", fontsize=BASE_PT, labelpad=2)
        if i // ncols == nrows - 1:
            ax.set_xlabel("Innovation $x_a - x_b$ (normalized)", fontsize=BASE_PT + 1.0,
                          labelpad=2)

    for j in range(len(instruments), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.subplots_adjust(left=0.085, right=0.995, top=0.900, bottom=0.165,
                        hspace=0.55, wspace=0.34)

    cax = fig.add_axes([0.36, 0.080, 0.30, 0.016])
    cb = fig.colorbar(hb, cax=cax, orientation="horizontal")
    cb.set_label("Hexbin count (log scale)", fontsize=BASE_PT, labelpad=2)
    cb.ax.tick_params(labelsize=BASE_PT + 1.0, width=0.5, length=2.0, pad=1.0)
    cb.outline.set_linewidth(0.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "figureS1_innovation_jgr"
    written = []
    for path, kw in [(out_dir / f"{stem}.pdf", {}),
                     (out_dir / f"{stem}.png", {"dpi": dpi}),
                     (out_dir / f"{stem}.tif", {"dpi": dpi, "pil_kwargs": {"compression": "tiff_lzw"}})]:
        fig.savefig(path, **kw)
        written.append(path)
    plt.close(fig)
    stats_df.to_csv(out_dir / f"{stem}_source.csv", index=False)
    written.append(out_dir / f"{stem}_source.csv")
    return written


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_root", type=Path,
                   default=Path("FSOI/fsoi_outputs/seasonal_combined_maps"))
    p.add_argument("--target", default="radiosonde")
    p.add_argument("--output", type=Path, default=Path("FSOI/fsoi_outputs/paper_figures_v2"))
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--max_points_per_instrument", type=int, default=260000)
    p.add_argument("--chunksize", type=int, default=500000)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--x_limit", type=float, default=5.0)
    a = p.parse_args()

    paths = _scatter_paths(a.input_root, a.target)
    if not paths:
        raise SystemExit(f"no scatter_samples.csv under {a.input_root}")
    sample_df, stats_df = load_scored_samples(paths, a.max_points_per_instrument,
                                              a.chunksize, a.seed)
    for path in build(sample_df, stats_df, a.output, a.dpi, a.x_limit):
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
