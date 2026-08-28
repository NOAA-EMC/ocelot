#!/usr/bin/env python
"""
Plot innovation (xa - xb) diagnostics to evaluate background field quality.

FSOI is only meaningful when xb is a reasonable background. This script
visualizes two data sources:

  scatter_samples.csv   — raw innovation values (δx = xa - xb) per observation
  innovation_diagnostics.csv — per-(pair, instrument, channel) summary statistics

Produced plots:
  innovation_histograms.png     — per-instrument δx distributions vs Gaussian fit
  innovation_bias_timeseries.png — innovation_mean drift per instrument over time
  background_quality_summary.png — normalized RMSE bar chart per instrument

Scientific thresholds used:
  normalized_rmse > 20%  → background is poor for that instrument (WARN)
  |innovation_mean| / innovation_std > 0.1 → systematic bias present (WARN)
  |skewness| > 1.0  → non-Gaussian, FSOI linear approximation may be unreliable

Usage:
    python FSOI/plot_innovation_diagnostics.py \
        --scatter  path/to/csv/scatter_samples.csv \
        --diag     path/to/evaluation/innovation_diagnostics.csv \
        --output   path/to/figures/innovation/
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm as scipy_norm  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_time(bin_name: str) -> pd.Timestamp:
    name = str(bin_name)
    if name.startswith("bin") and len(name) >= 13 and name[3:13].isdigit():
        return pd.to_datetime(name[3:13], format="%Y%m%d%H")
    if "_" in name:
        date_part, time_part = name.split("_", 1)
        return pd.to_datetime(f"{date_part} {time_part[:2]}:00")
    return pd.to_datetime(name)


# ── Plot 1: Innovation histograms ─────────────────────────────────────────────

def plot_innovation_histograms(df_scatter: pd.DataFrame, out_dir: Path) -> None:
    """One panel per instrument: histogram of δx values with fitted Gaussian.

    A healthy background has δx ≈ N(0, σ).
    A biased background shows |μ/σ| > ~0.1.
    Heavy tails (|skewness| > 1) suggest nonlinear model errors.
    """
    print("Creating innovation histograms...")

    if "innovation" not in df_scatter.columns or "instrument" not in df_scatter.columns:
        print("  Skipping: missing 'innovation' or 'instrument' column")
        return

    instruments = sorted(df_scatter["instrument"].dropna().unique())
    n = len(instruments)
    if n == 0:
        return

    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                             squeeze=False)

    for i, inst in enumerate(instruments):
        ax = axes[i // ncols][i % ncols]
        sub = df_scatter[df_scatter["instrument"] == inst]["innovation"].dropna()
        if len(sub) < 10:
            ax.set_title(f"{inst}\n(insufficient data)")
            ax.axis("off")
            continue

        vals = sub.to_numpy(dtype=float)
        mu, sigma = float(np.mean(vals)), float(np.std(vals))
        skew = float(np.mean(((vals - mu) / (sigma + 1e-12)) ** 3))
        bias_ratio = abs(mu) / (sigma + 1e-12)

        # Clip extreme outliers for plotting (keep 99.5th percentile)
        p_lo, p_hi = np.percentile(vals, [0.25, 99.75])
        plot_vals = vals[(vals >= p_lo) & (vals <= p_hi)]

        ax.hist(plot_vals, bins=50, density=True, alpha=0.6, color="steelblue",
                edgecolor="none", label="Observed")

        # Fitted Gaussian
        x_fit = np.linspace(p_lo, p_hi, 300)
        ax.plot(x_fit, scipy_norm.pdf(x_fit, mu, sigma), "r-", lw=1.5,
                label=f"N({mu:.3f}, {sigma:.3f})")
        ax.axvline(0, color="k", lw=0.8, linestyle="--")
        ax.axvline(mu, color="orange", lw=1.2, linestyle="-", label=f"μ={mu:.3f}")

        color = "red" if bias_ratio > 0.10 or abs(skew) > 1.0 else "black"
        flag = " [WARN]" if color == "red" else ""
        ax.set_title(f"{inst}{flag}\nμ={mu:.3f}  σ={sigma:.3f}  skew={skew:.2f}",
                     fontsize=9, color=color)
        ax.set_xlabel("Innovation (xa - xb)", fontsize=8)
        ax.set_ylabel("Density", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.2)

    # Hide unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")

    fig.suptitle("Innovation Distributions per Instrument\n"
                 "(WARN: |μ/σ| > 10% or |skewness| > 1.0)", fontsize=11, y=1.01)
    fig.tight_layout()
    out = out_dir / "innovation_histograms.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 2: Innovation bias time series ───────────────────────────────────────

def plot_innovation_bias_timeseries(df_diag: pd.DataFrame, out_dir: Path) -> None:
    """Time series of innovation mean per instrument (per channel 1 = first channel).

    A drifting mean signals that the background field is systematically shifting
    relative to observations — this biases all FSOI values for that instrument.
    """
    print("Creating innovation bias time series...")

    required = {"curr_bin", "instrument", "channel", "innovation_mean"}
    if not required.issubset(df_diag.columns):
        missing = required - set(df_diag.columns)
        print(f"  Skipping: missing columns {missing}")
        return

    try:
        df = df_diag.copy()
        df["date"] = df["curr_bin"].apply(_parse_time)
    except Exception as e:
        print(f"  Skipping: could not parse dates — {e}")
        return

    # Plot channel 1 (temperature / primary variable) for each instrument
    ch1 = df[df["channel"] == 1]
    if ch1.empty:
        ch1 = df  # fall back to all channels

    ts = (ch1.groupby(["date", "instrument"])["innovation_mean"]
             .mean()
             .unstack("instrument")
             .sort_index())

    if ts.empty:
        print("  Skipping: empty time series")
        return

    instruments = ts.columns.tolist()
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axhline(0, color="k", lw=1.0, linestyle="--")
    ax.axhspan(-0.05, 0.05, alpha=0.08, color="green",
               label="±5% bias zone (approximately acceptable)")

    cmap = plt.get_cmap("tab20", max(len(instruments), 1))
    for i, inst in enumerate(instruments):
        ax.plot(ts.index, ts[inst], label=inst, color=cmap(i), linewidth=1.5)

    ax.set_xlabel("Date")
    ax.set_ylabel("Mean innovation (xa - xb), channel 1")
    ax.set_title("Innovation Bias Time Series per Instrument\n"
                 "(Systematic drift = biased background field)")
    ax.legend(title="Instrument", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()

    out = out_dir / "innovation_bias_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 3: Background quality summary (normalized RMSE) ─────────────────────

def plot_background_quality_summary(df_diag: pd.DataFrame, out_dir: Path) -> None:
    """Bar chart of median normalized RMSE per instrument.

    normalized_rmse = RMSE(xa - xb) / obs_range
    Threshold: < 5% = good background, > 20% = poor background.
    """
    print("Creating background quality summary...")

    required = {"instrument", "normalized_rmse"}
    if not required.issubset(df_diag.columns):
        print(f"  Skipping: missing columns {required - set(df_diag.columns)}")
        return

    summary = (df_diag.groupby("instrument")["normalized_rmse"]
                      .median()
                      .sort_values(ascending=True))
    if summary.empty:
        return

    fig, ax = plt.subplots(figsize=(10, max(4, len(summary) * 0.4)))
    colors = ["green" if v < 0.05 else ("orange" if v < 0.20 else "red")
              for v in summary.values]
    summary.plot(kind="barh", ax=ax, color=colors, alpha=0.8)

    ax.axvline(0.05, color="green", lw=1.2, linestyle="--", label="5% (good)")
    ax.axvline(0.20, color="red", lw=1.2, linestyle="--", label="20% (poor)")
    ax.set_xlabel("Median normalized RMSE  [RMSE(δx) / obs_range]")
    ax.set_title("Background Field Quality per Instrument\n"
                 "(Green < 5%: good  |  Orange < 20%: moderate  |  Red ≥ 20%: poor)")
    ax.legend()
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    out = out_dir / "background_quality_summary.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")

    # Also print a text summary
    poor = summary[summary >= 0.20]
    if not poor.empty:
        print(f"  WARNING: {len(poor)} instruments with normalized_rmse ≥ 20%: "
              f"{list(poor.index)}")
    else:
        ok_count = (summary < 0.05).sum()
        print(f"  {ok_count}/{len(summary)} instruments have normalized_rmse < 5% (good background)")


# ── Plot 4: Skewness summary ──────────────────────────────────────────────────

def plot_skewness_summary(df_diag: pd.DataFrame, out_dir: Path) -> None:
    """Heatmap of |skewness| per (instrument, channel).

    |skewness| > 1.0 indicates non-Gaussian δx, which degrades the tangent-linear
    FSOI approximation and should be flagged.
    """
    print("Creating skewness summary...")

    if "innovation_skewness" not in df_diag.columns:
        print("  Skipping: no innovation_skewness column")
        return

    pivot = (df_diag.groupby(["instrument", "channel"])["innovation_skewness"]
                    .mean()
                    .abs()
                    .unstack("channel"))
    if pivot.empty:
        return

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.6),
                                    max(4, len(pivot) * 0.4)))
    im = ax.imshow(pivot.values, cmap="YlOrRd", vmin=0, vmax=2.0, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"Ch{c}" for c in pivot.columns], fontsize=8)
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Channel")
    ax.set_ylabel("Instrument")
    ax.set_title("|Innovation skewness| per instrument × channel\n"
                 "(> 1.0: non-Gaussian — FSOI linearity assumption may break)")
    plt.colorbar(im, ax=ax, label="|skewness|")
    plt.tight_layout()

    out = out_dir / "innovation_skewness_heatmap.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot innovation diagnostics for FSOI background quality")
    parser.add_argument("--scatter", type=Path, default=None,
                        help="Path to scatter_samples.csv (innovation values)")
    parser.add_argument("--diag", type=Path, default=None,
                        help="Path to innovation_diagnostics.csv (summary stats)")
    parser.add_argument("--output", type=Path, default=Path("innovation_plots"),
                        help="Output directory for PNG files")
    args = parser.parse_args()

    out_dir = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    # Auto-detect files from a common parent if explicit paths not given
    if args.scatter is None and args.diag is None:
        print("ERROR: provide at least --scatter or --diag")
        sys.exit(1)

    df_scatter = None
    if args.scatter and args.scatter.is_file():
        df_scatter = pd.read_csv(args.scatter)
        print(f"Loaded scatter_samples: {len(df_scatter):,} rows")
    elif args.scatter:
        print(f"WARNING: {args.scatter} not found — skipping histogram plots")

    df_diag = None
    if args.diag and args.diag.is_file():
        df_diag = pd.read_csv(args.diag)
        print(f"Loaded innovation_diagnostics: {len(df_diag):,} rows")
    elif args.diag:
        print(f"WARNING: {args.diag} not found — skipping time series / quality plots")

    if df_scatter is not None and not df_scatter.empty:
        plot_innovation_histograms(df_scatter, out_dir)

    if df_diag is not None and not df_diag.empty:
        plot_innovation_bias_timeseries(df_diag, out_dir)
        plot_background_quality_summary(df_diag, out_dir)
        plot_skewness_summary(df_diag, out_dir)

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
