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
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm as scipy_norm  # noqa: E402


# Conventional preprocessing fills missing channel values with SENT=-9.0 in
# normalised space.  The sentinel-contaminated innovations span a range because
# the background x_b is not always exactly 0: innovation = -9 - x_b.
# Empirically observed range in surface_obs CSVs: p1 ≈ -8.3, so the upper
# bound must be raised to -7.0 to capture all leaking sentinel rows.
SENTINEL_INNOVATION_LO = -12.0
SENTINEL_INNOVATION_HI = -7.0   # widened from -8.5 to catch surface_obs leakage

# Legacy alias.
SENTINEL_INNOVATION = -9.0
SENTINEL_ATOL = 0.25


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_time(bin_name: str) -> pd.Timestamp:
    name = str(bin_name)
    if name.startswith("bin") and len(name) >= 13 and name[3:13].isdigit():
        return pd.to_datetime(name[3:13], format="%Y%m%d%H")
    if "_" in name:
        date_part, time_part = name.split("_", 1)
        return pd.to_datetime(f"{date_part} {time_part[:2]}:00")
    return pd.to_datetime(name)


def _drop_sentinel_innovations(values: np.ndarray) -> np.ndarray:
    """Remove sentinel-coded innovations before plotting.

    The sentinel cluster observed in real scatter CSVs spans [-12, -8.5];
    the entire range is masked rather than a single point.
    """
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    mask = np.isfinite(arr) & ~((arr >= SENTINEL_INNOVATION_LO) & (arr <= SENTINEL_INNOVATION_HI))
    return arr[mask]


def _filter_sentinel_scatter(df_scatter: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Drop sentinel-coded innovation rows from the raw scatter dataframe."""
    if df_scatter.empty or "innovation" not in df_scatter.columns:
        return df_scatter, 0

    innovation = pd.to_numeric(df_scatter["innovation"], errors="coerce")
    inn_arr = innovation.to_numpy(dtype=float)
    keep_mask = innovation.notna() & ~((inn_arr >= SENTINEL_INNOVATION_LO) & (inn_arr <= SENTINEL_INNOVATION_HI))
    removed = int((~keep_mask).sum())
    if removed == 0:
        return df_scatter, 0
    return df_scatter.loc[keep_mask].copy(), removed


def _slugify_instrument(name: str) -> str:
    """Create a filesystem-safe suffix from instrument name."""
    slug = re.sub(r"[^A-Za-z0-9]+", "_", str(name).strip()).strip("_").lower()
    return slug or "unknown"


# Variable names for conventional obs instruments (1-based channel numbers as used in CSVs).
# For satellite instruments the fallback "Ch{N}" is used because channel numbers carry
# physical meaning (spectral channel index).
_CONVENTIONAL_VAR_NAMES: dict[str, dict[int, str]] = {
    "aircraft": {
        1: "temperature",
        2: "u_wind",
        3: "v_wind",
    },
    "radiosonde": {
        1: "temperature",
        2: "dewpoint_temperature",
        3: "u_wind",
        4: "v_wind",
    },
    "surface_obs": {
        1: "surface_pressure",
        2: "temperature",
        3: "dewpoint_temperature",
        4: "u_wind",
        5: "v_wind",
    },
}


def _channel_label(inst: str, ch: int) -> str:
    """Human-readable label for a (instrument, 1-based-channel) pair.

    Conventional obs instruments return the physical variable name
    (e.g. 'temperature').  Satellite instruments return 'Ch{N}' because
    channel numbers are the conventional scientific label there.
    """
    var_map = _CONVENTIONAL_VAR_NAMES.get(str(inst).lower(), {})
    return var_map.get(int(ch), f"Ch{int(ch)}")


# ── Plot 1: Innovation histograms ─────────────────────────────────────────────

def plot_innovation_histograms(df_scatter: pd.DataFrame, out_dir: Path) -> None:
    """One figure per instrument; panels are channel histograms with Gaussian fit.

    A healthy background has δx ≈ N(0, σ).
    A biased background shows |μ/σ| > ~0.1.
    Heavy tails (|skewness| > 1) suggest nonlinear model errors.
    """
    print("Creating innovation histograms...")

    if "innovation" not in df_scatter.columns or "instrument" not in df_scatter.columns:
        print("  Skipping: missing 'innovation' or 'instrument' column")
        return

    df_scatter, removed = _filter_sentinel_scatter(df_scatter)
    if removed > 0:
        print(f"  Filtered {removed:,} sentinel-coded innovation rows near -9 before plotting")

    if "channel" not in df_scatter.columns:
        print("  Skipping: missing 'channel' column for per-channel histograms")
        return

    instruments = sorted(df_scatter["instrument"].dropna().astype(str).unique().tolist())
    if not instruments:
        return

    for inst in instruments:
        inst_df = df_scatter[df_scatter["instrument"].astype(str) == inst]
        grouped = list(inst_df.groupby("channel", dropna=True))
        n = len(grouped)
        if n == 0:
            continue

        ncols = min(4, n)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows),
                                 squeeze=False)

        for i, (ch, group_df) in enumerate(grouped):
            ax = axes[i // ncols][i % ncols]
            var_name = _channel_label(inst, ch)
            title = f"{inst} / {var_name}"

            sub = group_df["innovation"].dropna().to_numpy(dtype=float)
            sub = _drop_sentinel_innovations(sub)
            if len(sub) < 10:
                ax.set_title(f"{title}\n(insufficient data)")
                ax.axis("off")
                continue

            vals = sub
            mu, sigma = float(np.mean(vals)), float(np.std(vals))
            skew = float(np.mean(((vals - mu) / (sigma + 1e-12)) ** 3))
            bias_ratio = abs(mu) / (sigma + 1e-12)

            # Clip extreme outliers for plotting (keep 99.5th percentile)
            p_lo, p_hi = np.percentile(vals, [0.25, 99.75])
            plot_vals = vals[(vals >= p_lo) & (vals <= p_hi)]

            ax.hist(plot_vals, bins=50, density=True, alpha=0.6, color="steelblue",
                    edgecolor="none", label="Observed (sentinel-masked)")
            # Fitted Gaussian
            x_fit = np.linspace(p_lo, p_hi, 300)
            ax.plot(x_fit, scipy_norm.pdf(x_fit, mu, sigma), "r-", lw=1.5,
                    label=f"N({mu:.3f}, {sigma:.3f})")
            ax.axvline(0, color="k", lw=0.8, linestyle="--")
            ax.axvline(mu, color="orange", lw=1.2, linestyle="-", label=f"μ={mu:.3f}")

            color = "red" if bias_ratio > 0.10 or abs(skew) > 1.0 else "black"
            flag = " [WARN]" if color == "red" else ""
            ax.set_title(f"{title}{flag}\nμ={mu:.3f}  σ={sigma:.3f}  skew={skew:.2f}",
                         fontsize=9, color=color)
            ax.set_xlabel("Innovation (xa - xb)", fontsize=8)
            ax.set_ylabel("Density", fontsize=8)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.2)

        # Hide unused panels
        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(f"Innovation Distributions: {inst}\n"
                     "(sentinel range [-12, -8.5] masked; WARN: |μ/σ| > 10% or |skewness| > 1.0)",
                     fontsize=11, y=1.01)
        fig.tight_layout()

        suffix = _slugify_instrument(inst)
        out = out_dir / f"innovation_histograms_{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        if len(instruments) == 1 or suffix == "aircraft":
            # Keep legacy filename for compatibility with existing paths.
            out_legacy = out_dir / "innovation_histograms.png"
            fig.savefig(out_legacy, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out_legacy}")
        plt.close(fig)


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

    for inst, inst_df in df.groupby("instrument", dropna=True):
        ts = (inst_df.groupby(["date", "channel"])["innovation_mean"]
                    .mean()
                    .unstack("channel")
                    .sort_index())

        if ts.empty:
            continue

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axhline(0, color="k", lw=1.0, linestyle="--")
        ax.axhspan(-0.05, 0.05, alpha=0.08, color="green",
                   label="±5% bias zone (approximately acceptable)")

        cmap = plt.get_cmap("tab20", max(len(ts.columns), 1))
        for i, ch in enumerate(ts.columns):
            ax.plot(ts.index, ts[ch], label=_channel_label(str(inst), ch), color=cmap(i), linewidth=1.5)

        ax.set_xlabel("Date")
        ax.set_ylabel("Mean innovation (xa - xb)")
        ax.set_title(f"Innovation Bias Time Series: {inst}\n"
                     "(Systematic drift = biased background field)")
        ax.legend(title="Channel", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()

        suffix = _slugify_instrument(str(inst))
        out = out_dir / f"innovation_bias_timeseries_{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        if df["instrument"].nunique(dropna=True) == 1 or suffix == "aircraft":
            out_legacy = out_dir / "innovation_bias_timeseries.png"
            fig.savefig(out_legacy, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out_legacy}")
        plt.close(fig)


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

    for inst, inst_df in df_diag.groupby("instrument", dropna=True):
        summary = (inst_df.groupby("channel")["normalized_rmse"]
                          .median()
                          .sort_values(ascending=True))
        if summary.empty:
            continue

        fig, ax = plt.subplots(figsize=(10, max(4, len(summary) * 0.4)))
        colors = ["green" if v < 0.05 else ("orange" if v < 0.20 else "red")
                  for v in summary.values]
        summary.index = [_channel_label(str(inst), c) for c in summary.index]
        summary.plot(kind="barh", ax=ax, color=colors, alpha=0.8)

        ax.axvline(0.05, color="green", lw=1.2, linestyle="--", label="5% (good)")
        ax.axvline(0.20, color="red", lw=1.2, linestyle="--", label="20% (poor)")
        ax.set_xlabel("Median normalized RMSE  [RMSE(δx) / obs_range]")
        ax.set_title(f"Background Field Quality: {inst} (per channel)\n"
                     "(Green < 5%: good  |  Orange < 20%: moderate  |  Red >= 20%: poor)")
        ax.legend()
        ax.grid(True, alpha=0.2, axis="x")

        plt.tight_layout()
        suffix = _slugify_instrument(str(inst))
        out = out_dir / f"background_quality_summary_{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        if df_diag["instrument"].nunique(dropna=True) == 1 or suffix == "aircraft":
            out_legacy = out_dir / "background_quality_summary.png"
            fig.savefig(out_legacy, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out_legacy}")
        plt.close(fig)

        poor = summary[summary >= 0.20]
        if not poor.empty:
            print(f"  WARNING: {inst} has {len(poor)} channels with normalized_rmse >= 20%: "
                  f"{list(poor.index)}")
        else:
            ok_count = (summary < 0.05).sum()
            print(f"  {inst}: {ok_count}/{len(summary)} channels have normalized_rmse < 5% (good background)")


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

    for inst, inst_df in df_diag.groupby("instrument", dropna=True):
        summary = (inst_df.groupby("channel")["innovation_skewness"]
                          .mean()
                          .abs()
                          .sort_index())
        if summary.empty:
            continue

        fig, ax = plt.subplots(figsize=(max(8, len(summary) * 0.7), 4.5))
        bars = ax.bar([_channel_label(str(inst), c) for c in summary.index], summary.values,
                      color=["red" if v > 1.0 else "goldenrod" for v in summary.values],
                      alpha=0.85)
        ax.axhline(1.0, color="k", linestyle="--", linewidth=1.0, label="|skewness| = 1.0")
        ax.set_xlabel("Channel")
        ax.set_ylabel("Mean |skewness|")
        ax.set_title(f"Innovation Skewness Summary: {inst}\n"
                     "(> 1.0: non-Gaussian — FSOI linearity assumption may break)")
        ax.grid(True, alpha=0.2, axis="y")
        ax.legend()

        for bar, val in zip(bars, summary.values):
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f"{val:.2f}", ha="center", va="bottom", fontsize=7)

        plt.tight_layout()
        suffix = _slugify_instrument(str(inst))
        out = out_dir / f"innovation_skewness_heatmap_{suffix}.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out}")
        if df_diag["instrument"].nunique(dropna=True) == 1 or suffix == "aircraft":
            out_legacy = out_dir / "innovation_skewness_heatmap.png"
            fig.savefig(out_legacy, dpi=150, bbox_inches="tight")
            print(f"  Saved: {out_legacy}")
        plt.close(fig)


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
        df_scatter, removed = _filter_sentinel_scatter(df_scatter)
        if removed > 0:
            print(f"Filtered scatter_samples: removed {removed:,} sentinel-coded innovation rows near -9")
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
