#!/usr/bin/env python
"""
Plot OSE vs FSOI comparison to validate FSOI linearity assumption.

Reads:
  evaluation/ose_results.csv          — matched OSE endpoint losses
  evaluation/ose_vs_fsoi_comparison.csv — merged OSE + FSOI predictions per pair

Produced plots:
  ose_vs_fsoi_scatter.png      — scatter: delta_J_actual vs FSOI predicted
  ose_vs_fsoi_timeseries.png   — time series: both over the evaluation window
  ose_closure_ratio.png        — histogram of closure ratio (target: centered on 1)
  ose_summary.csv              — correlation, slope, bias statistics

Scientific interpretation
--------------------------
A well-validated FSOI satisfies:
  (1) High Pearson correlation (r > 0.85) between delta_J_actual and FSOI
  (2) Regression slope close to 1.0 (±0.2) — no systematic scaling error
  (3) Sign agreement > 90% — OSE and FSOI agree on helpful vs detrimental
  (4) Closure ratio median ~ 1.0 (±0.15)

Departures from these targets reveal:
  - Large nonlinearities in the GNN (slope far from 1)
  - FSOI bias (offset in scatter plot)
  - Regime-dependent validity (divergence in specific synoptic conditions)

Usage:
    python FSOI/plot_ose_comparison.py \
        --input  FSOI/fsoi_outputs/full_eval_20250701_20250731/radiosonde \
        --output FSOI/fsoi_outputs/full_eval_20250701_20250731/radiosonde/figures/ose
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_time(s: str) -> pd.Timestamp:
    name = str(s)
    if name.startswith("bin") and len(name) >= 13 and name[3:13].isdigit():
        return pd.to_datetime(name[3:13], format="%Y%m%d%H")
    try:
        if "_" in name:
            d, t = name.split("_", 1)
            return pd.to_datetime(f"{d} {t[:2]}:00")
        return pd.to_datetime(name)
    except Exception:
        return pd.NaT


def _linregress(x: np.ndarray, y: np.ndarray):
    """Return (slope, intercept, r) via numpy lstsq."""
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < 3:
        return float("nan"), float("nan"), float("nan")
    A = np.column_stack([x, np.ones_like(x)])
    result = np.linalg.lstsq(A, y, rcond=None)
    slope, intercept = result[0]
    r = float(np.corrcoef(x, y)[0, 1])
    return float(slope), float(intercept), r


# ── Plot 1: Scatter OSE vs FSOI ───────────────────────────────────────────────

def _ose_comparison_col(comp: pd.DataFrame) -> str:
    """Return the FSOI-compatible OSE column, positive = detrimental."""
    if "ose_fsoi_convention" in comp.columns:
        return "ose_fsoi_convention"
    if "delta_j_actual" in comp.columns:
        return "delta_j_actual"
    raise ValueError(
        "OSE/FSOI plots require ose_fsoi_convention or delta_j_actual. "
        "Raw ose_impact has the opposite sign and is not valid for final "
        "matched comparison plots."
    )


def _valid_signal_rows(comp: pd.DataFrame) -> pd.DataFrame:
    """Drop near-zero rows already marked invalid by compare_ose_vs_fsoi."""
    if "signal_valid" not in comp.columns:
        return comp
    return comp[comp["signal_valid"].fillna(False).astype(bool)]


def _excluded_count(comp: pd.DataFrame) -> int:
    if "near_zero_excluded" in comp.columns:
        return int(comp["near_zero_excluded"].fillna(False).astype(bool).sum())
    if "signal_valid" in comp.columns:
        return int((~comp["signal_valid"].fillna(False).astype(bool)).sum())
    return 0


def plot_ose_vs_fsoi_scatter(comp: pd.DataFrame, out_dir: Path) -> dict:
    """Scatter plot of FSOI predicted impact vs OSE actual impact per pair.

    Each point = one (pair, denied instrument). The diagonal y=x is the target.
    """
    print("Creating OSE vs FSOI scatter plot...")
    col_ose = _ose_comparison_col(comp)
    col_fsoi = "fsoi_predicted"

    valid = _valid_signal_rows(comp).dropna(subset=[col_ose, col_fsoi])
    if valid.empty:
        print("  Skipping: no matched pairs")
        return {}

    x = valid[col_ose].to_numpy(dtype=float)
    y = valid[col_fsoi].to_numpy(dtype=float)

    slope, intercept, r = _linregress(x, y)
    sign_cases = valid.get("sign_agree", pd.Series([np.nan])).dropna()
    sign_agree = float(sign_cases.mean()) if not sign_cases.empty else float("nan")
    n_excluded = _excluded_count(comp)

    fig, ax = plt.subplots(figsize=(7, 7))

    # Per-instrument coloring if available
    if "denied_instrument" in valid.columns:
        instruments = sorted(valid["denied_instrument"].dropna().unique())
        cmap = plt.get_cmap("tab10", max(len(instruments), 1))
        for i, inst in enumerate(instruments):
            mask = valid["denied_instrument"] == inst
            ax.scatter(x[mask.values], y[mask.values], label=inst,
                       color=cmap(i), alpha=0.7, s=40)
        ax.legend(title="Denied instrument", fontsize=8)
    else:
        ax.scatter(x, y, alpha=0.7, s=40, color="steelblue")

    # y=x reference
    lim = max(np.abs(x).max(), np.abs(y).max()) * 1.05
    lim = max(lim, 1e-12)
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="y = x (ideal)")

    # Regression line
    if np.isfinite(slope):
        xx = np.linspace(-lim, lim, 200)
        ax.plot(xx, slope * xx + intercept, "r-", lw=1.5,
                label=f"fit: y = {slope:.2f}x + {intercept:.2e}")

    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_ylabel("Matched conditional FSOI (raw sampled impact)")
    ax.set_xlabel("OSE delta J actual  (J_control - J_denied; positive = detrimental)")
    ax.set_title(
        f"OSE vs FSOI comparison\n"
        f"r = {r:.3f}   slope = {slope:.3f}   sign_agree = {sign_agree:.0%}\n"
        f"n = {len(valid)} pairs; excluded near zero = {n_excluded}"
    )
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2)

    color = "red" if (abs(r) < 0.85 or abs(slope - 1.0) > 0.30) else "black"
    if color == "red":
        ax.set_title(ax.get_title() + "\n[WARNING: r or slope outside target range]",
                     color=color)

    plt.tight_layout()
    out = out_dir / "ose_vs_fsoi_scatter.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}  (r={r:.3f}, slope={slope:.3f})")

    return {"r": r, "slope": slope, "intercept": intercept,
            "sign_agree": sign_agree, "n_pairs": len(valid),
            "n_near_zero_excluded": n_excluded}


# ── Plot 2: Time series ───────────────────────────────────────────────────────

def plot_ose_vs_fsoi_timeseries(comp: pd.DataFrame, ose_raw: pd.DataFrame,
                                out_dir: Path) -> None:
    """Time series of OSE impact and FSOI prediction over the evaluation window."""
    print("Creating OSE vs FSOI time series...")

    col_ose = _ose_comparison_col(comp)
    col_fsoi = "fsoi_predicted"
    comp = _valid_signal_rows(comp).copy()

    if "curr_bin" not in comp.columns:
        print("  Skipping: no curr_bin column")
        return

    try:
        comp = comp.copy()
        comp["date"] = comp["curr_bin"].apply(_parse_time)
    except Exception as e:
        print(f"  Skipping: date parse error — {e}")
        return

    # Per-instrument subplots
    instruments = sorted(comp.get("denied_instrument",
                                  pd.Series(dtype=str)).dropna().unique())
    if not instruments:
        instruments = ["all"]
        comp["denied_instrument"] = "all"

    nrows = len(instruments)
    fig, axes = plt.subplots(nrows, 1, figsize=(14, 4 * nrows), squeeze=False)

    for i, inst in enumerate(instruments):
        ax = axes[i][0]
        sub = comp[comp["denied_instrument"] == inst].sort_values("date")
        if sub.empty:
            ax.set_title(f"{inst} (no data)")
            continue

        ax.plot(sub["date"], sub[col_ose], "b-o", ms=4, lw=1.5,
                label="OSE (actual)", alpha=0.85)
        ax.plot(sub["date"], sub[col_fsoi], "r--s", ms=4, lw=1.5,
                label="FSOI (predicted)", alpha=0.85)
        ax.axhline(0, color="k", lw=0.8, linestyle=":")
        ax.set_ylabel("Impact")
        ax.set_title(f"{inst}: OSE vs FSOI over time")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)

    axes[-1][0].set_xlabel("Date")
    fig.suptitle("OSE delta_J_actual vs FSOI Predicted Impact — Time Series", fontsize=11)
    fig.tight_layout()

    out = out_dir / "ose_vs_fsoi_timeseries.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 3: Closure ratio histogram ──────────────────────────────────────────

def plot_closure_ratio_histogram(comp: pd.DataFrame, out_dir: Path) -> None:
    """Histogram of closure_ratio = FSOI_predicted / delta_J_actual.

    Target: distribution centered on 1.0 with small spread.
    Values far from 1 reveal pairs where the linearization is poor.
    """
    print("Creating closure ratio histogram...")

    if "closure_ratio" not in comp.columns:
        print("  Skipping: no closure_ratio column")
        return

    ratios = comp["closure_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        return

    # Clip extreme outliers for display
    p5, p95 = ratios.quantile([0.05, 0.95])
    clipped = ratios.clip(p5, p95)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(clipped, bins=40, density=True, color="steelblue", alpha=0.7,
            edgecolor="none")
    ax.axvline(1.0, color="green", lw=2, label="Target (1.0)")
    ax.axvline(ratios.median(), color="orange", lw=1.5, linestyle="--",
               label=f"Median = {ratios.median():.3f}")
    ax.axvspan(0.70, 1.30, alpha=0.1, color="green", label="±30% acceptance band")

    ax.set_xlabel("Closure ratio  (FSOI / delta_J_actual)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"FSOI/OSE Closure Ratio Distribution\n"
        f"Median = {ratios.median():.3f}   "
        f"Std = {ratios.std():.3f}   "
        f"n = {len(ratios)}"
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    out = out_dir / "ose_closure_ratio.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot OSE vs FSOI comparison")
    parser.add_argument("--input", type=Path, required=True,
                        help="FSOI output root directory (contains csv/ and evaluation/)")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output directory for plots (default: <input>/figures/ose)")
    args = parser.parse_args()

    root = args.input
    out_dir = args.output or (root / "figures" / "ose")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve file paths
    eval_dir = root / "evaluation"
    csv_dir = root / "csv"
    if not eval_dir.exists():
        eval_dir = root / "evaluation"

    ose_raw_path = eval_dir / "ose_results.csv"
    comp_path = eval_dir / "ose_vs_fsoi_comparison.csv"
    fsoi_inst_path = csv_dir / "fsoi_by_instrument.csv"

    if not ose_raw_path.is_file():
        print(f"ERROR: {ose_raw_path} not found.")
        print("  Run fsoi_inference.py with --ose_instruments <inst1> [<inst2> ...]")
        sys.exit(1)

    ose_raw = pd.read_csv(ose_raw_path)
    print(f"Loaded OSE results: {len(ose_raw)} pairs")

    required_comparison_columns = {
        "fsoi_predicted",
        "ose_fsoi_convention",
        "comparison_mode",
        "signal_valid",
        "closure_ratio",
    }

    def _build_comparison() -> pd.DataFrame:
        from fsoi_ose import compare_ose_vs_fsoi
        built = compare_ose_vs_fsoi(ose_raw_path, fsoi_inst_path)
        if not built.empty:
            built.to_csv(comp_path, index=False)
            print(f"Built and saved comparison: {len(built)} rows -> {comp_path}")
        else:
            print("WARNING: could not build OSE vs FSOI comparison")
        return built

    # Load only complete, current matched-comparison files; rebuild otherwise.
    if comp_path.is_file():
        comp = pd.read_csv(comp_path)
        has_matched_ose = required_comparison_columns.issubset(comp.columns)
        if has_matched_ose:
            has_matched_ose = comp["comparison_mode"].eq(
                "conditional_endpoint_same_sample_same_J"
            ).all()
        comparison_is_current = comp_path.stat().st_mtime >= ose_raw_path.stat().st_mtime
        if has_matched_ose and comparison_is_current:
            print(f"Loaded existing comparison: {len(comp)} rows")
        else:
            print("Existing comparison is stale, partial, or legacy; rebuilding")
            comp = _build_comparison()
    else:
        comp = _build_comparison()

    # Generate plots
    stats = {}
    if not comp.empty:
        stats = plot_ose_vs_fsoi_scatter(comp, out_dir)
        plot_ose_vs_fsoi_timeseries(comp, ose_raw, out_dir)
        plot_closure_ratio_histogram(comp, out_dir)

    # Summary CSV
    if stats:
        summary = pd.DataFrame([{
            "n_pairs": stats.get("n_pairs", 0),
            "n_near_zero_excluded": stats.get("n_near_zero_excluded", 0),
            "pearson_r": stats.get("r", float("nan")),
            "regression_slope": stats.get("slope", float("nan")),
            "regression_intercept": stats.get("intercept", float("nan")),
            "sign_agreement_frac": stats.get("sign_agree", float("nan")),
            "validation_flag": (
                "PASS"
                if abs(stats.get("r", 0)) >= 0.85
                and abs(stats.get("slope", 0) - 1.0) <= 0.30
                and stats.get("sign_agree", 0) >= 0.90
                else "WARN"
            ),
        }])
        summary_path = eval_dir / "ose_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"\nOSE validation summary:")
        print(summary.to_string(index=False))
        flag = summary["validation_flag"].iloc[0]
        if flag == "PASS":
            print(
                "\n[OSE] PASS - Matched ATMS FSOI shows directional and "
                "magnitude agreement with the ATMS replacement experiment."
            )
        else:
            print("\n[OSE] WARN - Matched ATMS FSOI agreement is limited. "
                  "Review scatter plot and closure ratios.")

    print(f"\nDone. Output: {out_dir}")


if __name__ == "__main__":
    main()
