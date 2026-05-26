"""
Evaluate FSOI CSV outputs with pair-wise summaries and bootstrap intervals.

This script is intentionally lightweight: it reads fsoi_by_instrument.csv,
prefers sum_impact_scaled when present, and writes summary CSVs that are safer
for comparing instruments across date windows.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _resolve_csv_dir(path: Path) -> Path:
    if (path / "fsoi_by_instrument.csv").is_file():
        return path
    if (path / "csv" / "fsoi_by_instrument.csv").is_file():
        return path / "csv"
    raise FileNotFoundError(
        f"Could not find fsoi_by_instrument.csv in {path} or {path / 'csv'}"
    )


def _impact_col(df: pd.DataFrame) -> str:
    return "sum_impact_scaled" if "sum_impact_scaled" in df.columns else "sum_impact"


def _count_col(df: pd.DataFrame, impact_col: str) -> str:
    if impact_col == "sum_impact_scaled" and "raw_n_observations" in df.columns:
        return "raw_n_observations"
    if "n_observations" in df.columns:
        return "n_observations"
    return ""


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int, seed: int) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    if values.size == 1 or n_boot <= 0:
        return values.mean(), values.mean()

    rng = np.random.default_rng(seed)
    boot = rng.choice(values, size=(n_boot, values.size), replace=True).mean(axis=1)
    return tuple(np.percentile(boot, [2.5, 97.5]))


def _closure_group_columns(df: pd.DataFrame) -> list[str]:
    """Columns that define one forecast-error metric for closure checks."""
    cols = ["pair_idx"]
    for col in [
        "prev_bin",
        "curr_bin",
        "lead_step",
        "p_idx",
        "p_hpa",
        "target_variable",
        "target_channel",
    ]:
        if col in df.columns:
            cols.append(col)
    return cols


def _closure_quality_flag(
    sign_frac: float,
    median_ratio: float,
    n_cases: int,
    min_cases: int = 3,
    relative_signal: float = 1.0,
    snr: float = 1.0,
    is_global: bool = True,
) -> str:
    """Assign PASS / WARN / FAIL / LOW_SIGNAL to a closure summary row.

    Global row (is_global=True) — strict criteria, signal is large (~15% of ea):
      - sign_agreement_frac >= 0.90
      - |median_closure_ratio - 1| <= 0.15

    Per-level rows (is_global=False) — signal is 0.1-0.5% of ea per stratum:
      LOW_SIGNAL : relative_signal < 0.003 OR snr < 0.7
          The denominator ea_p - eb_p is too close to zero for closure_ratio to
          mean anything.  Do not interpret as a GNN quality failure.
      PASS       : sign_agree >= 0.65  (significantly above random at n~60)
      WARN       : sign_agree >= 0.55
      FAIL       : sign_agree <  0.55  (indistinguishable from random)

    Closure ratio is intentionally excluded from per-level criteria because
    dividing by a near-zero denominator makes it numerically unstable.
    """
    if n_cases < min_cases:
        return "INSUF"
    if np.isnan(sign_frac):
        return "UNKNOWN"

    if is_global:
        if np.isnan(median_ratio):
            return "UNKNOWN"
        ratio_ok = abs(median_ratio - 1.0) <= 0.15
        sign_ok = sign_frac >= 0.90
        if ratio_ok and sign_ok:
            return "PASS"
        if ratio_ok or sign_ok:
            return "WARN"
        return "FAIL"
    else:
        # Per-level: check signal strength first
        if relative_signal < 0.003 or snr < 0.7:
            return "LOW_SIGNAL"
        if sign_frac >= 0.65:
            return "PASS"
        if sign_frac >= 0.55:
            return "WARN"
        return "FAIL"


def _closure_summary_records(
    closure: pd.DataFrame,
    group_cols: list[str],
    eps: float,
) -> list[dict]:
    """Summarize closure quality overall and by (variable, pressure level)."""

    def _one_record(label: dict, g: pd.DataFrame, is_global: bool = False) -> dict:
        valid_sign = (
            g["sum_fsoi"].abs().gt(eps)
            & g["ea_minus_eb"].abs().gt(eps)
            & g["sign_agree"].notna()
        )
        valid_ratio = g["closure_ratio"].replace([np.inf, -np.inf], np.nan).dropna()
        valid_rel = g["relative_abs_closure_error"].replace([np.inf, -np.inf], np.nan).dropna()
        sign_frac = float(g.loc[valid_sign, "sign_agree"].mean()) if valid_sign.any() else np.nan
        med_ratio = float(valid_ratio.median()) if not valid_ratio.empty else np.nan

        # Signal-to-noise: how large is the per-stratum error change relative to
        # the total error at that stratum, and how stable is it across pairs?
        mean_ea = float(g["ea"].mean()) if "ea" in g.columns else np.nan
        mean_ea_minus_eb = float(g["ea_minus_eb"].mean())
        std_ea_minus_eb = float(g["ea_minus_eb"].std()) if len(g) > 1 else np.nan
        relative_signal = (abs(mean_ea_minus_eb) / abs(mean_ea)) if (
            np.isfinite(mean_ea) and abs(mean_ea) > eps) else np.nan
        snr = (abs(mean_ea_minus_eb) / std_ea_minus_eb) if (
            np.isfinite(std_ea_minus_eb) and std_ea_minus_eb > eps) else np.nan

        rec = {
            **label,
            "error_source": ",".join(sorted(str(v) for v in g["error_source"].dropna().unique())),
            "n_cases": int(len(g)),
            "n_sign_cases": int(valid_sign.sum()),
            "sign_agreement_frac": sign_frac,
            "median_closure_ratio": med_ratio,
            "closure_ratio_p05": float(valid_ratio.quantile(0.05)) if not valid_ratio.empty else np.nan,
            "closure_ratio_p95": float(valid_ratio.quantile(0.95)) if not valid_ratio.empty else np.nan,
            "median_abs_closure_error": float(g["abs_closure_error"].median()),
            "median_relative_abs_closure_error": float(valid_rel.median()) if not valid_rel.empty else np.nan,
            "mean_sum_fsoi": float(g["sum_fsoi"].mean()),
            "mean_ea_minus_eb": mean_ea_minus_eb,
            "relative_signal": float(relative_signal) if np.isfinite(relative_signal) else np.nan,
            "signal_snr": float(snr) if np.isfinite(snr) else np.nan,
            "quality_flag": _closure_quality_flag(
                sign_frac, med_ratio, int(len(g)),
                relative_signal=float(relative_signal) if np.isfinite(relative_signal) else 1.0,
                snr=float(snr) if np.isfinite(snr) else 1.0,
                is_global=is_global,
            ),
        }
        return rec

    records = [_one_record({"scope": "all"}, closure, is_global=True)]
    if group_cols:
        for keys, g in closure.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            label = {"scope": "by_" + "_".join(group_cols)}
            label.update(dict(zip(group_cols, keys)))
            records.append(_one_record(label, g, is_global=False))
    return records


def write_closure_diagnostics(
    df: pd.DataFrame,
    output_dir: Path,
    impact_col: str,
    eps: float = 1e-12,
) -> pd.DataFrame | None:
    """Write FSOI closure diagnostics.

    For each pair/pressure/target-variable metric, total FSOI should
    approximate ea - eb:

        closure_error = sum_fsoi - (ea - eb)
        closure_ratio = sum_fsoi / (ea - eb)
    """
    ea_col = "ea_p" if "ea_p" in df.columns else "ea"
    eb_col = "eb_p" if "eb_p" in df.columns else "eb"

    required = {"pair_idx", ea_col, eb_col, impact_col}
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"[SKIPPED] Closure diagnostics: missing columns {missing}")
        return None
    if (ea_col, eb_col) == ("ea", "eb") and {"p_hpa", "target_variable"}.issubset(df.columns):
        print(
            "[WARNING] Closure diagnostics are using ea/eb columns. "
            "For pressure/variable-stratified legacy CSVs these may be total "
            "step errors rather than per-metric errors. New FSOI runs write "
            "ea_p/eb_p and should be used for the trust check."
        )

    group_cols = _closure_group_columns(df)
    agg_spec = {
        "sum_fsoi": (impact_col, "sum"),
        "sum_fsoi_raw": ("sum_impact", "sum"),
        "ea": (ea_col, "first"),
        "eb": (eb_col, "first"),
        "n_rows": ("instrument", "size"),
        "n_instruments": ("instrument", "nunique"),
    }
    closure = df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    closure["ea_minus_eb"] = closure["ea"] - closure["eb"]
    closure["error_source"] = f"{ea_col}/{eb_col}"
    closure["closure_error"] = closure["sum_fsoi"] - closure["ea_minus_eb"]
    closure["abs_closure_error"] = closure["closure_error"].abs()
    closure["closure_ratio"] = np.where(
        closure["ea_minus_eb"].abs() > eps,
        closure["sum_fsoi"] / closure["ea_minus_eb"],
        np.nan,
    )
    closure["relative_abs_closure_error"] = np.where(
        closure["ea_minus_eb"].abs() > eps,
        closure["abs_closure_error"] / closure["ea_minus_eb"].abs(),
        np.nan,
    )
    valid_sign = closure["sum_fsoi"].abs().gt(eps) & closure["ea_minus_eb"].abs().gt(eps)
    closure["sign_agree"] = np.where(
        valid_sign,
        np.sign(closure["sum_fsoi"]) == np.sign(closure["ea_minus_eb"]),
        np.nan,
    )

    detail_path = output_dir / "fsoi_closure_diagnostics.csv"
    closure.to_csv(detail_path, index=False)

    summary_group_cols = [
        col for col in ["target_variable", "p_hpa"] if col in closure.columns
    ]
    closure_summary = pd.DataFrame(
        _closure_summary_records(closure, summary_group_cols, eps=eps)
    )
    summary_path = output_dir / "fsoi_closure_summary.csv"
    closure_summary.to_csv(summary_path, index=False)

    # Write the per-(variable, pressure) rows as a dedicated file for easy inspection
    per_level_rows = closure_summary[closure_summary["scope"] != "all"]
    if not per_level_rows.empty:
        per_level_path = output_dir / "fsoi_closure_per_level_summary.csv"
        per_level_rows.to_csv(per_level_path, index=False)
        print(f"Wrote: {per_level_path}")

    print(f"Wrote: {detail_path}")
    print(f"Wrote: {summary_path}")

    # ── Global closure row ────────────────────────────────────────────────
    global_row = closure_summary[closure_summary["scope"] == "all"]
    print("\nClosure summary (global):")
    display_cols = [
        c for c in [
            "scope", "error_source", "n_cases", "sign_agreement_frac",
            "median_closure_ratio", "median_relative_abs_closure_error", "quality_flag",
        ] if c in global_row.columns
    ]
    print(global_row[display_cols].to_string(index=False))

    # ── Per-(variable, pressure) closure table ────────────────────────────
    if not per_level_rows.empty:
        display_per = [
            c for c in [
                "target_variable", "p_hpa", "n_cases",
                "sign_agreement_frac", "median_closure_ratio",
                "relative_signal", "signal_snr", "quality_flag",
            ] if c in per_level_rows.columns
        ]
        print(f"\nPer-(variable, pressure) closure [{len(per_level_rows)} cells]:")
        print(per_level_rows.sort_values(
            ["target_variable", "p_hpa"] if "p_hpa" in per_level_rows.columns else ["target_variable"],
        )[display_per].to_string(index=False))

        flag_counts = per_level_rows["quality_flag"].value_counts().to_dict()
        n_low = flag_counts.get("LOW_SIGNAL", 0)
        n_pass = flag_counts.get("PASS", 0)
        n_warn = flag_counts.get("WARN", 0)
        n_fail = flag_counts.get("FAIL", 0)

        print(f"\nPer-level closure flag summary: "
              f"PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}  LOW_SIGNAL={n_low}")
        print("""
[NOTE] Per-level closure interpretation:
  LOW_SIGNAL : relative_signal < 0.3% of ea — denominator (ea_p - eb_p) is too
               small for closure_ratio to be meaningful.  Not a model failure.
  PASS/WARN  : sign_agreement >= 55-65%.  Linear FSOI predicts the direction of
               per-stratum error change with better-than-random reliability.
  FAIL       : sign_agreement < 55% — FSOI sign is indistinguishable from random
               at this stratum.  Expected causes:
                 (a) near-zero signal (ea_p ≈ eb_p at this level/variable);
                 (b) GNN nonlinearity at 3-4σ innovations breaks the tangent-
                     linear approximation for this stratum specifically;
                 (c) cross-level message passing means per-stratum gradients do
                     not isolate the stratum being scored.
  These failures do NOT invalidate the global closure or the instrument rankings.
  Use per-level FSOI for qualitative vertical structure, not quantitative impact.
""")

    return closure


def compute_beneficial_fraction(
    df: pd.DataFrame,
    output_dir: Path,
    impact_col: str,
) -> None:
    """Compute per-pair beneficial fraction = |helpful FSOI| / ea_total.

    A well-performing system should have beneficial_fraction ~0.60–0.80.
    Values below 0.50 mean observations are on balance harmful and require
    investigation (model over-fitting, bad background, or data quality issues).

    Writes:
      fsoi_beneficial_fraction.csv  — one row per pair
      fsoi_system_health.csv        — mean + std across all pairs with flag
    """
    ea_col = "ea_p" if "ea_p" in df.columns else "ea"
    if ea_col not in df.columns or impact_col not in df.columns:
        print("[SKIPPED] Beneficial fraction: missing ea or impact column")
        return

    records = []
    for pair_idx, grp in df.groupby("pair_idx", dropna=False):
        ea = float(grp[ea_col].iloc[0])
        if not np.isfinite(ea) or ea == 0:
            continue
        impacts = grp[impact_col].dropna()
        helpful = float(impacts[impacts < 0].sum())
        harmful = float(impacts[impacts > 0].sum())
        total_abs = abs(helpful) + abs(harmful)
        bf = abs(helpful) / ea if ea != 0 else np.nan
        hf = abs(helpful) / total_abs if total_abs > 0 else np.nan
        records.append({
            "pair_idx": pair_idx,
            "curr_bin": grp["curr_bin"].iloc[0] if "curr_bin" in grp.columns else "",
            "ea": ea,
            "helpful_fsoi": helpful,
            "harmful_fsoi": harmful,
            "beneficial_fraction_of_ea": bf,
            "helpful_fraction_of_abs_total": hf,
            "flag": "OK" if (np.isfinite(bf) and bf >= 0.50) else "WARN",
        })

    if not records:
        print("[SKIPPED] Beneficial fraction: no valid pairs")
        return

    bf_df = pd.DataFrame(records)
    bf_path = output_dir / "fsoi_beneficial_fraction.csv"
    bf_df.to_csv(bf_path, index=False)

    mean_bf = float(bf_df["beneficial_fraction_of_ea"].mean())
    std_bf = float(bf_df["beneficial_fraction_of_ea"].std())
    mean_hf = float(bf_df["helpful_fraction_of_abs_total"].mean())
    n_warn = int((bf_df["flag"] == "WARN").sum())

    # Primary metric: what fraction of total |FSOI| is from helpful (negative) obs?
    # This is the standard FSOI "beneficial fraction" used in NWP literature.
    # Values 50-80% are normal; < 40% warrants investigation.
    #
    # NOTE: beneficial_fraction_of_ea (helpful_FSOI / ea_total) is NOT a valid
    # fraction when ea uses per-level errors and FSOI is summed across all levels.
    # Use helpful_fraction_of_abs_total as the primary system health indicator.
    health = pd.DataFrame([{
        "n_pairs": len(bf_df),
        "mean_helpful_fraction_of_abs_total": mean_hf,
        "std_helpful_fraction_of_abs_total": float(bf_df["helpful_fraction_of_abs_total"].std()),
        "mean_beneficial_fraction_of_ea": mean_bf,
        "n_pairs_warn": n_warn,
        "system_flag": "OK" if mean_hf >= 0.40 else "WARN",
    }])
    health.to_csv(output_dir / "fsoi_system_health.csv", index=False)

    print(f"\nHelpful fraction (mean over {len(bf_df)} pairs): "
          f"{mean_hf:.1%}  "
          f"(fraction of total |FSOI| from helpful observations)")
    if mean_hf < 0.40:
        print("  [WARNING] Helpful fraction < 40% — most FSOI is detrimental. "
              "Check model quality and background field.")
    elif mean_hf < 0.50:
        print("  [NOTE] Helpful fraction slightly below typical 50-80% range.")
    else:
        print(f"  [OK] Helpful fraction {mean_hf:.1%} is within healthy range.")
    print(f"  Wrote: {bf_path}")
    print(f"  Wrote: {output_dir / 'fsoi_system_health.csv'}")


_DEFAULT_REGIONS = {
    "tropics": (-30, 30),
    "mid_lat_nh": (30, 60),
    "mid_lat_sh": (-60, -30),
    "polar_nh": (60, 90),
    "polar_sh": (-90, -60),
}


def compute_regional_impact(
    scatter_path: Path,
    output_dir: Path,
    regions: dict | None = None,
) -> None:
    """Compute per-instrument FSOI for each geographic region from scatter_samples.csv.

    This is pure post-processing — no additional model runs needed.
    Requires scatter_samples.csv to have 'lat' and 'lon' columns (enabled by
    default when metadata includes position information).

    Writes:
      fsoi_regional_summary.csv  — per-(region, instrument) aggregates
    """
    if not scatter_path.is_file():
        print(f"[SKIPPED] Regional analysis: {scatter_path} not found")
        return

    df = pd.read_csv(scatter_path)
    if "lat" not in df.columns or "lon" not in df.columns:
        print("[SKIPPED] Regional analysis: scatter_samples.csv has no lat/lon columns. "
              "Re-run fsoi_inference.py to populate them.")
        return

    df = df.dropna(subset=["lat", "fsoi", "instrument"])
    if df.empty:
        print("[SKIPPED] Regional analysis: no valid rows after dropping NaN")
        return

    if regions is None:
        regions = _DEFAULT_REGIONS

    records = []
    for region_name, (lat_lo, lat_hi) in regions.items():
        mask = (df["lat"] >= lat_lo) & (df["lat"] < lat_hi)
        sub = df[mask]
        if sub.empty:
            continue
        for inst, g in sub.groupby("instrument", dropna=False):
            fsoi_vals = g["fsoi"].dropna()
            if fsoi_vals.empty:
                continue
            records.append({
                "region": region_name,
                "lat_lo": lat_lo,
                "lat_hi": lat_hi,
                "instrument": inst,
                "n_samples": int(len(fsoi_vals)),
                "sum_fsoi": float(fsoi_vals.sum()),
                "mean_fsoi": float(fsoi_vals.mean()),
                "positive_frac": float((fsoi_vals > 0).mean()),
                "abs_mean_fsoi": float(fsoi_vals.abs().mean()),
            })

    if not records:
        print("[SKIPPED] Regional analysis: no data in any region")
        return

    reg_df = pd.DataFrame(records)

    # Relative contribution within each region (|mean_fsoi| / sum_region |mean_fsoi|)
    for region_name in reg_df["region"].unique():
        mask = reg_df["region"] == region_name
        denom = reg_df.loc[mask, "abs_mean_fsoi"].sum()
        if denom > 0:
            reg_df.loc[mask, "relative_contribution_pct"] = (
                100.0 * reg_df.loc[mask, "mean_fsoi"] / denom
            )

    out_path = output_dir / "fsoi_regional_summary.csv"
    reg_df.to_csv(out_path, index=False)
    print(f"\nRegional FSOI summary ({len(records)} region×instrument rows):")
    display = reg_df.sort_values(["region", "abs_mean_fsoi"], ascending=[True, False])
    print(display.to_string(index=False))
    print(f"  Wrote: {out_path}")


def evaluate(csv_dir: Path, output_dir: Path, n_boot: int, seed: int) -> None:
    csv_dir = _resolve_csv_dir(csv_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_dir / "fsoi_by_instrument.csv")
    if df.empty:
        raise ValueError(f"No rows found in {csv_dir / 'fsoi_by_instrument.csv'}")
    if "pair_idx" not in df.columns:
        raise ValueError("fsoi_by_instrument.csv must include pair_idx")

    impact_col = _impact_col(df)
    count_col = _count_col(df, impact_col)

    agg_spec = {
        "impact_sum": (impact_col, "sum"),
        "mean_impact": ("mean_impact", "mean"),
        "positive_frac": ("positive_frac", "mean"),
    }
    if count_col:
        agg_spec["n_observations"] = (count_col, "sum")

    pair_summary = (
        df.groupby(["pair_idx", "instrument"], dropna=False)
        .agg(**agg_spec)
        .reset_index()
    )
    pair_summary.to_csv(output_dir / "fsoi_pair_summary.csv", index=False)
    write_closure_diagnostics(df, output_dir, impact_col)
    compute_beneficial_fraction(df, output_dir, impact_col)

    records = []
    for instrument, g in pair_summary.groupby("instrument", dropna=False):
        values = g["impact_sum"].to_numpy(dtype=float)
        ci_low, ci_high = _bootstrap_mean_ci(values, n_boot=n_boot, seed=seed)
        record = {
            "instrument": instrument,
            "n_pairs": int(g["pair_idx"].nunique()),
            "impact_column": impact_col,
            "impact_total": float(np.nansum(values)),
            "impact_mean_per_pair": float(np.nanmean(values)),
            "impact_std_per_pair": float(np.nanstd(values, ddof=0)),
            "impact_mean_per_pair_ci95_low": float(ci_low),
            "impact_mean_per_pair_ci95_high": float(ci_high),
            "mean_impact_mean": float(g["mean_impact"].mean()),
            "positive_frac_mean": float(g["positive_frac"].mean()),
        }
        if "n_observations" in g.columns:
            record["n_observations"] = float(g["n_observations"].sum())
        records.append(record)

    summary = pd.DataFrame(records)
    summary["abs_impact_mean_per_pair"] = summary["impact_mean_per_pair"].abs()
    summary = summary.sort_values("abs_impact_mean_per_pair", ascending=False)
    summary = summary.drop(columns=["abs_impact_mean_per_pair"])
    summary.to_csv(output_dir / "fsoi_evaluation_summary.csv", index=False)

    print(f"Input CSV dir: {csv_dir}")
    print(f"Impact column: {impact_col}")
    print(f"Pairs: {df['pair_idx'].nunique()}")
    print(f"Wrote: {output_dir / 'fsoi_pair_summary.csv'}")
    print(f"Wrote: {output_dir / 'fsoi_evaluation_summary.csv'}")
    print("\nTop instruments by |mean pair impact|:")
    print(summary.head(10).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate FSOI CSV outputs")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="FSOI output directory or csv directory",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for evaluation CSVs (default: <input>/evaluation or sibling evaluation dir)",
    )
    parser.add_argument("--n_boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    input_path = args.input
    if args.output is not None:
        output_dir = args.output
    elif input_path.name == "csv":
        output_dir = input_path.parent / "evaluation"
    else:
        output_dir = input_path / "evaluation"

    evaluate(input_path, output_dir, n_boot=args.n_boot, seed=args.seed)

    # Regional analysis — runs automatically if scatter_samples.csv has lat/lon
    csv_dir = _resolve_csv_dir(input_path)
    scatter_path = csv_dir / "scatter_samples.csv"
    compute_regional_impact(scatter_path, output_dir)


if __name__ == "__main__":
    main()
