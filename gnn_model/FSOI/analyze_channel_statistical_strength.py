#!/usr/bin/env python
"""Statistical channel screening for OCELOT FSOI outputs.

This is post-processing only. It strengthens channel-level interpretation by
using 12-hour cycles, not variable-pressure rows, as statistical replicates.

Inputs are the current seasonal FSOI CSVs:

    fsoi_outputs/seasonal_sentinel_fixed/{target}_{month}2025/csv/fsoi_by_channel.csv

Outputs:

    channel_cycle_impacts.csv
    channel_monthly_bootstrap_stats.csv
    channel_fdr_results.csv
    channel_replication_summary.csv
    channel_cross_target_candidates.csv
    CHANNEL_STATISTICAL_STRENGTH.md
"""

from __future__ import annotations

import argparse
import math
import re
import zlib
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


MONTHS = ("jan", "apr", "jul", "oct")
TARGETS = ("radiosonde", "aircraft", "surface_obs")
SATELLITE_INSTRUMENTS = ("atms", "amsua", "ssmis", "avhrr", "ascat", "seviri_asr")
CI_LO = 2.5
CI_HI = 97.5
EPS = 1e-12


def _stable_seed(seed: int, *parts: object) -> int:
    key = "::".join(str(p) for p in parts).encode("utf-8")
    return (int(seed) + zlib.crc32(key)) % (2**32 - 1)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cycle-level statistical screening for FSOI channel results."
    )
    parser.add_argument(
        "--seasonal-root",
        type=Path,
        default=Path("gnn_model/FSOI/fsoi_outputs/seasonal_sentinel_fixed"),
        help="Root containing {target}_{month}2025 seasonal FSOI output folders.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("gnn_model/FSOI/fsoi_outputs/channel_statistical_strength_current"),
        help="Output directory for strengthened channel statistics.",
    )
    parser.add_argument(
        "--targets",
        nargs="+",
        default=list(TARGETS),
        help="Verification targets to include.",
    )
    parser.add_argument(
        "--months",
        nargs="+",
        default=list(MONTHS),
        help="Months to include, e.g. jan apr jul oct.",
    )
    parser.add_argument(
        "--instruments",
        nargs="+",
        default=list(SATELLITE_INSTRUMENTS),
        help="Source instruments/channels to test.",
    )
    parser.add_argument(
        "--block-cycles",
        type=int,
        default=10,
        help="Bootstrap block length in 12-hour cycles. 10 cycles = 5 days.",
    )
    parser.add_argument(
        "--n-boot",
        type=int,
        default=5000,
        help="Number of block-bootstrap samples.",
    )
    parser.add_argument(
        "--fdr-alpha",
        type=float,
        default=0.10,
        help="Benjamini-Hochberg false-discovery-rate threshold.",
    )
    parser.add_argument(
        "--min-relative-effect-pct",
        type=float,
        default=0.001,
        help=(
            "Minimum practical median effect size as percent of the matched "
            "target-group control error."
        ),
    )
    parser.add_argument(
        "--min-abs-impact",
        type=float,
        default=0.0,
        help="Optional minimum practical absolute cycle impact in normalized units.",
    )
    parser.add_argument(
        "--min-cycles",
        type=int,
        default=20,
        help="Minimum cycles required for a channel-month-group statistic.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20250825,
        help="Base random seed.",
    )
    return parser.parse_args()


def _month_label(month: str) -> str:
    return {
        "jan": "January",
        "apr": "April",
        "jul": "July",
        "oct": "October",
    }.get(month.lower(), month)


def _cycle_sort_key(value: object) -> tuple[int, str]:
    text = str(value)
    match = re.search(r"(\d{10})", text)
    if match:
        return int(match.group(1)), text
    try:
        return int(float(text)), text
    except ValueError:
        return 0, text


def _pressure_layer(p_hpa: object) -> str | None:
    try:
        p = float(p_hpa)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(p):
        return None
    if p >= 700.0:
        return "lower_troposphere"
    if p >= 300.0:
        return "middle_troposphere"
    return "upper_troposphere"


def _target_groups(row: pd.Series) -> list[str]:
    """Return physically meaningful target groups for one channel row."""
    groups = ["all"]
    var = str(row.get("target_variable", "")).strip()
    if var:
        groups.append(var)
    if var in {"u_wind", "v_wind"}:
        groups.append("wind")

    layer = _pressure_layer(row.get("p_hpa", row.get("pressure_hpa", np.nan)))
    if layer is not None:
        groups.append(layer)

    # Preserve insertion order while removing duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for group in groups:
        if group not in seen:
            unique.append(group)
            seen.add(group)
    return unique


def _read_seasonal_channel_csv(
    seasonal_root: Path,
    target: str,
    month: str,
    instruments: set[str],
) -> pd.DataFrame:
    path = seasonal_root / f"{target}_{month}2025" / "csv" / "fsoi_by_channel.csv"
    if not path.is_file():
        print(f"[WARN] Missing channel CSV: {path}")
        return pd.DataFrame()

    df = pd.read_csv(path)
    impact_col = "sum_impact_scaled" if "sum_impact_scaled" in df.columns else "sum_impact"
    required = {"instrument", "channel", impact_col, "pair_idx"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    df = df[df["instrument"].astype(str).str.lower().isin(instruments)].copy()
    if df.empty:
        return df

    df["instrument"] = df["instrument"].astype(str).str.lower()
    df["channel"] = pd.to_numeric(df["channel"], errors="coerce").astype("Int64")
    df = df[df["channel"].notna()].copy()
    df["channel"] = df["channel"].astype(int)
    df["target"] = target
    df["month"] = month
    df["month_label"] = _month_label(month)
    df["run_name"] = f"{target}_{month}2025"
    df["cycle_id"] = (
        df["curr_bin"].astype(str)
        if "curr_bin" in df.columns
        else df["pair_idx"].astype(str)
    )
    df["cycle_order"] = df["cycle_id"].map(_cycle_sort_key)
    df["impact"] = pd.to_numeric(df[impact_col], errors="coerce")

    if "ea" in df.columns:
        df["control_error"] = pd.to_numeric(df["ea"], errors="coerce")
    elif "ea_total" in df.columns:
        df["control_error"] = pd.to_numeric(df["ea_total"], errors="coerce")
    else:
        df["control_error"] = np.nan

    df = df[np.isfinite(df["impact"])].copy()
    return df


def build_cycle_impacts(
    seasonal_root: Path,
    targets: Iterable[str],
    months: Iterable[str],
    instruments: Iterable[str],
) -> pd.DataFrame:
    instruments_set = {inst.lower() for inst in instruments}
    frames: list[pd.DataFrame] = []
    for target in targets:
        for month in months:
            df = _read_seasonal_channel_csv(
                seasonal_root=seasonal_root,
                target=target,
                month=month.lower(),
                instruments=instruments_set,
            )
            if df.empty:
                continue

            work = df.copy()
            work["target_group"] = work.apply(_target_groups, axis=1)
            work = work.explode("target_group", ignore_index=True)

            group_cols = [
                "target",
                "month",
                "month_label",
                "run_name",
                "instrument",
                "channel",
                "target_group",
                "cycle_id",
                "pair_idx",
                "cycle_order",
            ]
            grouped = (
                work.groupby(group_cols, dropna=False)
                .agg(
                    cycle_impact=("impact", "sum"),
                    target_group_control_error=("control_error", "sum"),
                    n_target_rows=("impact", "size"),
                    n_positive_rows=("impact", lambda x: int((x > 0).sum())),
                    n_negative_rows=("impact", lambda x: int((x < 0).sum())),
                )
                .reset_index()
            )
            grouped["relative_impact_pct"] = np.where(
                np.abs(grouped["target_group_control_error"]) > EPS,
                100.0
                * grouped["cycle_impact"]
                / grouped["target_group_control_error"],
                np.nan,
            )
            frames.append(grouped)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(
        ["target", "month", "instrument", "channel", "target_group", "cycle_order"]
    ).reset_index(drop=True)
    return out


def _bootstrap_indices(
    n: int,
    block_cycles: int,
    n_boot: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 0), dtype=int)
    block = max(1, min(int(block_cycles), n))
    n_blocks = int(math.ceil(n / block))
    max_start = max(1, n - block + 1)
    starts = rng.integers(0, max_start, size=(n_boot, n_blocks))
    offsets = np.arange(block, dtype=int)
    idx = starts[:, :, None] + offsets[None, None, :]
    idx = idx.reshape(n_boot, n_blocks * block)[:, :n]
    return idx


def _ci(values: np.ndarray) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.nanpercentile(values, [CI_LO, CI_HI])
    return float(lo), float(hi)


def _dominant_direction(mean_impact: float) -> str:
    if mean_impact > 0:
        return "detrimental"
    if mean_impact < 0:
        return "beneficial"
    return "neutral"


def bootstrap_monthly_stats(
    cycle_df: pd.DataFrame,
    *,
    block_cycles: int,
    n_boot: int,
    seed: int,
    min_cycles: int,
    min_relative_effect_pct: float,
    min_abs_impact: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["target", "month", "month_label", "instrument", "channel", "target_group"]
    for key, grp in cycle_df.groupby(group_cols, sort=True):
        target, month, month_label, inst, channel, target_group = key
        grp = grp.sort_values("cycle_order").copy()
        n = len(grp)
        if n < min_cycles:
            continue

        impact = grp["cycle_impact"].to_numpy(dtype=float)
        rel = grp["relative_impact_pct"].to_numpy(dtype=float)
        control = grp["target_group_control_error"].to_numpy(dtype=float)
        practical_threshold = np.maximum(
            float(min_abs_impact),
            np.abs(control) * float(min_relative_effect_pct) / 100.0,
        )

        finite_rel = rel[np.isfinite(rel)]
        mean_impact = float(np.mean(impact))
        median_impact = float(np.median(impact))
        mean_rel = float(np.mean(finite_rel)) if finite_rel.size else float("nan")
        median_rel = float(np.median(finite_rel)) if finite_rel.size else float("nan")
        det_frac = float(np.mean(impact > 0))
        ben_frac = float(np.mean(impact < 0))
        det_practical_frac = float(np.mean(impact >= practical_threshold))
        ben_practical_frac = float(np.mean(impact <= -practical_threshold))
        direction = _dominant_direction(mean_impact)

        rng = np.random.default_rng(
            _stable_seed(seed, target, month, inst, channel, target_group)
        )
        idx = _bootstrap_indices(n, block_cycles, n_boot, rng)
        boot_impact = impact[idx]
        boot_mean = np.mean(boot_impact, axis=1)
        boot_median = np.median(boot_impact, axis=1)
        boot_det_frac = np.mean(boot_impact > 0, axis=1)
        boot_ben_frac = np.mean(boot_impact < 0, axis=1)
        boot_det_practical = np.mean(boot_impact >= practical_threshold[idx], axis=1)
        boot_ben_practical = np.mean(boot_impact <= -practical_threshold[idx], axis=1)
        if finite_rel.size == n:
            boot_rel = rel[idx]
            boot_median_rel = np.median(boot_rel, axis=1)
        else:
            boot_median_rel = np.full(n_boot, np.nan)

        mean_lo, mean_hi = _ci(boot_mean)
        med_lo, med_hi = _ci(boot_median)
        det_lo, det_hi = _ci(boot_det_frac)
        ben_lo, ben_hi = _ci(boot_ben_frac)
        det_pr_lo, det_pr_hi = _ci(boot_det_practical)
        ben_pr_lo, ben_pr_hi = _ci(boot_ben_practical)
        rel_lo, rel_hi = _ci(boot_median_rel)

        p_detrimental = float((1 + np.sum(boot_mean <= 0.0)) / (n_boot + 1))
        p_beneficial = float((1 + np.sum(boot_mean >= 0.0)) / (n_boot + 1))

        pass_practical = (
            np.isfinite(median_rel)
            and abs(median_rel) >= float(min_relative_effect_pct)
            and abs(median_impact) >= float(min_abs_impact)
        )

        rows.append(
            {
                "target": target,
                "month": month,
                "month_label": month_label,
                "instrument": inst,
                "channel": int(channel),
                "target_group": target_group,
                "n_cycles": n,
                "n_boot": int(n_boot),
                "block_cycles": int(block_cycles),
                "mean_cycle_impact": mean_impact,
                "mean_cycle_impact_ci95_low": mean_lo,
                "mean_cycle_impact_ci95_high": mean_hi,
                "median_cycle_impact": median_impact,
                "median_cycle_impact_ci95_low": med_lo,
                "median_cycle_impact_ci95_high": med_hi,
                "mean_relative_impact_pct": mean_rel,
                "median_relative_impact_pct": median_rel,
                "median_relative_impact_pct_ci95_low": rel_lo,
                "median_relative_impact_pct_ci95_high": rel_hi,
                "detrimental_cycle_fraction": det_frac,
                "detrimental_cycle_fraction_ci95_low": det_lo,
                "detrimental_cycle_fraction_ci95_high": det_hi,
                "beneficial_cycle_fraction": ben_frac,
                "beneficial_cycle_fraction_ci95_low": ben_lo,
                "beneficial_cycle_fraction_ci95_high": ben_hi,
                "practical_detrimental_cycle_fraction": det_practical_frac,
                "practical_detrimental_cycle_fraction_ci95_low": det_pr_lo,
                "practical_detrimental_cycle_fraction_ci95_high": det_pr_hi,
                "practical_beneficial_cycle_fraction": ben_practical_frac,
                "practical_beneficial_cycle_fraction_ci95_low": ben_pr_lo,
                "practical_beneficial_cycle_fraction_ci95_high": ben_pr_hi,
                "p_detrimental": p_detrimental,
                "p_beneficial": p_beneficial,
                "dominant_direction": direction,
                "pass_practical_effect": bool(pass_practical),
            }
        )

    return pd.DataFrame(rows)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        return q
    finite_idx = np.flatnonzero(finite)
    p_finite = p[finite]
    order = np.argsort(p_finite)
    sorted_p = p_finite[order]
    m = len(sorted_p)
    sorted_q = sorted_p * m / np.arange(1, m + 1)
    sorted_q = np.minimum.accumulate(sorted_q[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0.0, 1.0)
    q_finite = np.empty_like(sorted_q)
    q_finite[order] = sorted_q
    q[finite_idx] = q_finite
    return q


def add_fdr(monthly: pd.DataFrame, fdr_alpha: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    records: list[pd.DataFrame] = []
    for direction, p_col in [
        ("detrimental", "p_detrimental"),
        ("beneficial", "p_beneficial"),
    ]:
        tmp = monthly[
            [
                "target",
                "month",
                "month_label",
                "target_group",
                "instrument",
                "channel",
                p_col,
            ]
        ].copy()
        tmp = tmp.rename(columns={p_col: "p_value"})
        tmp["tested_direction"] = direction
        records.append(tmp)
    fdr = pd.concat(records, ignore_index=True)
    fdr["q_value"] = np.nan
    family_cols = ["target", "month", "target_group", "tested_direction"]
    for _, idx in fdr.groupby(family_cols).groups.items():
        idx = list(idx)
        fdr.loc[idx, "q_value"] = _benjamini_hochberg(
            fdr.loc[idx, "p_value"].to_numpy(dtype=float)
        )
    fdr["pass_fdr"] = fdr["q_value"] <= float(fdr_alpha)

    monthly = monthly.copy()
    monthly["dominant_p_value"] = np.where(
        monthly["dominant_direction"].eq("detrimental"),
        monthly["p_detrimental"],
        np.where(monthly["dominant_direction"].eq("beneficial"), monthly["p_beneficial"], np.nan),
    )
    monthly["dominant_q_value"] = np.nan
    monthly["pass_fdr"] = False

    key_cols = ["target", "month", "target_group", "instrument", "channel"]
    q_lookup = fdr.set_index(key_cols + ["tested_direction"])[["q_value", "pass_fdr"]]
    for idx, row in monthly.iterrows():
        direction = row["dominant_direction"]
        if direction == "neutral":
            continue
        key = tuple(row[col] for col in key_cols) + (direction,)
        if key in q_lookup.index:
            monthly.at[idx, "dominant_q_value"] = float(q_lookup.loc[key, "q_value"])
            monthly.at[idx, "pass_fdr"] = bool(q_lookup.loc[key, "pass_fdr"])

    monthly["robust_direction"] = np.where(
        monthly["pass_practical_effect"] & monthly["pass_fdr"],
        monthly["dominant_direction"],
        "not_robust",
    )
    return monthly, fdr


def build_replication_summary(monthly: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    group_cols = ["target", "target_group", "instrument", "channel"]
    for key, grp in monthly.groupby(group_cols, sort=True):
        target, target_group, inst, channel = key
        det_months = sorted(grp.loc[grp["robust_direction"].eq("detrimental"), "month_label"].unique())
        ben_months = sorted(grp.loc[grp["robust_direction"].eq("beneficial"), "month_label"].unique())
        rows.append(
            {
                "target": target,
                "target_group": target_group,
                "instrument": inst,
                "channel": int(channel),
                "n_months_tested": int(grp["month"].nunique()),
                "n_robust_detrimental_months": len(det_months),
                "n_robust_beneficial_months": len(ben_months),
                "robust_detrimental_months": ";".join(det_months),
                "robust_beneficial_months": ";".join(ben_months),
                "replicated_detrimental": len(det_months) >= 2,
                "replicated_beneficial": len(ben_months) >= 2,
                "median_of_monthly_relative_impact_pct": float(
                    grp["median_relative_impact_pct"].median()
                ),
                "median_of_monthly_cycle_impact": float(grp["median_cycle_impact"].median()),
                "mean_detrimental_cycle_fraction": float(
                    grp["detrimental_cycle_fraction"].mean()
                ),
                "mean_beneficial_cycle_fraction": float(
                    grp["beneficial_cycle_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_cross_target_candidates(replication: pd.DataFrame) -> pd.DataFrame:
    all_group = replication[replication["target_group"].eq("all")].copy()
    rows: list[dict[str, object]] = []
    for (inst, channel), grp in all_group.groupby(["instrument", "channel"], sort=True):
        mixed_mask = grp["replicated_detrimental"] & grp["replicated_beneficial"]
        det_mask = grp["replicated_detrimental"] & ~grp["replicated_beneficial"]
        ben_mask = grp["replicated_beneficial"] & ~grp["replicated_detrimental"]
        det_targets = sorted(grp.loc[det_mask, "target"].unique())
        ben_targets = sorted(grp.loc[ben_mask, "target"].unique())
        mixed_targets = sorted(grp.loc[mixed_mask, "target"].unique())
        rows.append(
            {
                "instrument": inst,
                "channel": int(channel),
                "n_targets_replicated_detrimental": len(det_targets),
                "replicated_detrimental_targets": ";".join(det_targets),
                "n_targets_replicated_beneficial": len(ben_targets),
                "replicated_beneficial_targets": ";".join(ben_targets),
                "n_targets_mixed_replicated_sign": len(mixed_targets),
                "mixed_replicated_sign_targets": ";".join(mixed_targets),
                "cross_target_detrimental_candidate": len(det_targets) >= 2,
                "cross_target_beneficial_candidate": len(ben_targets) >= 2,
            }
        )
    return pd.DataFrame(rows)


def _fmt_pct(value: float, digits: int = 1) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{100.0 * value:.{digits}f}%"


def _fmt_num(value: float, digits: int = 4) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{digits}g}"


def write_markdown_report(
    out_dir: Path,
    monthly: pd.DataFrame,
    replication: pd.DataFrame,
    cross: pd.DataFrame,
    *,
    fdr_alpha: float,
    min_relative_effect_pct: float,
    block_cycles: int,
    n_boot: int,
) -> None:
    report = out_dir / "CHANNEL_STATISTICAL_STRENGTH.md"
    lines: list[str] = []
    lines.append("# Channel Statistical Strength Results")
    lines.append("")
    lines.append("This report treats 12-hour cycles, not variable-pressure rows, as the statistical replicates.")
    lines.append("")
    lines.append("## Method")
    lines.append("")
    lines.append(f"- Block bootstrap: `{block_cycles}` cycles per block (`{block_cycles / 2:.1f}` days for 12-hour cycles).")
    lines.append(f"- Bootstrap samples: `{n_boot}`.")
    lines.append(f"- FDR threshold: `q <= {fdr_alpha}` using Benjamini-Hochberg correction within each target, month, target group, and sign direction.")
    lines.append(f"- Minimum practical median effect: `{min_relative_effect_pct}%` of the target-group control error.")
    lines.append("- A monthly channel result is robust only if it passes both FDR and practical-effect filters.")
    lines.append("- A replicated result requires the same robust sign in at least two months.")
    lines.append("- A cross-target candidate requires replication in at least two verification targets for the `all` target group.")
    lines.append("")

    lines.append("## Cross-Target Candidates")
    lines.append("")
    det = cross[cross["cross_target_detrimental_candidate"]].copy()
    ben = cross[cross["cross_target_beneficial_candidate"]].copy()
    if det.empty:
        lines.append("No channels pass the strict cross-target detrimental-candidate rule.")
    else:
        lines.append("| Channel | Replicated detrimental targets |")
        lines.append("|---|---|")
        for _, row in det.sort_values(
            ["n_targets_replicated_detrimental", "instrument", "channel"],
            ascending=[False, True, True],
        ).iterrows():
            lines.append(
                f"| {row['instrument']} ch{int(row['channel'])} | "
                f"{row['replicated_detrimental_targets']} |"
            )
    lines.append("")
    if ben.empty:
        lines.append("No channels pass the strict cross-target beneficial-candidate rule.")
    else:
        lines.append("| Channel | Replicated beneficial targets |")
        lines.append("|---|---|")
        for _, row in ben.sort_values(
            ["n_targets_replicated_beneficial", "instrument", "channel"],
            ascending=[False, True, True],
        ).iterrows():
            lines.append(
                f"| {row['instrument']} ch{int(row['channel'])} | "
                f"{row['replicated_beneficial_targets']} |"
            )
    lines.append("")
    mixed = cross[cross["n_targets_mixed_replicated_sign"] > 0].copy()
    if not mixed.empty:
        lines.append("Targets with replicated detrimental and replicated beneficial months are labeled mixed and are not counted as clean cross-target evidence.")
        lines.append("")
        lines.append("| Channel | Mixed targets |")
        lines.append("|---|---|")
        for _, row in mixed.sort_values(
            ["n_targets_mixed_replicated_sign", "instrument", "channel"],
            ascending=[False, True, True],
        ).head(20).iterrows():
            lines.append(
                f"| {row['instrument']} ch{int(row['channel'])} | "
                f"{row['mixed_replicated_sign_targets']} |"
            )
        lines.append("")

    lines.append("## Strongest Replicated Detrimental Results")
    lines.append("")
    rep_det = replication[replication["replicated_detrimental"]].copy()
    if rep_det.empty:
        lines.append("No target/group/channel passes replicated detrimental criteria.")
    else:
        rep_det = rep_det.sort_values(
            [
                "n_robust_detrimental_months",
                "mean_detrimental_cycle_fraction",
                "median_of_monthly_relative_impact_pct",
            ],
            ascending=[False, False, False],
        ).head(25)
        lines.append("| Target | Group | Channel | Months | Mean detrimental-cycle fraction | Median relative impact |")
        lines.append("|---|---|---|---|---:|---:|")
        for _, row in rep_det.iterrows():
            lines.append(
                f"| {row['target']} | {row['target_group']} | "
                f"{row['instrument']} ch{int(row['channel'])} | "
                f"{row['robust_detrimental_months']} | "
                f"{_fmt_pct(row['mean_detrimental_cycle_fraction'])} | "
                f"{_fmt_num(row['median_of_monthly_relative_impact_pct'])}% |"
            )
    lines.append("")

    lines.append("## SSMIS Channel 21 Check")
    lines.append("")
    ssmis21 = replication[
        (replication["instrument"].eq("ssmis")) & (replication["channel"].eq(21))
    ].copy()
    if ssmis21.empty:
        lines.append("SSMIS ch21 was not present in the replication summary.")
    else:
        cols = [
            "target",
            "target_group",
            "n_robust_detrimental_months",
            "robust_detrimental_months",
            "n_robust_beneficial_months",
            "robust_beneficial_months",
            "median_of_monthly_relative_impact_pct",
        ]
        lines.append("| Target | Group | Robust detrimental months | Robust beneficial months | Median monthly relative impact |")
        lines.append("|---|---|---:|---:|---:|")
        for _, row in ssmis21[cols].sort_values(["target", "target_group"]).iterrows():
            det_months = row["robust_detrimental_months"] or "-"
            ben_months = row["robust_beneficial_months"] or "-"
            lines.append(
                f"| {row['target']} | {row['target_group']} | "
                f"{int(row['n_robust_detrimental_months'])} ({det_months}) | "
                f"{int(row['n_robust_beneficial_months'])} ({ben_months}) | "
                f"{_fmt_num(row['median_of_monthly_relative_impact_pct'])}% |"
            )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `channel_cycle_impacts.csv`: cycle-level channel impacts by target group.")
    lines.append("- `channel_monthly_bootstrap_stats.csv`: cycle-block bootstrap statistics and robust monthly flags.")
    lines.append("- `channel_fdr_results.csv`: Benjamini-Hochberg q-values for beneficial and detrimental tests.")
    lines.append("- `channel_replication_summary.csv`: months of replicated robust sign by target/group/channel.")
    lines.append("- `channel_cross_target_candidates.csv`: strict cross-target candidates based on the `all` target group.")
    lines.append("")

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cycle_df = build_cycle_impacts(
        seasonal_root=args.seasonal_root,
        targets=args.targets,
        months=args.months,
        instruments=args.instruments,
    )
    if cycle_df.empty:
        raise SystemExit("No channel cycle impacts were built. Check --seasonal-root.")

    cycle_path = args.out_dir / "channel_cycle_impacts.csv"
    cycle_df.to_csv(cycle_path, index=False)
    print(f"[WRITE] {cycle_path} ({len(cycle_df)} rows)")

    monthly = bootstrap_monthly_stats(
        cycle_df,
        block_cycles=args.block_cycles,
        n_boot=args.n_boot,
        seed=args.seed,
        min_cycles=args.min_cycles,
        min_relative_effect_pct=args.min_relative_effect_pct,
        min_abs_impact=args.min_abs_impact,
    )
    monthly, fdr = add_fdr(monthly, args.fdr_alpha)

    monthly_path = args.out_dir / "channel_monthly_bootstrap_stats.csv"
    fdr_path = args.out_dir / "channel_fdr_results.csv"
    monthly.to_csv(monthly_path, index=False)
    fdr.to_csv(fdr_path, index=False)
    print(f"[WRITE] {monthly_path} ({len(monthly)} rows)")
    print(f"[WRITE] {fdr_path} ({len(fdr)} rows)")

    replication = build_replication_summary(monthly)
    cross = build_cross_target_candidates(replication)

    replication_path = args.out_dir / "channel_replication_summary.csv"
    cross_path = args.out_dir / "channel_cross_target_candidates.csv"
    replication.to_csv(replication_path, index=False)
    cross.to_csv(cross_path, index=False)
    print(f"[WRITE] {replication_path} ({len(replication)} rows)")
    print(f"[WRITE] {cross_path} ({len(cross)} rows)")

    write_markdown_report(
        args.out_dir,
        monthly,
        replication,
        cross,
        fdr_alpha=args.fdr_alpha,
        min_relative_effect_pct=args.min_relative_effect_pct,
        block_cycles=args.block_cycles,
        n_boot=args.n_boot,
    )
    print(f"[WRITE] {args.out_dir / 'CHANNEL_STATISTICAL_STRENGTH.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
