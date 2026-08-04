#!/usr/bin/env python
"""
Cross-target channel impact analysis.

For each satellite instrument × channel, determine whether its FSOI impact
is detrimental (positive) or beneficial (negative) when verified against:
  - Radiosonde variables (T, T_d, u, v at pressure levels)
  - Aircraft variables (T, q, u, v at flight levels)
  - Surface obs variables (T, T_d, u, v, p_s)

Identifies channels that are universally detrimental across all three
verification targets, and generates per-target heatmaps for comparison.

Usage:
    python FSOI/analyze_cross_target_channels.py
    python FSOI/analyze_cross_target_channels.py --output FSOI/fsoi_outputs/cross_target_analysis
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_GNN_MODEL = Path(__file__).resolve().parents[1]
_OCELOT    = Path(__file__).resolve().parents[2]
_FSOI      = Path(__file__).resolve().parent

for _p in [str(_GNN_MODEL), str(_FSOI)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_instrument_channel_heatmaps import (
    _prepare_aggregate,
    plot_instrument_heatmap,
    DISPLAY_NAMES,
    CHANNEL_LABELS,
)

# ── Data source configuration ─────────────────────────────────────────────────

def _radiosonde_csv_dir(season: str) -> Path:
    """Resolve a radiosonde season csv dir, preferring the canonical
    ``fsoi_outputs/seasonal_fixed`` location but falling back to the legacy
    repo-root ``seasonal_fixed`` layout when that is where the data lives."""
    primary = _GNN_MODEL / "FSOI" / "fsoi_outputs" / "seasonal_fixed" / f"radiosonde_{season}" / "csv"
    legacy = _OCELOT / "seasonal_fixed" / f"radiosonde_{season}" / "csv"
    return legacy if (legacy.is_dir() and not primary.is_dir()) else primary


TARGETS = {
    "radiosonde": {
        "label": "Radiosonde verification (T, T_d, u, v)",
        "csv_dirs": [
            _radiosonde_csv_dir("jan2025"),
            _radiosonde_csv_dir("apr2025"),
            _radiosonde_csv_dir("oct2025"),
        ],
        "seasons": ["Jan 2025", "Apr 2025", "Oct 2025"],
    },
    "aircraft": {
        "label": "Aircraft verification (T, q, u, v at flight levels)",
        "csv_dirs": [
            _GNN_MODEL / "FSOI" / "fsoi_outputs" / "aircraft_seasonal" / "aircraft_jan2025" / "csv",
            _GNN_MODEL / "FSOI" / "fsoi_outputs" / "aircraft_seasonal" / "aircraft_apr2025" / "csv",
            _GNN_MODEL / "FSOI" / "fsoi_outputs" / "aircraft_seasonal" / "aircraft_jul2025" / "csv",
            _GNN_MODEL / "FSOI" / "fsoi_outputs" / "aircraft_seasonal" / "aircraft_oct2025" / "csv",
        ],
        "seasons": ["Jan 2025", "Apr 2025", "Jul 2025", "Oct 2025"],
    },
    "surface_obs": {
        "label": "Surface obs verification (T, T_d, u, v, p_s)",
        "csv_dirs": [
            _GNN_MODEL / "FSOI" / "fsoi_outputs" / "surface_obs_seasonal" / "surface_obs_jan2025" / "csv",
            _GNN_MODEL / "FSOI" / "fsoi_outputs" / "surface_obs_seasonal" / "surface_obs_apr2025" / "csv",
            _GNN_MODEL / "FSOI" / "fsoi_outputs" / "surface_obs_seasonal" / "surface_obs_jul2025" / "csv",
            _GNN_MODEL / "FSOI" / "fsoi_outputs" / "surface_obs_seasonal" / "surface_obs_oct2025" / "csv",
        ],
        "seasons": ["Jan 2025", "Apr 2025", "Jul 2025", "Oct 2025"],
    },
}

SATELLITE_INSTRUMENTS = ["atms", "amsua", "ssmis", "avhrr", "ascat", "seviri_asr"]

TARGET_DISPLAY = {
    "radiosonde":  "Radiosonde",
    "aircraft":    "Aircraft",
    "surface_obs": "Surface Obs",
}


# ── Data loading ──────────────────────────────────────────────────────────────

def load_and_average(csv_dirs: list, seasons: list) -> pd.DataFrame:
    """Load fsoi_by_channel.csv from multiple seasonal dirs and average."""
    frames = []
    for d, s in zip(csv_dirs, seasons):
        csv = Path(d) / "fsoi_by_channel.csv"
        if not csv.is_file():
            print(f"  [SKIP] {csv}")
            continue
        df = pd.read_csv(csv)
        df["_season"] = s
        frames.append(df)
        print(f"  [OK]   {csv.parent.parent.name}  ({len(df)} rows)")

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    impact_col = "sum_impact_scaled" if "sum_impact_scaled" in combined.columns else "sum_impact"

    group_cols = [c for c in
        ["instrument", "channel", "target_variable", "p_idx", "p_hpa",
         "pressure_hpa", "pressure_level_idx"]
        if c in combined.columns]

    agg = (combined
           .groupby(group_cols, dropna=False)[impact_col]
           .mean()
           .reset_index()
           .rename(columns={impact_col: "mean_impact"}))

    # Keep a renamed column that _prepare_aggregate expects
    if impact_col != "mean_impact":
        agg[impact_col] = agg["mean_impact"]
    return agg


def load_all_targets() -> dict[str, pd.DataFrame]:
    dfs = {}
    for target_name, cfg in TARGETS.items():
        print(f"\nLoading {target_name} channel data:")
        df = load_and_average(cfg["csv_dirs"], cfg["seasons"])
        if df.empty:
            print(f"  WARNING: no data found for {target_name}")
        else:
            print(f"  => {len(df)} averaged rows")
        dfs[target_name] = df
    return dfs


# ── Cross-target impact summary ───────────────────────────────────────────────

def _pressure_level_verdict(df: pd.DataFrame, inst: str, ch, impact_col: str) -> dict:
    """
    For one (instrument, channel) in one target DataFrame, compute:
    - mean_impact: average over all (target_variable, pressure_hpa) combinations
    - frac_detrimental: fraction of (var, pressure) rows that are detrimental (>0)
    - all_levels_detrimental: True only if EVERY (var, pressure) row is detrimental
    - n_rows: number of distinct (var, pressure) combinations
    """
    mask = (df["instrument"] == inst) & (df["channel"] == ch)
    col = impact_col if impact_col in df.columns else "mean_impact"
    sub = df.loc[mask, col].dropna()
    if len(sub) == 0:
        return {"mean": np.nan, "frac_det": np.nan, "all_det": False, "n_rows": 0}
    mean_val = float(sub.mean())
    frac_det = float((sub > 0).mean())
    all_det  = bool((sub > 0).all())
    return {"mean": mean_val, "frac_det": frac_det, "all_det": all_det, "n_rows": len(sub)}


def cross_target_summary(dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    For each (instrument, channel), compute impact statistics within each
    verification target, distinguishing:
      - mean_impact:           average over all (target_variable, pressure) rows
      - frac_detrimental:      fraction of rows with positive (detrimental) impact
      - all_levels_detrimental: True only if EVERY (var, pressure) row is detrimental

    Verdict levels:
      DETRIMENTAL ALL LEVELS  — mean > 0 AND all (var, pressure) rows > 0 for all 3 targets
      DETRIMENTAL AGGREGATE   — mean > 0 but some beneficial rows exist, across all 3 targets
      beneficial all          — mean <= 0 for all 3 targets
      mixed (N/3 detrimental) — sign differs across targets
    """
    impact_col = "sum_impact_scaled"
    records = []

    pairs = set()
    for df in dfs.values():
        if df.empty:
            continue
        pairs.update(zip(df["instrument"], df["channel"]))

    for inst, ch in sorted(pairs):
        if str(inst) not in SATELLITE_INSTRUMENTS:
            continue
        row = {"instrument": inst, "channel": int(ch) if not pd.isna(ch) else ch}

        n_det_mean    = 0   # targets where mean > 0
        n_det_all_lev = 0   # targets where ALL pressure levels > 0
        target_stats  = {}

        for target_name, df in dfs.items():
            if df.empty:
                target_stats[target_name] = {"mean": np.nan, "frac_det": np.nan,
                                              "all_det": False, "n_rows": 0}
                continue
            stats = _pressure_level_verdict(df, inst, ch, impact_col)
            target_stats[target_name] = stats
            if not np.isnan(stats["mean"]) and stats["mean"] > 0:
                n_det_mean += 1
            if stats["all_det"]:
                n_det_all_lev += 1

        for t in TARGETS:
            st = target_stats[t]
            row[f"{t}_mean"]     = st["mean"]
            row[f"{t}_frac_det"] = st["frac_det"]
            row[f"{t}_all_det"]  = st["all_det"]
            row[f"{t}_n_rows"]   = st["n_rows"]

        row["n_detrimental_targets"]    = n_det_mean
        row["n_all_levels_detrimental"] = n_det_all_lev

        n_valid = sum(1 for t in TARGETS
                      if not np.isnan(target_stats[t]["mean"]))

        if n_valid == 0:
            row["verdict"] = "no data"
        elif n_det_mean == n_valid and n_det_all_lev == n_valid:
            row["verdict"] = "DETRIMENTAL ALL LEVELS"
        elif n_det_mean == n_valid:
            row["verdict"] = "DETRIMENTAL AGGREGATE"
        elif n_det_mean == 0:
            row["verdict"] = "beneficial all"
        else:
            row["verdict"] = f"mixed ({n_det_mean}/{n_valid} detrimental)"

        records.append(row)

    return pd.DataFrame(records)


# ── Per-target heatmaps ───────────────────────────────────────────────────────

def _ensure_target_variable_column(df: pd.DataFrame, target_name: str) -> pd.DataFrame:
    """
    Some verification targets (surface_obs) do not stratify by variable/pressure
    so fsoi_by_channel.csv lacks target_variable and pressure_hpa columns.
    Add synthetic ones so _prepare_aggregate can work.

    Important: pressure_hpa may be NaN only for satellite instruments (which have
    no pressure-level stratification for surface_obs verification) while being
    valid for conventional obs. Use fillna(0.0) to preserve existing values.
    """
    df = df.copy()
    if "target_variable" not in df.columns:
        df["target_variable"] = f"{target_name}_agg"
    if "pressure_hpa" not in df.columns:
        df["pressure_hpa"] = 0.0
    else:
        # Fill only NaN values — preserves real pressure values for conventional obs
        df["pressure_hpa"] = df["pressure_hpa"].fillna(0.0)
    if "p_hpa" not in df.columns:
        df["p_hpa"] = df["pressure_hpa"]
    else:
        df["p_hpa"] = df["p_hpa"].fillna(0.0)
    return df


def generate_target_heatmaps(
    df: pd.DataFrame,
    target_name: str,
    target_label: str,
    output_dir: Path,
    instruments: list[str],
):
    """Generate per-channel heatmaps for one verification target."""
    if df.empty:
        print(f"  [SKIP] No data for {target_name}")
        return

    df = _ensure_target_variable_column(df, target_name)

    impact_col = "sum_impact_scaled" if "sum_impact_scaled" in df.columns else "mean_impact"
    if impact_col not in df.columns and "mean_impact" in df.columns:
        df = df.copy()
        df[impact_col] = df["mean_impact"]

    try:
        agg, value_col, _ = _prepare_aggregate(df, basis="total")
    except Exception as e:
        print(f"  [ERROR] _prepare_aggregate failed for {target_name}: {e}")
        return

    title_suffix = f"Seasonal average — {target_label}"
    sub_dir = output_dir / target_name
    sub_dir.mkdir(parents=True, exist_ok=True)

    for inst in instruments:
        inst_lower = str(inst).lower()
        matched = [i for i in agg["instrument"].unique() if str(i).lower() == inst_lower]
        if not matched:
            print(f"  [SKIP] {inst}: no rows in {target_name} data")
            continue
        vt = TARGET_DISPLAY.get(target_name, target_name.replace("_", " ").title())
        try:
            out, _ = plot_instrument_heatmap(
                agg=agg,
                instrument=matched[0],
                output_dir=sub_dir,
                value_col=value_col,
                title_suffix=title_suffix,
                mode="value",
                verification_target=vt,
            )
            print(f"  Saved: {out.relative_to(output_dir.parent.parent)}")
        except Exception as e:
            print(f"  [ERROR] {inst} in {target_name}: {e}")


# ── Cross-target comparison summary table (markdown) ─────────────────────────

def write_markdown_summary(cross_df: pd.DataFrame, output_dir: Path, dfs: dict):
    """Write a markdown summary of cross-target channel impacts."""
    md_path = output_dir / "CROSS_TARGET_CHANNEL_ANALYSIS.md"

    lines = []
    lines.append("# Cross-Target Satellite Channel Impact Analysis")
    lines.append("")
    lines.append("Impact sign across all three observation-space verification targets.")
    lines.append("Values are mean per-observation FSOI averaged across all available")
    lines.append("seasons and all (target variable, pressure level) combinations.")
    lines.append("**Negative = beneficial; Positive = detrimental.**")
    lines.append("")
    lines.append("> **Verdict definitions:**")
    lines.append("> - **DETRIMENTAL ALL LEVELS**: mean FSOI > 0 AND every individual")
    lines.append(">   (target_variable, pressure_level) row is detrimental for all 3 targets.")
    lines.append(">   This is the strongest possible detriment — no beneficial combination exists.")
    lines.append("> - **DETRIMENTAL AGGREGATE**: mean FSOI > 0 for all 3 targets, but some")
    lines.append(">   (var, pressure) rows are individually beneficial. The net is detrimental")
    lines.append(">   but the channel carries useful information at specific levels.")
    lines.append("> - **beneficial all**: mean FSOI ≤ 0 for all 3 targets.")
    lines.append("> - **mixed**: sign differs across verification targets.")
    lines.append("> `frac_det` = fraction of (variable × pressure) rows that are detrimental.")
    lines.append("")

    for inst in SATELLITE_INSTRUMENTS:
        inst_df = cross_df[cross_df["instrument"] == inst].copy()
        if inst_df.empty:
            continue
        inst_df = inst_df.sort_values("channel")
        display = DISPLAY_NAMES.get(inst, inst.upper())

        lines.append(f"---")
        lines.append(f"## {display}")
        lines.append("")

        # Summary counts
        n_det_all = (inst_df["verdict"] == "DETRIMENTAL ALL").sum()
        n_ben_all = (inst_df["verdict"] == "beneficial all").sum()
        n_mixed   = len(inst_df) - n_det_all - n_ben_all
        lines.append(f"**{n_det_all} channels universally detrimental | "
                     f"{n_ben_all} universally beneficial | {n_mixed} mixed**")
        lines.append("")

        # Table header
        lines.append("| Ch | Radiosonde mean (frac_det) | Aircraft mean (frac_det) | Surface obs mean (frac_det) | Verdict |")
        lines.append("|---|---|---|---|---|")

        for _, r in inst_df.iterrows():
            ch = int(r["channel"]) if not pd.isna(r["channel"]) else "?"

            def fmt(target):
                m  = r.get(f"{target}_mean",     np.nan)
                fd = r.get(f"{target}_frac_det", np.nan)
                ad = r.get(f"{target}_all_det",  False)
                if pd.isna(m):
                    return "—"
                flag = " ★" if ad else ""  # ★ = detrimental at ALL (var, pressure) rows
                return f"{m:+.4f} ({fd:.0%}){flag}"

            rs = fmt("radiosonde")
            ac = fmt("aircraft")
            sf = fmt("surface_obs")
            vd = r["verdict"]
            tag = "**" if "DETRIMENTAL" in str(vd) else ""
            lines.append(f"| {tag}{ch}{tag} | {tag}{rs}{tag} | {tag}{ac}{tag} | {tag}{sf}{tag} | {tag}{vd}{tag} |")

        lines.append("")

        # Highlight channels detrimental at ALL pressure levels
        det_all_lev = inst_df[inst_df["verdict"] == "DETRIMENTAL ALL LEVELS"]
        if not det_all_lev.empty:
            chs = ", ".join(str(int(c)) for c in det_all_lev["channel"])
            lines.append(f"> **Channels detrimental at ALL pressure levels across all 3 targets: {chs}**")
            lines.append("> (No beneficial (variable, pressure) combination found for these channels)")
            lines.append("")

        # Highlight channels detrimental in aggregate (but not at every level)
        det_agg = inst_df[inst_df["verdict"] == "DETRIMENTAL AGGREGATE"]
        if not det_agg.empty:
            chs = ", ".join(str(int(c)) for c in det_agg["channel"])
            lines.append(f"> Channels with net detrimental aggregate across all 3 targets")
            lines.append(f"> (but beneficial at some levels): {chs}")
            lines.append("")

        # Highlight universally beneficial channels
        ben_all = inst_df[inst_df["verdict"] == "beneficial all"]
        if not ben_all.empty:
            chs = ", ".join(str(int(c)) for c in ben_all["channel"])
            lines.append(f"> Channels beneficial across all 3 targets: {chs}")
            lines.append("")

    # Overall cross-instrument summary
    lines.append("---")
    lines.append("## Summary: Channels Detrimental Across All Three Verification Targets")
    lines.append("")
    lines.append("| Instrument | Detrimental at ALL levels (strongest) | Net detrimental aggregate only |")
    lines.append("|---|---|---|")
    for inst in SATELLITE_INSTRUMENTS:
        inst_df = cross_df[cross_df["instrument"] == inst]
        det_lev = inst_df[inst_df["verdict"] == "DETRIMENTAL ALL LEVELS"]
        det_agg = inst_df[inst_df["verdict"] == "DETRIMENTAL AGGREGATE"]
        chs_lev = ", ".join(f"ch{int(c)}" for c in sorted(det_lev["channel"])) if not det_lev.empty else "none"
        chs_agg = ", ".join(f"ch{int(c)}" for c in sorted(det_agg["channel"])) if not det_agg.empty else "none"
        display = DISPLAY_NAMES.get(inst, inst.upper())
        lines.append(f"| {display} | {chs_lev} | {chs_agg} |")

    lines.append("")
    lines.append("---")
    lines.append("## Summary: Channels Beneficial Across All Three Verification Targets")
    lines.append("")
    lines.append("| Instrument | Universally beneficial channels |")
    lines.append("|---|---|")
    for inst in SATELLITE_INSTRUMENTS:
        inst_df = cross_df[cross_df["instrument"] == inst]
        ben = inst_df[inst_df["verdict"] == "beneficial all"]
        if ben.empty:
            chs = "none"
        else:
            chs = ", ".join(f"ch{int(c)}" for c in sorted(ben["channel"]))
        display = DISPLAY_NAMES.get(inst, inst.upper())
        lines.append(f"| {display} | {chs} |")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved: {md_path.name}")
    return md_path


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-target satellite channel impact analysis and heatmap generation."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_FSOI / "fsoi_outputs" / "cross_target_analysis",
        help="Output directory for figures and summary",
    )
    parser.add_argument(
        "--instruments",
        nargs="*",
        default=SATELLITE_INSTRUMENTS,
        help="Instruments to analyse",
    )
    parser.add_argument(
        "--skip_heatmaps",
        action="store_true",
        help="Skip heatmap generation (only produce the summary table)",
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("=" * 70)
    print("Loading channel data for all 3 verification targets")
    print("=" * 70)
    dfs = load_all_targets()

    # ── Generate per-target heatmaps ──────────────────────────────────────────
    if not args.skip_heatmaps:
        print("\n" + "=" * 70)
        print("Generating per-target heatmaps")
        print("=" * 70)
        for target_name, cfg in TARGETS.items():
            print(f"\n  Target: {target_name}")
            generate_target_heatmaps(
                df=dfs[target_name],
                target_name=target_name,
                target_label=cfg["label"],
                output_dir=args.output / "heatmaps",
                instruments=args.instruments,
            )

    # ── Cross-target analysis ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Cross-target channel analysis")
    print("=" * 70)
    cross_df = cross_target_summary(dfs)

    # Save CSV
    csv_path = args.output / "cross_target_channel_summary.csv"
    cross_df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path.name}")

    # Save markdown
    write_markdown_summary(cross_df, args.output, dfs)

    # Print console summary
    print("\n=== DETRIMENTAL AT ALL PRESSURE LEVELS (all 3 targets — strongest finding) ===")
    det_all = cross_df[cross_df["verdict"] == "DETRIMENTAL ALL LEVELS"]
    if det_all.empty:
        print("  None found.")
    else:
        for _, r in det_all.iterrows():
            print(f"  {DISPLAY_NAMES.get(r['instrument'], r['instrument']):12s}  "
                  f"ch{int(r['channel']):2d}  |  "
                  f"radiosonde={r['radiosonde_mean']:+.4f}  "
                  f"aircraft={r['aircraft_mean']:+.4f}  "
                  f"surface={r['surface_obs_mean']:+.4f}")

    print("\n=== NET DETRIMENTAL AGGREGATE but NOT all levels (all 3 targets) ===")
    det_agg = cross_df[cross_df["verdict"] == "DETRIMENTAL AGGREGATE"]
    if det_agg.empty:
        print("  None found.")
    else:
        for _, r in det_agg.iterrows():
            print(f"  {DISPLAY_NAMES.get(r['instrument'], r['instrument']):12s}  "
                  f"ch{int(r['channel']):2d}  |  "
                  f"radiosonde={r['radiosonde_mean']:+.4f} ({r['radiosonde_frac_det']:.0%} lev)  "
                  f"aircraft={r['aircraft_mean']:+.4f} ({r['aircraft_frac_det']:.0%} lev)  "
                  f"surface={r['surface_obs_mean']:+.4f} ({r['surface_obs_frac_det']:.0%} lev)")

    print("\n=== UNIVERSALLY BENEFICIAL CHANNELS (all 3 targets) ===")
    ben_all = cross_df[cross_df["verdict"] == "beneficial all"]
    if ben_all.empty:
        print("  None found.")
    else:
        for _, r in ben_all.iterrows():
            print(f"  {DISPLAY_NAMES.get(r['instrument'], r['instrument']):12s}  "
                  f"ch{int(r['channel']):2d}  |  "
                  f"radiosonde={r['radiosonde_mean']:+.4f}  "
                  f"aircraft={r['aircraft_mean']:+.4f}  "
                  f"surface={r['surface_obs_mean']:+.4f}")

    print(f"\nAll outputs in: {args.output}")


if __name__ == "__main__":
    main()
