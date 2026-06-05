#!/usr/bin/env python
"""
Generate publication-quality per-channel heatmaps for ATMS, AMSU-A, SSMIS, and ASCAT.

Averages fsoi_by_channel.csv across all available seasons (Jan, Apr, Oct 2025)
then calls plot_instrument_channel_heatmaps to produce one PNG per instrument.

Usage:
    python FSOI/generate_paper_heatmaps.py
    python FSOI/generate_paper_heatmaps.py --output FSOI/fsoi_outputs/paper_figures
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_GNN_MODEL = Path(__file__).resolve().parents[1]   # gnn_model/
_OCELOT    = Path(__file__).resolve().parents[2]   # ocelot/ (repo root)
_FSOI      = Path(__file__).resolve().parent       # gnn_model/FSOI/

for _p in (_GNN_MODEL, str(_FSOI)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from plot_instrument_channel_heatmaps import (  # noqa: E402
    _prepare_aggregate,
    plot_instrument_heatmap,
    DISPLAY_NAMES,
    STANDARD_PRESSURE_LEVELS,
)

# ── Configuration ────────────────────────────────────────────────────────────

SEASONAL_CSV_DIRS = [
    _OCELOT / "seasonal_fixed" / "radiosonde_jan2025" / "csv",
    _OCELOT / "seasonal_fixed" / "radiosonde_apr2025" / "csv",
    _OCELOT / "seasonal_fixed" / "radiosonde_oct2025" / "csv",
]

SEASON_LABELS = ["Jan 2025", "Apr 2025", "Oct 2025"]

TARGET_INSTRUMENTS = ["atms", "amsua", "ssmis", "ascat"]

INSTRUMENT_TITLES = {
    "atms":  "ATMS (22 channels)",
    "amsua": "AMSU-A (15 channels)",
    "ssmis": "SSMIS (24 channels)",
    "ascat": "ASCAT (u-wind, v-wind)",
}


def load_and_combine(csv_dirs: list[Path]) -> pd.DataFrame:
    """Load channel CSVs from multiple seasons and average impact values."""
    frames = []
    for d, label in zip(csv_dirs, SEASON_LABELS):
        csv = d / "fsoi_by_channel.csv"
        if not csv.is_file():
            print(f"[SKIP] {csv} not found")
            continue
        df = pd.read_csv(csv)
        df["_season"] = label
        frames.append(df)
        print(f"[LOADED] {csv}  ({len(df)} rows)")

    if not frames:
        raise FileNotFoundError("No fsoi_by_channel.csv files found in any seasonal directory.")

    combined = pd.concat(frames, ignore_index=True)

    # Identify the impact column (prefer sum_impact_scaled)
    impact_col = "sum_impact_scaled" if "sum_impact_scaled" in combined.columns else "sum_impact"

    # Average across seasons for each (instrument, channel, target_variable, pressure)
    group_cols = [c for c in
        ["instrument", "channel", "target_variable", "p_idx", "p_hpa",
         "pressure_hpa", "pressure_level_idx"]
        if c in combined.columns]

    numeric_cols = [impact_col]
    if "total_count" in combined.columns:
        numeric_cols.append("total_count")
    if "n_observations" in combined.columns:
        numeric_cols.append("n_observations")

    agg = (
        combined
        .groupby(group_cols, dropna=False)[numeric_cols]
        .mean()
        .reset_index()
    )

    # Rename averaged column back so downstream code finds it
    if impact_col not in agg.columns:
        agg = agg.rename(columns={numeric_cols[0]: impact_col})

    print(f"\nCombined: {len(frames)} seasons -> {len(agg)} averaged rows")
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper-quality FSOI channel heatmaps averaged across seasons."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=_FSOI / "fsoi_outputs" / "paper_figures",
        help="Output directory for figure PNGs",
    )
    parser.add_argument(
        "--instruments",
        nargs="*",
        default=TARGET_INSTRUMENTS,
        help="Instruments to plot (default: atms amsua ssmis ascat)",
    )
    parser.add_argument(
        "--mode",
        choices=["relative", "value"],
        default="value",
        help="'value' = raw impact sum; 'relative' = % of row total",
    )
    parser.add_argument(
        "--basis",
        choices=["mean_per_obs", "total"],
        default="mean_per_obs",
        help="Per-observation mean or total impact",
    )
    args = parser.parse_args()

    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and average channel data across seasons
    df_combined = load_and_combine(SEASONAL_CSV_DIRS)

    # Prepare aggregate (reuse existing logic from plot_instrument_channel_heatmaps)
    agg, value_col, title_suffix = _prepare_aggregate(df_combined, basis=args.basis)

    seasons_used = ", ".join(SEASON_LABELS[: len(SEASONAL_CSV_DIRS)])
    title_suffix = f"Averaged across {seasons_used}"

    print(f"\nGenerating heatmaps -> {output_dir}")
    for instrument in args.instruments:
        inst_lower = str(instrument).lower()
        if inst_lower not in set(agg["instrument"].str.lower()):
            print(f"[SKIP] {instrument}: no rows in combined data")
            continue

        # Normalise instrument name to match what's in the dataframe
        matched = [i for i in agg["instrument"].unique()
                   if str(i).lower() == inst_lower]
        if not matched:
            continue
        inst_key = matched[0]

        out, _ = plot_instrument_heatmap(
            agg=agg,
            instrument=inst_key,
            output_dir=output_dir,
            value_col=value_col,
            title_suffix=title_suffix,
            mode=args.mode,
        )
        print(f"  Saved: {out.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
