#!/usr/bin/env python
"""
Create per-instrument channel heatmaps for radiosonde target metrics.

Rows are the scored radiosonde target variable and pressure level.  Columns are
the source instrument channels or variables.  One PNG is written per source
instrument.
"""

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    from visualize_fsoi import (
        STANDARD_PRESSURE_LEVELS,
        _count_col_for_impact,
        _impact_sum_col,
    )
except ImportError:
    STANDARD_PRESSURE_LEVELS = [
        1000, 925, 850, 700, 500, 400, 300, 250,
        200, 150, 100, 70, 50, 30, 20, 10,
    ]

    def _impact_sum_col(df: pd.DataFrame) -> str:
        return "sum_impact_scaled" if "sum_impact_scaled" in df.columns else "sum_impact"

    def _count_col_for_impact(df: pd.DataFrame, default: str) -> str:
        if _impact_sum_col(df) == "sum_impact_scaled":
            if default == "total_count" and "total_count_scaled" in df.columns:
                return "total_count_scaled"
        return default


TARGET_VARIABLE_ORDER = ["temperature", "dewpoint_temperature", "u_wind", "v_wind"]

CHANNEL_LABELS = {
    "radiosonde": {
        1: "temperature",
        2: "dewpoint",
        3: "u wind",
        4: "v wind",
    },
    "aircraft": {
        1: "temperature",
        2: "humidity",
        3: "u wind",
        4: "v wind",
    },
    "surface_obs": {
        1: "temperature",
        2: "dewpoint",
        3: "u wind",
        4: "v wind",
        5: "pressure",
    },
    "ascat": {
        1: "u wind",
        2: "v wind",
    },
}

DISPLAY_NAMES = {
    "amsua": "AMSU-A",
    "atms": "ATMS",
    "avhrr": "AVHRR",
    "ssmis": "SSMIS",
    "seviri_asr": "SEVIRI ASR",
    "seviri_csr": "SEVIRI CSR",
    "radiosonde": "Radiosonde",
    "aircraft": "Aircraft",
    "surface_obs": "Surface",
    "ascat": "ASCAT",
}


def _resolve_csv_dir(path: Path) -> Path:
    if (path / "fsoi_by_channel.csv").is_file():
        return path
    if (path / "csv" / "fsoi_by_channel.csv").is_file():
        return path / "csv"
    raise FileNotFoundError(
        f"Could not find fsoi_by_channel.csv in {path} or {path / 'csv'}"
    )


def _default_output_dir(input_path: Path, csv_dir: Path) -> Path:
    if input_path.name == "csv":
        return input_path.parent / "figures"
    if csv_dir.name == "csv":
        return csv_dir.parent / "figures"
    return input_path / "figures"


def _safe_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(name).strip()).strip("_").lower()


def _ensure_pressure_hpa(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "pressure_hpa" in df.columns and df["pressure_hpa"].notna().any():
        return df
    if "p_hpa" in df.columns and df["p_hpa"].notna().any():
        df["pressure_hpa"] = df["p_hpa"]
        return df
    if "pressure_level_idx" in df.columns:
        level_map = {i: p for i, p in enumerate(STANDARD_PRESSURE_LEVELS)}
        df["pressure_hpa"] = df["pressure_level_idx"].map(level_map)
    else:
        df["pressure_hpa"] = np.nan
    return df


def _add_channel_display(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 1-based channel_display column, supporting old zero-based CSVs."""
    df = df.copy()
    channels = pd.to_numeric(df["channel"], errors="coerce")
    df["channel_display"] = channels

    for inst, idx in df.groupby("instrument", dropna=False).groups.items():
        vals = channels.loc[idx].dropna()
        if vals.empty:
            continue
        if vals.min() == 0:
            df.loc[idx, "channel_display"] = channels.loc[idx] + 1

    return df


def _channel_label(instrument: str, channel: float) -> str:
    inst = str(instrument).lower()
    if not np.isfinite(channel):
        return "unknown"
    ch = int(channel)

    if inst in CHANNEL_LABELS and ch in CHANNEL_LABELS[inst]:
        return CHANNEL_LABELS[inst][ch]
    if inst in {"seviri_asr", "seviri_csr"}:
        return f"ch{ch + 3}"
    return f"ch{ch}"


def _row_label(row: pd.Series) -> str:
    p = float(row["pressure_hpa"])
    p_txt = f"{int(p)}" if p.is_integer() else f"{p:g}"
    return f"{row['target_variable']} @ {p_txt}hPa"


def _prepare_aggregate(df_ch: pd.DataFrame, basis: str) -> tuple[pd.DataFrame, str, str]:
    df = _ensure_pressure_hpa(df_ch)
    required = {"instrument", "channel", "target_variable", "pressure_hpa"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fsoi_by_channel.csv is missing required columns: {sorted(missing)}")

    df = df[df["target_variable"].notna() & df["pressure_hpa"].notna()].copy()
    if df.empty:
        raise ValueError("No rows with target_variable and pressure_hpa found")

    df = _add_channel_display(df)
    impact_col = _impact_sum_col(df)
    count_col = _count_col_for_impact(df, "total_count")

    group_cols = ["instrument", "channel_display", "target_variable", "pressure_hpa"]
    agg_spec = {"impact_sum": (impact_col, "sum")}
    has_counts = basis == "mean_per_obs" and count_col in df.columns
    if has_counts:
        agg_spec["total_count"] = (count_col, "sum")

    agg = df.groupby(group_cols, dropna=False).agg(**agg_spec).reset_index()
    if has_counts:
        agg["mean_per_obs"] = agg["impact_sum"] / agg["total_count"].replace(0, np.nan)
        value_col = "mean_per_obs"
        title_suffix = f"per-observation mean from {impact_col}"
    else:
        if basis == "mean_per_obs":
            print(f"[WARNING] Count column {count_col!r} not found; using total impact instead")
        value_col = "impact_sum"
        title_suffix = impact_col

    agg["channel_label"] = [
        _channel_label(inst, ch)
        for inst, ch in zip(agg["instrument"], agg["channel_display"])
    ]
    return agg, value_col, title_suffix


def _ordered_rows(agg_inst: pd.DataFrame) -> pd.DataFrame:
    out = agg_inst.copy()
    out["target_variable"] = out["target_variable"].astype(str)
    out["row_label"] = out.apply(_row_label, axis=1)
    var_rank = {v: i for i, v in enumerate(TARGET_VARIABLE_ORDER)}
    out["_var_rank"] = out["target_variable"].map(var_rank).fillna(len(var_rank)).astype(int)
    out["_p_sort"] = -out["pressure_hpa"].astype(float)
    return out.sort_values(["_var_rank", "_p_sort", "channel_display"])


def plot_instrument_heatmap(
    agg: pd.DataFrame,
    instrument: str,
    output_dir: Path,
    value_col: str,
    title_suffix: str,
    mode: str,
    verification_target: str = "Radiosonde",
) -> tuple[Path, pd.DataFrame]:
    inst_df = agg[agg["instrument"] == instrument].copy()
    if inst_df.empty:
        raise ValueError(f"No rows found for instrument {instrument}")

    inst_df = _ordered_rows(inst_df)
    channel_order = sorted(inst_df["channel_display"].dropna().unique())
    channel_labels = {
        ch: _channel_label(instrument, ch)
        for ch in channel_order
    }

    pivot = inst_df.pivot_table(
        index=["_var_rank", "_p_sort", "target_variable", "pressure_hpa", "row_label"],
        columns="channel_display",
        values=value_col,
        aggfunc="sum",
        fill_value=0.0,
    ).reindex(columns=channel_order, fill_value=0.0)

    if mode == "relative":
        row_abs = pivot.abs().sum(axis=1).replace(0.0, np.nan)
        plot_values = pivot.div(row_abs, axis=0).mul(100.0).fillna(0.0)
        cbar_label = "Relative channel contribution (%)  (negative = helpful)"
        mode_title = "Relative Channel Contribution"
    else:
        plot_values = pivot
        cbar_label = f"{value_col}  (negative = helpful)"
        mode_title = "Channel Impact"

    labels = [idx[4] for idx in plot_values.index]
    xlabels = [channel_labels[ch] for ch in plot_values.columns]

    width = max(10.0, 0.45 * len(xlabels) + 4.0)
    height = max(8.0, 0.24 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(width, height))

    vmax = np.nanmax(np.abs(plot_values.values))
    vmax = 1.0 if (not np.isfinite(vmax) or vmax == 0) else float(vmax)
    im = ax.imshow(
        plot_values.values,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xticks(np.arange(len(xlabels)))
    ax.set_xticklabels(xlabels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"({verification_target} Target Variable, Pressure)")
    ax.set_xlabel(f"{DISPLAY_NAMES.get(instrument, instrument)} channels / variables")
    ax.set_title(
        f"{DISPLAY_NAMES.get(instrument, instrument)}: {mode_title} by {verification_target} Target\n"
        f"{title_suffix}"
    )

    # Add separators between target variables for readability.
    row_meta = pd.DataFrame(
        {
            "label": labels,
            "var_rank": [idx[0] for idx in plot_values.index],
        }
    )
    changes = np.where(row_meta["var_rank"].to_numpy()[1:] != row_meta["var_rank"].to_numpy()[:-1])[0]
    for pos in changes:
        ax.axhline(pos + 0.5, color="black", linewidth=0.5, alpha=0.35)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)

    plt.tight_layout()
    out = output_dir / f"instrument_channel_variable_pressure_heatmap_{_safe_name(instrument)}.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    plt.close()

    long_values = plot_values.stack().rename("plot_value").reset_index()
    long_values["instrument"] = instrument
    long_values["channel_label"] = long_values["channel_display"].map(channel_labels)
    return out, long_values


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create one channel-by-target heatmap per FSOI instrument."
    )
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
        help="Output figure directory (default: <input>/figures)",
    )
    parser.add_argument(
        "--instruments",
        nargs="*",
        default=None,
        help="Optional instrument names to plot. Default: all instruments.",
    )
    parser.add_argument(
        "--mode",
        choices=["relative", "value"],
        default="relative",
        help="Plot signed relative channel contribution or raw values.",
    )
    parser.add_argument(
        "--basis",
        choices=["mean_per_obs", "total"],
        default="mean_per_obs",
        help="Base value before optional relative normalization.",
    )
    args = parser.parse_args()

    csv_dir = _resolve_csv_dir(args.input)
    output_dir = args.output or _default_output_dir(args.input, csv_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_ch = pd.read_csv(csv_dir / "fsoi_by_channel.csv")
    agg, value_col, title_suffix = _prepare_aggregate(df_ch, basis=args.basis)

    instruments = args.instruments or sorted(str(v) for v in agg["instrument"].dropna().unique())
    all_values = []
    print(f"Input CSV dir: {csv_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Mode: {args.mode}; basis: {args.basis}; value column: {value_col}")
    for instrument in instruments:
        if instrument not in set(agg["instrument"]):
            print(f"[SKIPPED] {instrument}: no rows")
            continue
        out, values = plot_instrument_heatmap(
            agg=agg,
            instrument=instrument,
            output_dir=output_dir,
            value_col=value_col,
            title_suffix=title_suffix,
            mode=args.mode,
        )
        all_values.append(values)
        print(f"Saved: {out}")

    if all_values:
        values_df = pd.concat(all_values, ignore_index=True)
        values_csv = output_dir / "instrument_channel_variable_pressure_heatmap_values.csv"
        values_df.to_csv(values_csv, index=False)
        print(f"Wrote: {values_csv}")


if __name__ == "__main__":
    main()
