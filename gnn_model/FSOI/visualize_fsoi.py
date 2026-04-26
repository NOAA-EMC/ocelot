"""
FSOI Visualization - Create plots from FSOI results.

This script creates standard visualizations from FSOI CSV output:
1. Bar chart: Total impact by instrument
2. Time series: Impact evolution over time
3. Heatmap: Channel-level impacts
4. Scatter: Positive vs negative fractions

Usage:
    python visualize_fsoi.py --input FSOI/fsoi_outputs/csv --output fsoi_plots

Author: Azadeh Gholoubi
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


STANDARD_PRESSURE_LEVELS = [
    1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10
]


def _ensure_pressure_hpa(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure `pressure_hpa` exists using `pressure_level_idx` if available."""
    df = df.copy()
    if 'pressure_hpa' in df.columns and df['pressure_hpa'].notna().any():
        return df
    if 'pressure_level_idx' in df.columns:
        level_map = {i: p for i, p in enumerate(STANDARD_PRESSURE_LEVELS)}
        df['pressure_hpa'] = df['pressure_level_idx'].map(level_map)
    else:
        df['pressure_hpa'] = np.nan
    return df


def plot_instrument_impacts(df_inst, output_dir):
    """Bar chart of total impact by instrument."""
    print("Creating instrument impact plot...")

    # Aggregate by instrument
    impacts = df_inst.groupby('instrument')['sum_impact'].sum().sort_values()

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if x < 0 else 'red' for x in impacts.values]
    impacts.plot(kind='barh', ax=ax, color=colors, alpha=0.7)

    ax.set_xlabel('Total FSOI Impact (negative = helpful)')
    ax.set_ylabel('Instrument')
    ax.set_title('Observation Impact on Forecast Error')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'instrument_impacts.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'instrument_impacts.png'}")


def plot_instrument_relative_contribution(df_inst: pd.DataFrame, output_dir: Path):
    """Bar chart of signed relative contribution by data type (instrument).

    Following the GraphDOP FSOI paper convention, we primarily use the
    *mean* FSOI per data type (per-observation average) to avoid domination
    by data types with many observations.

    We compute:
      mean_inst = sum_impact_inst / n_observations_inst  (if available)
                  else mean_impact_inst
      rel_% = 100 * mean_inst / sum(|mean_inst|)
    """
    print("Creating instrument relative contribution plot...")

    gb = df_inst.groupby('instrument')
    sum_impact = gb['sum_impact'].sum()

    # Prefer a true per-observation mean if counts exist
    count_col = None
    for c in ('n_observations', 'total_count', 'count'):
        if c in df_inst.columns:
            count_col = c
            break

    if count_col is not None:
        counts = gb[count_col].sum().replace(0, np.nan)
        mean_inst = (sum_impact / counts)
        subtitle = f"(normalized by {count_col})"
    elif 'mean_impact' in df_inst.columns:
        mean_inst = gb['mean_impact'].mean()
        subtitle = "(using mean_impact)"
    else:
        # Fall back to sum-based if no other choice
        mean_inst = sum_impact
        subtitle = "(sum-based; counts unavailable)"

    mean_inst = mean_inst.sort_values()
    denom = mean_inst.abs().sum()
    if denom == 0 or not np.isfinite(denom):
        print("  Skipping: cannot compute relative contribution (zero/invalid total)")
        return

    rel = 100.0 * mean_inst / denom

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['green' if x < 0 else 'red' for x in rel.values]
    rel.plot(kind='barh', ax=ax, color=colors, alpha=0.7)

    ax.set_xlabel('Relative Contribution to Radiosonde Forecast Error (%)')
    ax.set_ylabel('Data Type (Instrument)')
    ax.set_title(f'Global Relative Contribution by Observation Type\n{subtitle}')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = output_dir / 'instrument_relative_contribution.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_pressure_instrument_heatmap(df_ch: pd.DataFrame, output_dir: Path, top_n: int = 20):
    """Heatmap: pressure (y) vs data type/instrument (x).

    Value is signed relative contribution (%) at each pressure level:
      rel_%(p, inst) = 100 * impact(p, inst) / sum_inst |impact(p, inst)|

    This requires pressure-stratified output (pressure_hpa or pressure_level_idx).
    """
    print("Creating pressure × instrument relative contribution heatmap...")
    if df_ch is None or df_ch.empty:
        print("  Skipping: no channel data")
        return

    df = _ensure_pressure_hpa(df_ch)
    if 'pressure_hpa' not in df.columns or not df['pressure_hpa'].notna().any():
        print("  Skipping: no pressure stratification found (need pressure_hpa/pressure_level_idx)")
        return

    # Aggregate channel → instrument per pressure level.
    # Prefer per-observation mean at each (pressure, instrument) when counts are available.
    dfp = df[df['pressure_hpa'].notna()].copy()

    has_counts = 'total_count' in dfp.columns
    if has_counts:
        agg = dfp.groupby(['pressure_hpa', 'instrument']).agg(
            sum_impact=('sum_impact', 'sum'),
            total_count=('total_count', 'sum'),
        ).reset_index()
        agg['mean_per_obs'] = agg['sum_impact'] / agg['total_count'].replace(0, np.nan)
        value_col = 'mean_per_obs'
        title_suffix = '(per-observation mean)'
    elif 'mean_impact' in dfp.columns:
        agg = dfp.groupby(['pressure_hpa', 'instrument']).agg(
            mean_impact=('mean_impact', 'mean'),
        ).reset_index()
        value_col = 'mean_impact'
        title_suffix = '(mean_impact)'
    else:
        agg = dfp.groupby(['pressure_hpa', 'instrument']).agg(
            sum_impact=('sum_impact', 'sum'),
        ).reset_index()
        value_col = 'sum_impact'
        title_suffix = '(sum-based)'
    if agg.empty:
        print("  Skipping: no (pressure, instrument) aggregated data")
        return

    # Optionally restrict to top-N instruments by total absolute impact
    top = agg.groupby('instrument')[value_col].sum().abs().sort_values(ascending=False).head(top_n).index
    agg = agg[agg['instrument'].isin(top)]

    pivot = agg.pivot(index='pressure_hpa', columns='instrument', values=value_col).fillna(0.0)
    # Sort pressures ascending then invert axis (so 1000 hPa at bottom)
    pivot = pivot.sort_index(ascending=True)

    # Compute signed relative contribution per pressure row
    row_abs = pivot.abs().sum(axis=1).replace(0.0, np.nan)
    rel = (pivot.div(row_abs, axis=0) * 100.0).fillna(0.0)

    fig, ax = plt.subplots(figsize=(max(12, 0.6 * len(rel.columns)), 8))
    vmax = np.nanmax(np.abs(rel.values))
    vmax = 1.0 if (not np.isfinite(vmax) or vmax == 0) else float(vmax)

    im = ax.imshow(
        rel.values,
        cmap='RdBu_r',
        aspect='auto',
        vmin=-vmax,
        vmax=vmax,
        interpolation='nearest',
    )

    ax.set_yticks(np.arange(len(rel.index)))
    ax.set_yticklabels([f"{int(p)}" if float(p).is_integer() else f"{p:g}" for p in rel.index])
    ax.set_xticks(np.arange(len(rel.columns)))
    ax.set_xticklabels(rel.columns, rotation=45, ha='right')

    ax.set_ylabel('Radiosonde Target Pressure (hPa)')
    ax.set_xlabel('Data Type (Instrument)')
    ax.set_title(f'Relative Contribution by Data Type vs Radiosonde Pressure Level\n{title_suffix}')
    ax.invert_yaxis()

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Contribution (%)  (negative = helpful)')

    plt.tight_layout()
    out = output_dir / 'instrument_contribution_by_pressure_heatmap.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_pressure_instrument_heatmaps_by_variable(df_ch: pd.DataFrame, output_dir: Path, top_n: int = 20):
    """One heatmap per target variable: pressure (y) × instrument (x)."""
    if df_ch is None or df_ch.empty:
        return
    if 'target_variable' not in df_ch.columns or df_ch['target_variable'].isna().all():
        print("No target_variable column found; skipping per-variable heatmaps")
        return

    vars_found = [v for v in df_ch['target_variable'].dropna().unique().tolist() if str(v).strip()]
    if not vars_found:
        print("No target variables found; skipping per-variable heatmaps")
        return

    order = ['temperature', 'dewpoint_temperature', 'u_wind', 'v_wind']
    vars_sorted = [v for v in order if v in vars_found] + [v for v in vars_found if v not in order]

    for v in vars_sorted:
        print(f"Creating per-variable heatmap for target_variable={v}...")
        dfv = df_ch[df_ch['target_variable'] == v].copy()
        if dfv.empty:
            continue

        # Reuse the same computation but customize output name/title
        df = _ensure_pressure_hpa(dfv)
        if 'pressure_hpa' not in df.columns or not df['pressure_hpa'].notna().any():
            print(f"  Skipping {v}: missing pressure_hpa")
            continue

        dfp = df[df['pressure_hpa'].notna()].copy()
        has_counts = 'total_count' in dfp.columns
        if has_counts:
            agg = dfp.groupby(['pressure_hpa', 'instrument']).agg(
                sum_impact=('sum_impact', 'sum'),
                total_count=('total_count', 'sum'),
            ).reset_index()
            agg['mean_per_obs'] = agg['sum_impact'] / agg['total_count'].replace(0, np.nan)
            value_col = 'mean_per_obs'
            title_suffix = '(per-observation mean)'
        elif 'mean_impact' in dfp.columns:
            agg = dfp.groupby(['pressure_hpa', 'instrument']).agg(
                mean_impact=('mean_impact', 'mean'),
            ).reset_index()
            value_col = 'mean_impact'
            title_suffix = '(mean_impact)'
        else:
            agg = dfp.groupby(['pressure_hpa', 'instrument']).agg(
                sum_impact=('sum_impact', 'sum'),
            ).reset_index()
            value_col = 'sum_impact'
            title_suffix = '(sum-based)'
        if agg.empty:
            continue

        top = agg.groupby('instrument')[value_col].sum().abs().sort_values(ascending=False).head(top_n).index
        agg = agg[agg['instrument'].isin(top)]
        pivot = agg.pivot(index='pressure_hpa', columns='instrument', values=value_col).fillna(0.0)
        pivot = pivot.sort_index(ascending=True)
        row_abs = pivot.abs().sum(axis=1).replace(0.0, np.nan)
        rel = (pivot.div(row_abs, axis=0) * 100.0).fillna(0.0)

        fig, ax = plt.subplots(figsize=(max(12, 0.6 * len(rel.columns)), 8))
        vmax = np.nanmax(np.abs(rel.values))
        vmax = 1.0 if (not np.isfinite(vmax) or vmax == 0) else float(vmax)
        im = ax.imshow(
            rel.values,
            cmap='RdBu_r',
            aspect='auto',
            vmin=-vmax,
            vmax=vmax,
            interpolation='nearest',
        )
        ax.set_yticks(np.arange(len(rel.index)))
        ax.set_yticklabels([f"{int(p)}" if float(p).is_integer() else f"{p:g}" for p in rel.index])
        ax.set_xticks(np.arange(len(rel.columns)))
        ax.set_xticklabels(rel.columns, rotation=45, ha='right')
        ax.set_ylabel('Radiosonde Target Pressure (hPa)')
        ax.set_xlabel('Data Type (Instrument)')
        ax.set_title(f'Relative Contribution vs Pressure for Target Variable: {v}\n{title_suffix}')
        ax.invert_yaxis()
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Relative Contribution (%)  (negative = helpful)')

        plt.tight_layout()
        out = output_dir / f'instrument_contribution_by_pressure_heatmap_{v}.png'
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out}")


def plot_variable_pressure_combined_heatmap(df_ch: pd.DataFrame, output_dir: Path, top_n: int = 20):
    """Combined heatmap with y-axis = (target_variable, pressure_hpa) pairs and x-axis = instrument."""
    print("Creating combined (variable, pressure) × instrument heatmap...")
    if df_ch is None or df_ch.empty:
        print("  Skipping: no channel data")
        return
    if 'target_variable' not in df_ch.columns or df_ch['target_variable'].isna().all():
        print("  Skipping: no target_variable column")
        return

    df = _ensure_pressure_hpa(df_ch)
    if 'pressure_hpa' not in df.columns or not df['pressure_hpa'].notna().any():
        print("  Skipping: no pressure_hpa")
        return

    dfp = df[df['pressure_hpa'].notna() & df['target_variable'].notna()].copy()
    if dfp.empty:
        print("  Skipping: empty after filtering")
        return

    has_counts = 'total_count' in dfp.columns
    if has_counts:
        agg = dfp.groupby(['target_variable', 'pressure_hpa', 'instrument']).agg(
            sum_impact=('sum_impact', 'sum'),
            total_count=('total_count', 'sum'),
        ).reset_index()
        agg['mean_per_obs'] = agg['sum_impact'] / agg['total_count'].replace(0, np.nan)
        value_col = 'mean_per_obs'
        title_suffix = '(per-observation mean)'
    elif 'mean_impact' in dfp.columns:
        agg = dfp.groupby(['target_variable', 'pressure_hpa', 'instrument']).agg(
            mean_impact=('mean_impact', 'mean'),
        ).reset_index()
        value_col = 'mean_impact'
        title_suffix = '(mean_impact)'
    else:
        agg = dfp.groupby(['target_variable', 'pressure_hpa', 'instrument']).agg(
            sum_impact=('sum_impact', 'sum'),
        ).reset_index()
        value_col = 'sum_impact'
        title_suffix = '(sum-based)'
    if agg.empty:
        print("  Skipping: no aggregated data")
        return

    top = agg.groupby('instrument')[value_col].sum().abs().sort_values(ascending=False).head(top_n).index
    agg = agg[agg['instrument'].isin(top)]

    # Build row labels as "var @ p" and sort with a stable variable order.
    order = ['temperature', 'dewpoint_temperature', 'u_wind', 'v_wind']
    agg['target_variable'] = agg['target_variable'].astype(str)

    def _row_label(r):
        p = float(r['pressure_hpa'])
        if p.is_integer():
            p_txt = f"{int(p)}"
        else:
            p_txt = f"{p:g}"
        return f"{r['target_variable']} @ {p_txt}hPa"

    agg['row_label'] = agg.apply(_row_label, axis=1)

    # Sort rows by variable then descending pressure
    var_rank = {v: i for i, v in enumerate(order)}
    agg['_var_rank'] = agg['target_variable'].map(var_rank).fillna(len(order)).astype(int)
    agg['_p_sort'] = -agg['pressure_hpa'].astype(float)
    agg = agg.sort_values(['_var_rank', '_p_sort', 'instrument'])

    pivot = agg.pivot(index=['_var_rank', '_p_sort', 'row_label'], columns='instrument', values=value_col).fillna(0.0)
    # Normalize per-row to signed relative contribution
    row_abs = pivot.abs().sum(axis=1).replace(0.0, np.nan)
    rel = (pivot.div(row_abs, axis=0) * 100.0).fillna(0.0)

    labels = [idx[2] for idx in rel.index]
    fig, ax = plt.subplots(figsize=(max(12, 0.6 * len(rel.columns)), max(8, 0.25 * len(labels))))
    vmax = np.nanmax(np.abs(rel.values))
    vmax = 1.0 if (not np.isfinite(vmax) or vmax == 0) else float(vmax)
    im = ax.imshow(
        rel.values,
        cmap='RdBu_r',
        aspect='auto',
        vmin=-vmax,
        vmax=vmax,
        interpolation='nearest',
    )

    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_xticks(np.arange(len(rel.columns)))
    ax.set_xticklabels(rel.columns, rotation=45, ha='right')
    ax.set_ylabel('(Target Variable, Pressure)')
    ax.set_xlabel('Data Type (Instrument)')
    ax.set_title(f'Relative Contribution by Data Type for (Variable, Pressure) Targets\n{title_suffix}')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Relative Contribution (%)  (negative = helpful)')

    plt.tight_layout()
    out = output_dir / 'instrument_contribution_by_variable_pressure_heatmap.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_innovation_vs_fsoi_scatter(df_scatter: pd.DataFrame, output_dir: Path, top_instruments: int = 4):
    """Innovation vs FSOI scatter/hexbin plots from sampled per-observation points."""
    print("Creating innovation vs FSOI scatter plots...")
    if df_scatter is None or df_scatter.empty:
        print("  Skipping: no scatter sample data")
        return
    required = {'instrument', 'innovation', 'fsoi'}
    if not required.issubset(set(df_scatter.columns)):
        print(f"  Skipping: scatter_samples missing columns {required - set(df_scatter.columns)}")
        return

    df = df_scatter.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['instrument', 'innovation', 'fsoi'])
    if df.empty:
        print("  Skipping: empty after cleaning")
        return

    # Normalize innovation per instrument for comparability
    inst_std = df.groupby('instrument')['innovation'].std().replace(0, np.nan)
    df['innovation_norm'] = df.apply(lambda r: r['innovation'] / inst_std.get(r['instrument'], np.nan), axis=1)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['innovation_norm', 'fsoi'])
    if df.empty:
        print("  Skipping: empty after normalization")
        return

    top = df.groupby('instrument').size().sort_values(ascending=False).head(top_instruments).index.tolist()
    n = len(top)
    nrows = 2
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), squeeze=False)

    for i, inst in enumerate(top[: nrows * ncols]):
        ax = axes[i // ncols][i % ncols]
        sub = df[df['instrument'] == inst]
        if len(sub) < 100:
            ax.scatter(sub['innovation_norm'], sub['fsoi'], s=2, alpha=0.3)
        else:
            ax.hexbin(sub['innovation_norm'], sub['fsoi'], gridsize=60, bins='log', cmap='viridis')
        ax.set_title(inst)
        ax.set_xlabel('Normalized Innovation (δx / σ)')
        ax.set_ylabel('FSOI (δx ⊙ (ga+gb))')
        ax.grid(True, alpha=0.2)

    # Hide unused axes
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis('off')

    plt.tight_layout()
    out = output_dir / 'innovation_vs_fsoi_scatter.png'
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out}")


def plot_time_series(df_inst, output_dir):
    """Time series of impact evolution."""
    print("Creating time series plot...")

    # Extract dates from bin names
    if 'curr_bin' not in df_inst.columns:
        print("  Skipping: No 'curr_bin' column found")
        return

    try:
        df_inst['date'] = pd.to_datetime(df_inst['curr_bin'].str[:10])
    except Exception as e:
        print(f"  Skipping: Could not parse dates - {e}")
        return

    # Daily aggregates by instrument
    daily = df_inst.groupby(['date', 'instrument'])['sum_impact'].sum().unstack(fill_value=0)

    fig, ax = plt.subplots(figsize=(14, 6))
    daily.plot(ax=ax, linewidth=2, alpha=0.8)

    ax.set_xlabel('Date')
    ax.set_ylabel('Daily FSOI Impact')
    ax.set_title('Observation Impact Time Series')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.grid(True, alpha=0.3)
    ax.legend(title='Instrument', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(output_dir / 'impact_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'impact_timeseries.png'}")


def plot_channel_heatmap(df_ch, output_dir, top_n=10):
    """Heatmap of channel-level impacts."""
    print("Creating channel heatmap...")

    if df_ch.empty:
        print("  Skipping: No channel data")
        return

    # Get top instruments by total impact
    top_insts = df_ch.groupby('instrument')['sum_impact'].sum().abs().nlargest(top_n).index
    df_top = df_ch[df_ch['instrument'].isin(top_insts)]

    # Pivot to matrix format
    pivot = df_top.groupby(['instrument', 'channel'])['mean_impact'].mean().unstack(fill_value=0)

    if pivot.empty:
        print("  Skipping: No data after filtering")
        return

    fig, ax = plt.subplots(figsize=(12, 8))

    # Use diverging colormap centered at zero
    vmax = np.abs(pivot.values).max()

    # Create heatmap using imshow
    im = ax.imshow(
        pivot.values,
        cmap='RdBu_r',
        aspect='auto',
        vmin=-vmax,
        vmax=vmax,
        interpolation='nearest'
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticklabels(pivot.index)

    # Rotate x-axis labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Mean FSOI Impact')

    ax.set_xlabel('Channel')
    ax.set_ylabel('Instrument')
    ax.set_title(f'Channel-Level FSOI Impact (Top {len(pivot)} Instruments)')

    plt.tight_layout()
    plt.savefig(output_dir / 'channel_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'channel_heatmap.png'}")


def plot_positive_negative_scatter(df_inst, output_dir):
    """Scatter plot of positive fraction vs total impact."""
    print("Creating positive/negative scatter plot...")

    # Aggregate by instrument
    summary = df_inst.groupby('instrument').agg({
        'sum_impact': 'sum',
        'positive_frac': 'mean',
        'n_observations': 'sum'
    }).reset_index()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Size by number of observations
    sizes = summary['n_observations'] / summary['n_observations'].max() * 500

    scatter = ax.scatter(
        summary['positive_frac'],
        summary['sum_impact'],
        s=sizes,
        alpha=0.6,
        c=summary['sum_impact'],
        cmap='RdYlGn_r',
        edgecolors='black',
        linewidth=0.5,
    )

    # Add labels for each point
    for idx, row in summary.iterrows():
        ax.annotate(
            row['instrument'],
            (row['positive_frac'], row['sum_impact']),
            fontsize=8,
            alpha=0.7,
        )

    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.axvline(x=0.5, color='gray', linestyle=':', linewidth=1)

    ax.set_xlabel('Positive Impact Fraction (higher = more detrimental)')
    ax.set_ylabel('Total FSOI Impact (negative = helpful)')
    ax.set_title('Observation Impact: Magnitude vs. Sign Distribution')
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Total Impact')

    # Add quadrant labels
    ax.text(0.25, ax.get_ylim()[1]*0.9, 'Mostly Helpful\n(Negative Impact)',
            ha='center', fontsize=10, alpha=0.5)
    ax.text(0.75, ax.get_ylim()[1]*0.9, 'Mostly Harmful\n(Positive Impact)',
            ha='center', fontsize=10, alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_dir / 'positive_negative_scatter.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'positive_negative_scatter.png'}")


def plot_satellite_channel_impacts(df_ch, output_dir):
    """Bar chart showing impact of individual satellite channels."""
    print("Creating satellite channel impact plot...")

    # Filter for satellite instruments only
    satellite_instruments = ['atms', 'amsua', 'ssmis', 'seviri', 'seviri_asr', 'seviri_csr', 'avhrr', 'iasi', 'cris', 'airs', 'mhs', 'amsub']
    df_sat = df_ch[df_ch['instrument'].str.lower().isin(satellite_instruments)].copy()

    if df_sat.empty:
        print("  Skipping: No satellite channel data found")
        return

    # Aggregate by instrument and channel
    channel_impacts = df_sat.groupby(['instrument', 'channel'])['sum_impact'].sum().reset_index()

    # Create instrument_channel label
    channel_impacts['label'] = channel_impacts['instrument'] + '_ch' + channel_impacts['channel'].astype(str)

    # Sort by impact within each instrument
    channel_impacts = channel_impacts.sort_values(['instrument', 'sum_impact'])

    # Create figure with subplots for each instrument
    instruments = sorted(channel_impacts['instrument'].unique())
    n_instruments = len(instruments)

    if n_instruments == 0:
        print("  Skipping: No satellite instruments found")
        return

    # Calculate figure size dynamically
    fig_height = max(8, n_instruments * 3)
    fig, axes = plt.subplots(n_instruments, 1, figsize=(12, fig_height))

    # Handle single instrument case
    if n_instruments == 1:
        axes = [axes]

    for idx, inst in enumerate(instruments):
        ax = axes[idx]
        inst_data = channel_impacts[channel_impacts['instrument'] == inst].copy()

        # Get impacts and colors
        impacts = inst_data['sum_impact'].values
        channels = inst_data['channel'].values
        colors = ['green' if x < 0 else 'red' for x in impacts]

        # Create bar chart
        bars = ax.barh(range(len(channels)), impacts, color=colors, alpha=0.7)
        ax.set_yticks(range(len(channels)))
        ax.set_yticklabels([f'Ch{int(c)}' for c in channels], fontsize=8)

        ax.set_xlabel('FSOI Impact (negative = helpful)', fontsize=10)
        ax.set_ylabel('Channel', fontsize=10)
        ax.set_title(f'{inst.upper()} Channel Impacts', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
        ax.grid(True, alpha=0.3, axis='x')

        # Add value labels for top/bottom channels
        max_idx = impacts.argmax()
        min_idx = impacts.argmin()
        ax.text(
            impacts[max_idx],
            max_idx,
            f' {impacts[max_idx]:.1f}',
            va='center',
            fontsize=8,
            fontweight='bold',
        )
        ax.text(
            impacts[min_idx],
            min_idx,
            f' {impacts[min_idx]:.1f}',
            va='center',
            fontsize=8,
            fontweight='bold',
        )

    plt.tight_layout()
    plt.savefig(output_dir / 'satellite_channel_impacts.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'satellite_channel_impacts.png'}")

    # Also create a summary plot showing top beneficial and detrimental channels across all satellites
    print("Creating top channels comparison plot...")

    # Get top 20 most beneficial and detrimental channels
    top_beneficial = channel_impacts.nsmallest(20, 'sum_impact')
    top_detrimental = channel_impacts.nlargest(20, 'sum_impact')
    top_channels = pd.concat([top_beneficial, top_detrimental]).drop_duplicates()
    top_channels = top_channels.sort_values('sum_impact')

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ['green' if x < 0 else 'red' for x in top_channels['sum_impact'].values]

    y_pos = range(len(top_channels))
    ax.barh(y_pos, top_channels['sum_impact'].values, color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_channels['label'].values, fontsize=9)

    ax.set_xlabel('FSOI Impact (negative = helpful)', fontsize=11)
    ax.set_ylabel('Satellite Channel', fontsize=11)
    ax.set_title('Top Beneficial and Detrimental Satellite Channels', fontsize=13, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5)
    ax.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(output_dir / 'top_satellite_channels.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_dir / 'top_satellite_channels.png'}")


def create_summary_table(df_inst, output_dir):
    """Create summary statistics table."""
    print("Creating summary table...")

    agg_dict = {
        'sum_impact': ['sum', 'mean', 'std'],
        'mean_impact': 'mean',
        'positive_frac': 'mean',
        'n_observations': 'sum',
    }

    optional_aggs = {
        'innovation_abs_mean': 'mean',
        'innovation_rms': 'mean',
        'gradient_abs_mean': 'mean',
        'gradient_rms': 'mean',
        'alignment_cosine': 'mean',
        'alignment_frac': 'mean',
    }

    for col, agg in optional_aggs.items():
        if col in df_inst.columns:
            agg_dict[col] = agg

    summary = df_inst.groupby('instrument').agg(agg_dict).reset_index()

    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    # Sort by total impact (absolute value)
    summary['abs_impact'] = summary['sum_impact_sum'].abs()
    summary = summary.sort_values('abs_impact', ascending=False)
    summary = summary.drop('abs_impact', axis=1)

    # Save as CSV
    summary.to_csv(output_dir / 'summary_statistics.csv', index=False, float_format='%.6f')

    # Save as formatted text
    with open(output_dir / 'summary_statistics.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("FSOI SUMMARY STATISTICS\n")
        f.write("="*80 + "\n\n")
        f.write(summary.to_string(index=False))
        f.write("\n\n")
        f.write("Interpretation:\n")
        f.write("  - sum_impact: Total contribution (negative = helpful)\n")
        f.write("  - positive_frac: % of obs that increased error\n")
        f.write("  - Large |sum_impact|: High impact instrument\n")
        f.write("  - innovation_*: Magnitude of δx (analysis - background)\n")
        f.write("  - gradient_*: Magnitude of adjoint (ga+gb)\n")
        f.write("  - alignment_cosine: +1 aligned, -1 opposed between δx and gradient\n")

    print(f"  Saved: {output_dir / 'summary_statistics.csv'}")
    print(f"  Saved: {output_dir / 'summary_statistics.txt'}")


def main():
    parser = argparse.ArgumentParser(description="Visualize FSOI results")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to directory containing FSOI CSV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for plots (default: input_dir/plots)",
    )

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output) if args.output else (input_dir / "plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*80)
    print("FSOI VISUALIZATION")
    print("="*80)
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print("="*80 + "\n")

    # Load data
    print("Loading FSOI results...")

    # Check for CSV files in csv/ subdirectory first, then root
    csv_dir = input_dir / "csv"
    if csv_dir.exists():
        inst_file = csv_dir / "fsoi_by_instrument.csv"
        ch_file = csv_dir / "fsoi_by_channel.csv"
        print(f"  Looking in csv/ subdirectory...")
    else:
        inst_file = input_dir / "fsoi_by_instrument.csv"
        ch_file = input_dir / "fsoi_by_channel.csv"
        print(f"  Looking in root directory...")

    if not inst_file.exists():
        print(f"ERROR: {inst_file} not found")
        return

    df_inst = pd.read_csv(inst_file)
    print(f"  Loaded instrument data: {len(df_inst)} rows")

    df_ch = None
    if ch_file.exists():
        df_ch = pd.read_csv(ch_file)
        print(f"  Loaded channel data: {len(df_ch)} rows")
    else:
        print(f"  Channel data not found (optional)")

    scatter_file = (csv_dir / 'scatter_samples.csv') if csv_dir.exists() else (input_dir / 'scatter_samples.csv')
    df_scatter = None
    if scatter_file.exists():
        df_scatter = pd.read_csv(scatter_file)
        print(f"  Loaded scatter sample data: {len(df_scatter)} rows")

    # Create visualizations
    print("\nCreating visualizations...\n")

    plot_instrument_impacts(df_inst, output_dir)
    plot_instrument_relative_contribution(df_inst, output_dir)
    plot_time_series(df_inst, output_dir)
    plot_positive_negative_scatter(df_inst, output_dir)

    if df_ch is not None and not df_ch.empty:
        plot_channel_heatmap(df_ch, output_dir)
        plot_satellite_channel_impacts(df_ch, output_dir)
        plot_pressure_instrument_heatmap(df_ch, output_dir)
        plot_pressure_instrument_heatmaps_by_variable(df_ch, output_dir)
        plot_variable_pressure_combined_heatmap(df_ch, output_dir)

    if df_scatter is not None and not df_scatter.empty:
        plot_innovation_vs_fsoi_scatter(df_scatter, output_dir)

    create_summary_table(df_inst, output_dir)

    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print("="*80)
    print(f"\nPlots saved to: {output_dir}")
    print("\nGenerated files:")
    print("  - instrument_impacts.png")
    print("  - instrument_relative_contribution.png")
    print("  - impact_timeseries.png")
    print("  - positive_negative_scatter.png")
    print("  - channel_heatmap.png (if channel data available)")
    print("  - instrument_contribution_by_pressure_heatmap.png (if pressure stratified)")
    print("  - satellite_channel_impacts.png (satellite channels by instrument)")
    print("  - top_satellite_channels.png (top beneficial/detrimental channels)")
    print("  - summary_statistics.csv")
    print("  - summary_statistics.txt")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
