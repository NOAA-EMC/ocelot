#!/usr/bin/env python
"""
Plot gridded global maps of FSOI relative contribution.

Reads scatter_samples.csv (which now includes 'lat' and 'lon' columns),
bins observations onto a regular lat/lon grid, and produces:

  1. Total FSOI map         — signed sum per grid cell across all instruments
  2. Absolute impact map    — |FSOI| sum per cell (highlights active regions)
  3. Relative contribution  — per-cell |FSOI| as % of global total
  4. Per-instrument maps    — one map per instrument showing its contribution
  5. Beneficial fraction    — fraction of grid cells where FSOI < 0 (helpful)

Requirements:
  - scatter_samples.csv must have 'lat', 'lon', 'fsoi', 'instrument' columns.
    These are present when fsoi_inference.py is run with save_scatter_samples=true
    and the metadata includes lat/lon (enabled by default for all conventional
    and satellite instruments that store position metadata).

Usage:
    python FSOI/plot_fsoi_maps.py \
        --input  path/to/csv/scatter_samples.csv \
        --output path/to/output/maps/ \
        [--grid_deg 2.5] \
        [--title "Surface Obs FSOI"] \
        [--instruments atms amsua radiosonde] \
        [--min_count 5]

Output PNGs:
    fsoi_total_map.png
    fsoi_absolute_map.png
    fsoi_relative_contribution_map.png
    fsoi_beneficial_fraction_map.png
    fsoi_map_<instrument>.png  (one per instrument)
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.colors as mcolors  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Try to import cartopy for proper geographic projections; fall back to plain
# pcolormesh on a rectangular grid if cartopy is not installed.
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    _HAS_CARTOPY = True
except ImportError:
    _HAS_CARTOPY = False


# ---------------------------------------------------------------------------
# Gridding helpers
# ---------------------------------------------------------------------------

def _build_grid(lat: np.ndarray, lon: np.ndarray, values: np.ndarray,
                grid_deg: float = 2.5) -> tuple:
    """Bin (lat, lon, value) onto a regular grid.

    Returns (lat_centers, lon_centers, grid_sum, grid_count).
    Cells with no observations have grid_sum=0, grid_count=0.
    """
    lat_edges = np.arange(-90, 90 + grid_deg, grid_deg)
    lon_edges = np.arange(-180, 180 + grid_deg, grid_deg)

    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    # Filter NaN coordinates
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(values)
    lat_v, lon_v, val_v = lat[valid], lon[valid], values[valid]

    if val_v.size == 0:
        nlat, nlon = len(lat_centers), len(lon_centers)
        return lat_centers, lon_centers, np.zeros((nlat, nlon)), np.zeros((nlat, nlon))

    lat_idx = np.clip(np.digitize(lat_v, lat_edges) - 1, 0, len(lat_centers) - 1)
    lon_idx = np.clip(np.digitize(lon_v, lon_edges) - 1, 0, len(lon_centers) - 1)

    nlat, nlon = len(lat_centers), len(lon_centers)
    grid_sum = np.zeros((nlat, nlon))
    grid_count = np.zeros((nlat, nlon))

    np.add.at(grid_sum, (lat_idx, lon_idx), val_v)
    np.add.at(grid_count, (lat_idx, lon_idx), 1)

    return lat_centers, lon_centers, grid_sum, grid_count


def _mask_low_count(grid: np.ndarray, count: np.ndarray,
                    min_count: int) -> np.ndarray:
    """Return masked array where cells with fewer than min_count obs are hidden."""
    masked = np.ma.array(grid, mask=(count < min_count))
    return masked


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _make_axes(figsize=(14, 6)):
    """Create figure + axes, using cartopy if available."""
    if _HAS_CARTOPY:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(1, 1, 1, projection=ccrs.Robinson())
        ax.add_feature(cfeature.COASTLINE, linewidth=0.5, edgecolor='0.3')
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='0.5')
        ax.set_global()
        return fig, ax, True
    else:
        fig, ax = plt.subplots(figsize=figsize)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
        return fig, ax, False


def _pcolormesh(ax, lon_centers, lat_centers, data, cmap, vmin, vmax,
                use_cartopy: bool):
    """Wrapper that calls pcolormesh with or without cartopy transform."""
    LON, LAT = np.meshgrid(lon_centers, lat_centers)
    if use_cartopy:
        return ax.pcolormesh(LON, LAT, data, cmap=cmap, vmin=vmin, vmax=vmax,
                             transform=ccrs.PlateCarree(), shading='auto')
    else:
        return ax.pcolormesh(LON, LAT, data, cmap=cmap, vmin=vmin, vmax=vmax,
                             shading='auto')


def _save(fig, path: Path, title: str):
    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Individual map functions
# ---------------------------------------------------------------------------

def plot_total_fsoi(df: pd.DataFrame, out_dir: Path, title: str,
                    grid_deg: float, min_count: int):
    """Signed total FSOI per grid cell (negative = net beneficial)."""
    lats = df['lat'].values
    lons = df['lon'].values
    fsoi = df['fsoi'].values

    lat_c, lon_c, g_sum, g_cnt = _build_grid(lats, lons, fsoi, grid_deg)
    masked = _mask_low_count(g_sum, g_cnt, min_count)

    vmax = np.nanpercentile(np.abs(g_sum[g_cnt >= min_count]), 95) if (g_cnt >= min_count).any() else 1.0
    vmax = max(vmax, 1e-10)

    fig, ax, use_cartopy = _make_axes()
    pcm = _pcolormesh(ax, lon_c, lat_c, masked, 'RdBu_r', -vmax, vmax, use_cartopy)
    plt.colorbar(pcm, ax=ax, label='Sum FSOI (neg = helpful)', shrink=0.6, pad=0.02)
    _save(fig, out_dir / 'fsoi_total_map.png', f'{title} — Signed FSOI per {grid_deg}° cell')


def plot_absolute_fsoi(df: pd.DataFrame, out_dir: Path, title: str,
                       grid_deg: float, min_count: int):
    """|FSOI| sum per grid cell — highlights regions where obs have most impact."""
    lats = df['lat'].values
    lons = df['lon'].values
    abs_fsoi = np.abs(df['fsoi'].values)

    lat_c, lon_c, g_sum, g_cnt = _build_grid(lats, lons, abs_fsoi, grid_deg)
    masked = _mask_low_count(g_sum, g_cnt, min_count)

    vmax = np.nanpercentile(g_sum[g_cnt >= min_count], 95) if (g_cnt >= min_count).any() else 1.0
    vmax = max(vmax, 1e-10)

    fig, ax, use_cartopy = _make_axes()
    pcm = _pcolormesh(ax, lon_c, lat_c, masked, 'YlOrRd', 0, vmax, use_cartopy)
    plt.colorbar(pcm, ax=ax, label='|FSOI| sum', shrink=0.6, pad=0.02)
    _save(fig, out_dir / 'fsoi_absolute_map.png', f'{title} — |FSOI| per {grid_deg}° cell')


def plot_relative_contribution(df: pd.DataFrame, out_dir: Path, title: str,
                               grid_deg: float, min_count: int):
    """Per-cell |FSOI| as % of the global total — relative contribution map."""
    lats = df['lat'].values
    lons = df['lon'].values
    abs_fsoi = np.abs(df['fsoi'].values)

    lat_c, lon_c, g_sum, g_cnt = _build_grid(lats, lons, abs_fsoi, grid_deg)

    global_total = g_sum.sum()
    if global_total == 0:
        print("  [SKIP] relative contribution: global total is zero")
        return

    rel = 100.0 * g_sum / global_total
    masked = _mask_low_count(rel, g_cnt, min_count)

    vmax = np.nanpercentile(rel[g_cnt >= min_count], 98) if (g_cnt >= min_count).any() else 1.0
    vmax = max(vmax, 1e-10)

    fig, ax, use_cartopy = _make_axes()
    pcm = _pcolormesh(ax, lon_c, lat_c, masked, 'viridis', 0, vmax, use_cartopy)
    plt.colorbar(pcm, ax=ax, label='% of global |FSOI|', shrink=0.6, pad=0.02)
    _save(fig, out_dir / 'fsoi_relative_contribution_map.png',
          f'{title} — Relative |FSOI| contribution (%) per {grid_deg}° cell')


def plot_beneficial_fraction(df: pd.DataFrame, out_dir: Path, title: str,
                             grid_deg: float, min_count: int):
    """Fraction of helpful (FSOI < 0) observations per grid cell.

    Interpretation:
      - Blue (>0.7): most obs in this cell are improving the forecast
      - Red (<0.3):  most obs are making it worse (potential quality issue)
    """
    lats = df['lat'].values
    lons = df['lon'].values
    helpful = (df['fsoi'].values < 0).astype(float)

    lat_c, lon_c, g_sum, g_cnt = _build_grid(lats, lons, helpful, grid_deg)
    frac = np.where(g_cnt > 0, g_sum / np.maximum(g_cnt, 1), np.nan)
    masked = _mask_low_count(frac, g_cnt, min_count)

    fig, ax, use_cartopy = _make_axes()
    cmap = plt.cm.RdYlBu
    pcm = _pcolormesh(ax, lon_c, lat_c, masked, cmap, 0.0, 1.0, use_cartopy)
    cbar = plt.colorbar(pcm, ax=ax, label='Fraction helpful (FSOI < 0)', shrink=0.6, pad=0.02)
    cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])
    _save(fig, out_dir / 'fsoi_beneficial_fraction_map.png',
          f'{title} — Helpful-obs fraction per {grid_deg}° cell')


def plot_instrument_maps(df: pd.DataFrame, out_dir: Path, title: str,
                         grid_deg: float, min_count: int,
                         instruments: list = None):
    """One signed FSOI map per instrument."""
    all_instruments = sorted(df['instrument'].dropna().unique())
    if instruments:
        all_instruments = [i for i in instruments if i in all_instruments]

    # Shared vmax across all instruments for fair comparison
    abs_vals = np.abs(df['fsoi'].values[np.isfinite(df['fsoi'].values)])
    global_vmax = float(np.percentile(abs_vals, 95)) if len(abs_vals) > 0 else 1.0
    global_vmax = max(global_vmax, 1e-10)

    for inst in all_instruments:
        sub = df[df['instrument'] == inst]
        if len(sub) < min_count:
            print(f"  [SKIP] {inst}: only {len(sub)} scatter points (< {min_count})")
            continue

        lat_c, lon_c, g_sum, g_cnt = _build_grid(
            sub['lat'].values, sub['lon'].values, sub['fsoi'].values, grid_deg)
        masked = _mask_low_count(g_sum, g_cnt, min_count)

        fig, ax, use_cartopy = _make_axes()
        pcm = _pcolormesh(ax, lon_c, lat_c, masked, 'RdBu_r',
                          -global_vmax, global_vmax, use_cartopy)
        plt.colorbar(pcm, ax=ax, label='Sum FSOI (neg = helpful)', shrink=0.6, pad=0.02)

        n_obs = int(sub['lat'].notna().sum())
        frac_helpful = float((sub['fsoi'] < 0).mean())
        info = f"n={n_obs:,}  {frac_helpful:.0%} helpful"

        _save(fig, out_dir / f'fsoi_map_{inst}.png',
              f'{title} — {inst}  [{info}]  per {grid_deg}° cell')


# ---------------------------------------------------------------------------
# Multi-variable map (one row per variable)
# ---------------------------------------------------------------------------

def plot_per_variable_maps(df: pd.DataFrame, out_dir: Path, title: str,
                           grid_deg: float, min_count: int):
    """If 'target_variable' column is present, plot one global map per variable."""
    if 'target_variable' not in df.columns:
        return
    variables = sorted(df['target_variable'].dropna().unique())
    if not variables:
        return

    n_vars = len(variables)
    if _HAS_CARTOPY:
        fig, axes = plt.subplots(n_vars, 1,
                                 figsize=(14, 4 * n_vars),
                                 subplot_kw={'projection': ccrs.Robinson()})
    else:
        fig, axes = plt.subplots(n_vars, 1, figsize=(14, 4 * n_vars))

    if n_vars == 1:
        axes = [axes]

    abs_all = np.abs(df['fsoi'].values[np.isfinite(df['fsoi'].values)])
    vmax = float(np.percentile(abs_all, 95)) if len(abs_all) > 0 else 1.0
    vmax = max(vmax, 1e-10)

    for ax, var in zip(axes, variables):
        sub = df[df['target_variable'] == var]
        lat_c, lon_c, g_sum, g_cnt = _build_grid(
            sub['lat'].values, sub['lon'].values, sub['fsoi'].values, grid_deg)
        masked = _mask_low_count(g_sum, g_cnt, min_count)
        LON, LAT = np.meshgrid(lon_c, lat_c)

        if _HAS_CARTOPY:
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor='0.3')
            ax.set_global()
            pcm = ax.pcolormesh(LON, LAT, masked, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax,
                                transform=ccrs.PlateCarree(), shading='auto')
        else:
            pcm = ax.pcolormesh(LON, LAT, masked, cmap='RdBu_r',
                                vmin=-vmax, vmax=vmax, shading='auto')
            ax.set_xlim(-180, 180)
            ax.set_ylim(-90, 90)

        plt.colorbar(pcm, ax=ax, label='FSOI', shrink=0.6, pad=0.02)
        ax.set_title(var, fontsize=10)

    fig.suptitle(f'{title} — FSOI by target variable', fontsize=11, y=1.01)
    fig.tight_layout()
    out_path = out_dir / 'fsoi_per_variable_maps.png'
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot gridded FSOI maps from scatter_samples.csv")
    parser.add_argument('--input', required=True,
                        help="Path to scatter_samples.csv (must contain lat, lon columns)")
    parser.add_argument('--output', default='fsoi_maps',
                        help="Output directory for PNG files")
    parser.add_argument('--grid_deg', type=float, default=2.5,
                        help="Grid resolution in degrees (default: 2.5)")
    parser.add_argument('--title', default='FSOI',
                        help="Plot title prefix")
    parser.add_argument('--instruments', nargs='*', default=None,
                        help="Subset of instruments for per-instrument maps (default: all)")
    parser.add_argument('--min_count', type=int, default=5,
                        help="Min observations per grid cell to show (default: 5)")
    args = parser.parse_args()

    csv_path = Path(args.input)
    if not csv_path.is_file():
        print(f"ERROR: Input file not found: {csv_path}")
        sys.exit(1)

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path)
    print(f"  {len(df):,} rows loaded")
    print(f"  Columns: {list(df.columns)}")

    if 'lat' not in df.columns or 'lon' not in df.columns:
        print("ERROR: scatter_samples.csv does not have 'lat' and 'lon' columns.")
        print("  Re-run fsoi_inference.py — lat/lon are saved when the batch")
        print("  metadata includes position information (enabled by default).")
        sys.exit(1)

    if 'fsoi' not in df.columns:
        print("ERROR: 'fsoi' column missing from input CSV.")
        sys.exit(1)

    # Drop rows missing coordinates or FSOI
    before = len(df)
    df = df.dropna(subset=['lat', 'lon', 'fsoi'])
    if len(df) < before:
        print(f"  Dropped {before - len(df)} rows with missing lat/lon/fsoi")

    if len(df) == 0:
        print("ERROR: No valid rows after filtering. Check that lat/lon are populated.")
        sys.exit(1)

    n_instruments = df['instrument'].nunique() if 'instrument' in df.columns else 0
    print(f"  Instruments in data: {sorted(df['instrument'].unique()) if 'instrument' in df.columns else []}")
    print(f"  Lat range: [{df['lat'].min():.1f}, {df['lat'].max():.1f}]")
    print(f"  Lon range: [{df['lon'].min():.1f}, {df['lon'].max():.1f}]")
    print(f"  FSOI range: [{df['fsoi'].min():.3e}, {df['fsoi'].max():.3e}]")
    print(f"  Grid resolution: {args.grid_deg}°")
    print(f"  Cartopy available: {_HAS_CARTOPY}")
    print()

    print("Generating maps...")

    print("  [1/6] Total signed FSOI map")
    plot_total_fsoi(df, out_dir, args.title, args.grid_deg, args.min_count)

    print("  [2/6] Absolute |FSOI| map")
    plot_absolute_fsoi(df, out_dir, args.title, args.grid_deg, args.min_count)

    print("  [3/6] Relative contribution map")
    plot_relative_contribution(df, out_dir, args.title, args.grid_deg, args.min_count)

    print("  [4/6] Beneficial fraction map")
    plot_beneficial_fraction(df, out_dir, args.title, args.grid_deg, args.min_count)

    print(f"  [5/6] Per-instrument maps ({n_instruments} instruments)")
    plot_instrument_maps(df, out_dir, args.title, args.grid_deg, args.min_count,
                         instruments=args.instruments)

    print("  [6/6] Per-variable maps")
    plot_per_variable_maps(df, out_dir, args.title, args.grid_deg, args.min_count)

    print(f"\nDone. Output: {out_dir}")


if __name__ == '__main__':
    main()
