#!/usr/bin/env python
"""
Create the ATMS physical interpretation case-study figure.

The figure combines:
  1. Pair-level ATMS FSOI maps for selected channels.
  2. Innovation maps using delta x = observation minus background.
  3. Optional mesh-space OSE forecast-error-difference fields saved by
     fsoi_inference.py --ose_save_spatial_fields.

If the spatial OSE field is not available, the figure explicitly marks panel
(c) as unavailable instead of inferring a spatial map from scalar OSE CSVs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def _parse_bin_time(value: str) -> str:
    value = str(value)
    if value.startswith("bin") and len(value) >= 13 and value[3:13].isdigit():
        return f"{value[3:7]}-{value[7:9]}-{value[9:11]} {value[11:13]}Z"
    return value


def _read_pair_ose(ose_csv: Path | None, pair_idx: int) -> dict:
    if ose_csv is None or not ose_csv.is_file():
        return {}
    df = pd.read_csv(ose_csv)
    if "pair_idx" not in df.columns:
        return {}
    sub = df[df["pair_idx"] == pair_idx]
    if sub.empty:
        return {}
    return sub.iloc[0].to_dict()


def _filter_scatter(scatter_csv: Path, pair_idx: int, channels: list[int]) -> pd.DataFrame:
    usecols = [
        "instrument", "channel", "innovation", "fsoi",
        "lat", "lon", "pair_idx", "lead_step",
        "target_variable", "p_hpa",
    ]
    cols = pd.read_csv(scatter_csv, nrows=0).columns
    usecols = [c for c in usecols if c in cols]
    if not {"instrument", "channel", "innovation", "fsoi", "lat", "lon", "pair_idx"}.issubset(usecols):
        missing = {"instrument", "channel", "innovation", "fsoi", "lat", "lon", "pair_idx"} - set(usecols)
        raise ValueError(f"scatter CSV is missing required columns: {sorted(missing)}")

    chunks = []
    for chunk in pd.read_csv(scatter_csv, usecols=usecols, chunksize=500_000):
        sub = chunk[
            (chunk["instrument"].astype(str).str.lower() == "atms")
            & (chunk["pair_idx"].astype(int) == int(pair_idx))
            & (chunk["channel"].astype(int).isin(channels))
        ]
        if not sub.empty:
            chunks.append(sub.copy())
    if not chunks:
        raise ValueError(
            f"No ATMS scatter samples found for pair_idx={pair_idx}, channels={channels}"
        )
    return pd.concat(chunks, ignore_index=True)


def _grid_sum_or_mean(
    df: pd.DataFrame,
    value_col: str,
    grid_deg: float,
    reducer: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat_edges = np.arange(-90.0, 90.0 + grid_deg, grid_deg)
    lon_edges = np.arange(-180.0, 180.0 + grid_deg, grid_deg)
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])

    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    val = df[value_col].to_numpy(dtype=float)
    valid = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(val)
    lat, lon, val = lat[valid], lon[valid], val[valid]

    nlat, nlon = len(lat_centers), len(lon_centers)
    grid_sum = np.zeros((nlat, nlon), dtype=float)
    grid_count = np.zeros((nlat, nlon), dtype=float)
    if val.size:
        ilat = np.clip(np.digitize(lat, lat_edges) - 1, 0, nlat - 1)
        ilon = np.clip(np.digitize(lon, lon_edges) - 1, 0, nlon - 1)
        np.add.at(grid_sum, (ilat, ilon), val)
        np.add.at(grid_count, (ilat, ilon), 1)

    if reducer == "mean":
        grid = np.divide(
            grid_sum,
            grid_count,
            out=np.full_like(grid_sum, np.nan),
            where=grid_count > 0,
        )
    else:
        grid = grid_sum
        grid[grid_count == 0] = np.nan
    return lat_centers, lon_centers, grid, grid_count


def _grid_points(
    lat: np.ndarray,
    lon: np.ndarray,
    values: np.ndarray,
    grid_deg: float,
    reducer: str = "mean",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    df = pd.DataFrame({"lat": lat, "lon": lon, "value": values})
    return _grid_sum_or_mean(df, "value", grid_deg, reducer)


def _robust_abs_scale(values: list[np.ndarray], pct: float = 98.0) -> float:
    flat = []
    for arr in values:
        v = np.asarray(arr, dtype=float)
        v = v[np.isfinite(v)]
        if v.size:
            flat.append(np.abs(v))
    if not flat:
        return 1.0
    scale = float(np.nanpercentile(np.concatenate(flat), pct))
    return max(scale, 1e-12)


def _draw_map(ax, lon_c, lat_c, grid, cmap, scale, title, count=None, min_count=1):
    data = np.asarray(grid, dtype=float).copy()
    if count is not None:
        data[np.asarray(count) < min_count] = np.nan
    mesh = ax.pcolormesh(lon_c, lat_c, data, cmap=cmap, vmin=-scale, vmax=scale, shading="auto")
    ax.set_title(title, fontsize=9)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks([-180, 0, 180])
    ax.set_yticks([-60, 0, 60])
    ax.tick_params(labelsize=7, length=2)
    ax.grid(True, color="0.88", linewidth=0.4)
    return mesh


def _load_spatial_npz(path: Path | None) -> dict | None:
    if path is None or not path.is_file():
        return None
    data = np.load(path, allow_pickle=True)
    return {k: data[k] for k in data.files}


def _as_text(value) -> str:
    arr = np.asarray(value)
    if arr.shape == ():
        return str(arr.item())
    return ",".join(map(str, arr.tolist()))


def make_figure(
    scatter_csv: Path,
    output: Path,
    ose_csv: Path | None,
    spatial_npz: Path | None,
    pair_idx: int,
    channels: list[int],
    grid_deg: float,
    min_count: int,
    mesh_channel: int,
    combine_mesh_channels: bool,
) -> None:
    df = _filter_scatter(scatter_csv, pair_idx, channels)
    ose = _read_pair_ose(ose_csv, pair_idx)
    spatial = _load_spatial_npz(spatial_npz)

    fsoi_grids = {}
    innov_grids = {}
    for ch in channels:
        sub = df[df["channel"].astype(int) == int(ch)]
        fsoi_grids[ch] = _grid_sum_or_mean(sub, "fsoi", grid_deg, "sum")
        innov_grids[ch] = _grid_sum_or_mean(sub, "innovation", grid_deg, "mean")

    fsoi_scale = _robust_abs_scale([v[2] for v in fsoi_grids.values()])
    innov_scale = _robust_abs_scale([v[2] for v in innov_grids.values()])

    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(
        4, 4,
        height_ratios=[0.16, 1.0, 1.0, 1.12],
        width_ratios=[1.0, 1.0, 1.0, 0.95],
        hspace=0.55,
        wspace=0.18,
    )

    title_ax = fig.add_subplot(gs[0, :])
    title_ax.axis("off")
    prev_bin = ose.get("prev_bin", "")
    curr_bin = ose.get("curr_bin", "")
    if prev_bin or curr_bin:
        window = f"{_parse_bin_time(prev_bin)} to {_parse_bin_time(curr_bin)}"
    else:
        window = f"pair {pair_idx}"
    title_ax.text(
        0.0, 0.72,
        f"ATMS physical case study, pair {pair_idx}: {window}",
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="center",
    )
    title_ax.text(
        0.0, 0.18,
        "Positive FSOI is detrimental. Innovation uses delta x = observation minus background.",
        fontsize=9,
        color="0.35",
        ha="left",
        va="center",
    )

    fsoi_mesh = None
    innov_mesh = None
    fsoi_axes = []
    innov_axes = []
    for i, ch in enumerate(channels):
        ax = fig.add_subplot(gs[1, i])
        fsoi_axes.append(ax)
        lat_c, lon_c, grid, count = fsoi_grids[ch]
        fsoi_mesh = _draw_map(
            ax, lon_c, lat_c, grid, "RdBu_r", fsoi_scale,
            f"ATMS channel {ch}: FSOI sum",
            count=count,
            min_count=min_count,
        )

        ax = fig.add_subplot(gs[2, i])
        innov_axes.append(ax)
        lat_c, lon_c, grid, count = innov_grids[ch]
        innov_mesh = _draw_map(
            ax, lon_c, lat_c, grid, "RdBu_r", innov_scale,
            f"ATMS channel {ch}: innovation mean",
            count=count,
            min_count=min_count,
        )

    info_ax = fig.add_subplot(gs[1:3, 3])
    info_ax.axis("off")
    info_lines = ["OSE scalar check"]
    if ose:
        for label, key in [
            ("Full error", "ea_control"),
            ("Denied error", "ea_denied"),
            ("OSE impact", "ose_impact"),
            ("FSOI predicted", "fsoi_predicted"),
            ("Sign agree", "sign_agree"),
        ]:
            if key in ose and pd.notna(ose[key]):
                val = ose[key]
                if isinstance(val, (float, int, np.floating)):
                    info_lines.append(f"{label}: {float(val):+.4g}")
                else:
                    info_lines.append(f"{label}: {val}")
    else:
        info_lines.append("No OSE CSV row found for this pair.")
    info_lines.extend([
        "",
        "Moist/convective proxy",
        "No precipitation or cloud proxy is inferred here.",
        f"ATMS channel {channels[-1]} innovation can be read only as",
        "a water-vapor-sensitive context field.",
    ])
    info_ax.text(
        0.02, 0.98, "\n".join(info_lines),
        ha="left", va="top", fontsize=10,
        bbox=dict(facecolor="white", edgecolor="0.3", boxstyle="square,pad=0.65"),
    )

    if fsoi_mesh is not None:
        cb = fig.colorbar(
            fsoi_mesh, ax=fsoi_axes, orientation="horizontal",
            fraction=0.04, pad=0.10,
        )
        cb.set_label("FSOI grid-cell signed sum", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    if innov_mesh is not None:
        cb = fig.colorbar(
            innov_mesh, ax=innov_axes, orientation="horizontal",
            fraction=0.04, pad=0.10,
        )
        cb.set_label("Innovation mean (observation - background)", fontsize=8)
        cb.ax.tick_params(labelsize=7)

    ose_ax = fig.add_subplot(gs[3, 0:3])
    if spatial is not None and "error_diff" in spatial:
        lat = np.asarray(spatial.get("lat"), dtype=float)
        lon = np.asarray(spatial.get("lon"), dtype=float)
        err = np.asarray(spatial["error_diff"], dtype=float)
        names = spatial.get("channel_names", np.asarray([]))
        pressure = spatial.get("mesh_pressure_hpa", np.asarray(np.nan))
        if combine_mesh_channels:
            vals = np.nanmean(err, axis=1)
            label = "valid-channel mean"
        else:
            ch_idx = max(0, min(int(mesh_channel), err.shape[1] - 1))
            vals = err[:, ch_idx]
            label = str(names[ch_idx]) if len(names) > ch_idx else f"channel {ch_idx}"
        lat_c, lon_c, grid, count = _grid_points(lat, lon, vals, grid_deg, reducer="mean")
        scale = _robust_abs_scale([grid])
        mesh = _draw_map(
            ose_ax, lon_c, lat_c, grid, "RdBu_r", scale,
            f"Full minus ATMS-denied error difference ({label})",
            count=count,
            min_count=1,
        )
        pressure_text = ""
        try:
            pval = float(np.asarray(pressure).item())
            if np.isfinite(pval):
                pressure_text = f", {pval:.0f} hPa"
        except Exception:
            pass
        ose_ax.text(
            0.01, 0.02,
            "Positive means the ATMS-denied run has lower local squared error"
            + pressure_text,
            transform=ose_ax.transAxes,
            fontsize=8,
            ha="left",
            va="bottom",
            bbox=dict(facecolor="white", alpha=0.82, edgecolor="none"),
        )
        cb = fig.colorbar(mesh, ax=ose_ax, orientation="horizontal", pad=0.12, fraction=0.06)
        cb.set_label("Full/control squared error minus ATMS-denied squared error", fontsize=8)
        cb.ax.tick_params(labelsize=7)
    else:
        ose_ax.axis("off")
        ose_ax.text(
            0.5, 0.55,
            "Spatial OSE error-difference field is not available.",
            ha="center",
            va="center",
            fontsize=13,
            fontweight="bold",
        )
        ose_ax.text(
            0.5, 0.38,
            "Rerun fsoi_inference.py with --verification_target mesh "
            "--ose_instruments atms --ose_save_spatial_fields.",
            ha="center",
            va="center",
            fontsize=10,
            color="0.35",
            wrap=True,
        )

    note_ax = fig.add_subplot(gs[3, 3])
    note_ax.axis("off")
    note_lines = [
        "Data notes",
        f"Scatter source: {scatter_csv.name}",
        f"Grid: {grid_deg:g} degree bins",
        f"Retained ATMS samples: {len(df):,}",
    ]
    if spatial_npz is not None:
        note_lines.append(f"Spatial OSE: {spatial_npz.name if spatial_npz.is_file() else 'missing'}")
    else:
        note_lines.append("Spatial OSE: not supplied")
    if spatial is not None:
        note_lines.extend([
            f"Spatial convention: {_as_text(spatial.get('error_diff_convention', ''))}",
            f"Mesh instrument: {_as_text(spatial.get('mesh_instrument', ''))}",
        ])
    note_ax.text(
        0.02, 0.98,
        "\n".join(note_lines),
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="0.3", boxstyle="square,pad=0.65"),
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ATMS physical case-study maps")
    parser.add_argument("--scatter", type=Path, required=True,
                        help="Path to csv/scatter_samples.csv")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output PNG path")
    parser.add_argument("--ose_csv", type=Path, default=None,
                        help="Optional evaluation/ose_vs_fsoi_comparison.csv or ose_results.csv")
    parser.add_argument("--spatial_npz", type=Path, default=None,
                        help="Optional OSE spatial field .npz saved by --ose_save_spatial_fields")
    parser.add_argument("--pair_idx", type=int, default=0)
    parser.add_argument("--channels", type=int, nargs="+", default=[10, 15, 22])
    parser.add_argument("--grid_deg", type=float, default=5.0)
    parser.add_argument("--min_count", type=int, default=1)
    parser.add_argument("--mesh_channel", type=int, default=0,
                        help="Zero-based mesh target channel for the OSE map")
    parser.add_argument("--combine_mesh_channels", action="store_true",
                        help="Plot mean error difference across valid mesh channels")
    args = parser.parse_args()

    make_figure(
        scatter_csv=args.scatter,
        output=args.output,
        ose_csv=args.ose_csv,
        spatial_npz=args.spatial_npz,
        pair_idx=args.pair_idx,
        channels=args.channels,
        grid_deg=args.grid_deg,
        min_count=args.min_count,
        mesh_channel=args.mesh_channel,
        combine_mesh_channels=args.combine_mesh_channels,
    )


if __name__ == "__main__":
    main()
