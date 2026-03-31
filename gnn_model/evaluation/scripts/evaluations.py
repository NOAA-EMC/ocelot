#!/usr/bin/env python
"""evaluations.py

Evaluation utilities for OCELOT weather prediction model.

Author: Azadeh Gholoubi

This file historically focused on plotting diagnostics from CSV artifacts.
We now also support *pointwise* verification metrics directly from the new
`val_csv` format that includes init/valid timestamps.
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_FIG_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "figures"))

# Plotting dependencies are optional for metrics-only usage.
try:
    import cartopy.crs as ccrs
except Exception:  # pragma: no cover
    ccrs = None

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
except Exception:  # pragma: no cover
    plt = None
    TwoSlopeNorm = None


# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(
        description="Run evaluation plots for OCELOT model predictions"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="plots",
        choices=["plots", "metrics"],
        help=(
            "plots: generate diagnostic figures (existing behavior). "
            "metrics: compute pointwise verification metrics from val_csv using valid time + lat/lon."
        ),
    )
    parser.add_argument(
        "--init_time",
        type=str,
        default=None,
        help="Initialization time (format: YYYYMMDDHH, e.g., 2025011000)"
    )
    parser.add_argument(
        "--fhr",
        type=int,
        default=3,
        help="Forecast hour for single-lead products (default: 3). Common: 3, 6, 9, 12",
    )
    parser.add_argument(
        "--plot_all_fhrs",
        action="store_true",
        help="If set, generate per-lead plots for all 4 steps (3/6/9/12) in addition to the 12h-horizon aggregate plot.",
    )
    parser.add_argument(
        "--plot_horizon_12h",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate the 12h-horizon aggregate plot that uses all obs from leads 3/6/9/12 in one map (default: enabled).",
    )
    parser.add_argument(
        "--strict_obs_window",
        action="store_true",
        help=(
            "If set, horizon plots will raise an error when obs_time_unix falls outside the expected horizon window "
            "(relative to init_time_unix). By default this only prints a warning."
        ),
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Epoch number for training mode"
    )
    parser.add_argument(
        "--batch_idx",
        type=int,
        default=None,
        help="Batch index for training mode"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="val_csv",
        help="Directory containing CSV files"
    )
    parser.add_argument(
        "--metrics_out",
        type=str,
        default="metrics_pointwise.csv",
        help="Output CSV path for --mode metrics.",
    )
    parser.add_argument(
        "--metrics_groupby",
        type=str,
        default="instrument,lead_hours_nominal",
        help=(
            "Comma-separated grouping keys for --mode metrics. "
            "Common: instrument,lead_hours_nominal or instrument,lead_hours_nominal,pressure_level_label"
        ),
    )
    parser.add_argument(
        "--metrics_recursive",
        action="store_true",
        help="If set, recursively search under data_dir for matching CSVs (recommended for val_csv root).",
    )
    parser.add_argument(
        "--metrics_pattern",
        type=str,
        default="val_*.csv",
        help="Glob pattern (relative to data_dir) to include for --mode metrics.",
    )
    parser.add_argument(
        "--metrics_min_count",
        type=int,
        default=100,
        help="Minimum sample count per group/variable to report (metrics mode).",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=DEFAULT_FIG_DIR,
        help="Directory to save output plots"
    )
    parser.add_argument(
        "--has_ground_truth",
        action="store_true",
        help="Set if CSV files contain true_ columns for comparison with predictions"
    )
    return parser.parse_args()


def _require_plotting():
    if ccrs is None or plt is None:
        raise RuntimeError(
            "Plotting dependencies (cartopy/matplotlib) are not available in this environment. "
            "Run with --mode metrics or install cartopy+matplotlib."
        )


def _parse_groupby_keys(s: str) -> list[str]:
    keys = [k.strip() for k in (s or "").split(",") if k.strip()]
    return keys or ["instrument", "lead_hours_nominal"]


def _collect_csv_files(data_dir: str, pattern: str, recursive: bool) -> list[str]:
    data_dir = os.path.abspath(data_dir)
    if recursive:
        glob_pat = os.path.join(data_dir, "**", pattern)
        files = glob.glob(glob_pat, recursive=True)
    else:
        glob_pat = os.path.join(data_dir, pattern)
        files = glob.glob(glob_pat)
    return sorted([f for f in files if os.path.isfile(f)])


def _infer_instrument_from_filename(path: str) -> str:
    base = os.path.basename(path)
    # Expected formats:
    #   val_<instrument>_init_<YYYYMMDDHH>_epochE_batchB.csv
    #   pred_<instrument>_init_<YYYYMMDDHH>.csv
    if base.startswith("val_"):
        base = base[len("val_"):]
    elif base.startswith("pred_"):
        base = base[len("pred_"):]
    # chop suffixes
    for token in ("_init_", "_epoch", "_batch", ".csv"):
        if token in base:
            base = base.split(token, 1)[0]
    return base


def compute_pointwise_metrics_from_val_csv(
    data_dir: str,
    out_path: str,
    groupby_keys: list[str] | None = None,
    pattern: str = "val_*.csv",
    recursive: bool = True,
    min_count: int = 100,
):
    """Compute pointwise metrics from val_csv artifacts.

    Uses per-row (lat,lon,valid_time_unix) metadata and compares pred_* vs true_*.
    Intended to mirror standard NWP verification: verify forecasts at observation locations
    at the correct valid time, and aggregate by lead time.

    Output rows are aggregated by the requested groupby keys and variable.
    """
    groupby_keys = groupby_keys or ["instrument", "lead_hours_nominal"]
    files = _collect_csv_files(data_dir, pattern=pattern, recursive=recursive)
    if not files:
        raise FileNotFoundError(f"No CSV files found under {data_dir!r} with pattern {pattern!r} (recursive={recursive})")

    rows_out = []

    for fp in files:
        try:
            df = pd.read_csv(fp)
        except Exception as e:
            print(f"[metrics] Skipping unreadable CSV: {fp} ({e})")
            continue

        instrument = _infer_instrument_from_filename(fp)
        df["instrument"] = instrument

        # Ensure lead_hours_nominal exists; if missing, derive from unix times.
        if "lead_hours_nominal" not in df.columns and {"init_time_unix", "valid_time_unix"}.issubset(df.columns):
            try:
                df["lead_hours_nominal"] = (df["valid_time_unix"].astype("int64") - df["init_time_unix"].astype("int64")) / 3600.0
            except Exception:
                pass

        # Identify base variables from pred_* columns.
        pred_cols = [c for c in df.columns if c.startswith("pred_")]
        for pred_col in pred_cols:
            base_var = pred_col[len("pred_"):]
            true_col = f"true_{base_var}"
            if true_col not in df.columns:
                continue
            mask_col = f"mask_{base_var}"

            # Build validity mask
            p = df[pred_col]
            t = df[true_col]
            valid = np.isfinite(p.to_numpy(dtype=float, na_value=np.nan)) & np.isfinite(t.to_numpy(dtype=float, na_value=np.nan))
            if mask_col in df.columns:
                try:
                    valid &= df[mask_col].fillna(False).astype(bool).to_numpy()
                except Exception:
                    pass

            if not valid.any():
                continue

            err = (p.to_numpy(dtype=float)[valid] - t.to_numpy(dtype=float)[valid]).astype(np.float64)
            abs_err = np.abs(err)
            sq_err = err * err

            # Prepare grouping dataframe
            gcols = {}
            for k in groupby_keys:
                if k in df.columns:
                    vals = df.loc[valid, k].to_numpy()
                    if k == "lead_hours_nominal":
                        try:
                            vals = vals.astype(float)
                        except Exception:
                            pass
                    gcols[k] = vals
                elif k == "variable":
                    gcols[k] = np.array([base_var] * int(valid.sum()), dtype=object)
                elif k == "instrument":
                    gcols[k] = np.array([instrument] * int(valid.sum()), dtype=object)
                else:
                    # missing group key -> constant "unknown"
                    gcols[k] = np.array(["unknown"] * int(valid.sum()), dtype=object)

            # Always include variable
            gcols["variable"] = np.array([base_var] * int(valid.sum()), dtype=object)

            gdf = pd.DataFrame(gcols)
            gdf["abs_err"] = abs_err
            gdf["sq_err"] = sq_err
            gdf["err"] = err

            gb = gdf.groupby([k for k in groupby_keys if k != "variable"] + ["variable"], dropna=False)
            agg = gb.agg(
                n=("err", "size"),
                sum_abs=("abs_err", "sum"),
                sum_sq=("sq_err", "sum"),
                sum_err=("err", "sum"),
            ).reset_index()

            if len(agg):
                rows_out.append(agg)

    if not rows_out:
        raise RuntimeError("No metrics produced; check that your val_csv files contain pred_*/true_* columns and masks.")

    # Combine across all files/batches by summing sufficient statistics.
    out_df = pd.concat(rows_out, ignore_index=True)
    gb_keys = [k for k in groupby_keys if k != "variable" and k in out_df.columns] + ["variable"]
    out_df = out_df.groupby(gb_keys, dropna=False, as_index=False).agg(
        n=("n", "sum"),
        sum_abs=("sum_abs", "sum"),
        sum_sq=("sum_sq", "sum"),
        sum_err=("sum_err", "sum"),
    )

    out_df = out_df[out_df["n"] >= int(min_count)].copy()
    denom = out_df["n"].astype(float).replace(0.0, np.nan)
    out_df["mae"] = out_df["sum_abs"].astype(float) / denom
    out_df["bias"] = out_df["sum_err"].astype(float) / denom
    out_df["rmse"] = np.sqrt(out_df["sum_sq"].astype(float) / denom)
    out_df.drop(columns=["sum_abs", "sum_sq", "sum_err"], inplace=True)

    # Stable column ordering
    base_cols = [k for k in groupby_keys if k in out_df.columns and k != "variable"]
    cols = base_cols + ["variable", "n", "rmse", "mae", "bias"]
    cols = [c for c in cols if c in out_df.columns] + [c for c in out_df.columns if c not in cols]
    out_df = out_df[cols]

    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[metrics] Wrote: {out_path} (rows={len(out_df)})")


# ----------------- helpers -----------------
TINY_THRESH = {
    "airTemperature": 2.0,  # °C
    "dewPointTemperature": 2.0,  # °C
    "relativeHumidity": 5.0,  # percentage points
    "wind_u": 1.0,  # m/s
    "wind_v": 1.0,  # m/s
    "airPressure": 5.0,  # hPa
}

# features that should default to ABS error when error_metric="auto"
AUTO_ABS = {"airTemperature", "dewPointTemperature", "relativeHumidity", "wind_u", "wind_v", "windU", "windV", "specificHumidity"}

CALM_WIND_THRESHOLD = 2.0  # m/s


def find_csv_files(
    data_dir: str,
    instrument_name: str,
    epoch: int | None = None,
    batch_idx: int | None = None,
    init_time: str | None = None,
    fhr: int | None = None,
) -> tuple[list[str], str, str]:
    """
    Find CSV files matching the specified criteria.

    For training mode: specify epoch and batch_idx
    For testing mode: specify init_time and/or fhr

    Args:
        data_dir: Directory containing CSV files
        instrument_name: Name of instrument to filter by
        epoch: Epoch number for training mode (optional)
        batch_idx: Batch index for training mode (optional)
        init_time: Initialization time for testing mode (optional, format: YYYYMMDDHH)
        fhr: Forecast hour for target predictions (optional, e.g., 3, 6, 9, 12)

    Returns:
        Tuple of (file_list, filename_tag, title_tag):
        - file_list: List of matching file paths
        - filename_tag: String to append to filenames (e.g., "_init_2024112500_epoch_156_f003")
        - title_tag: String to append to plot titles (e.g., " • Init 2024112500 • Epoch 156 • F003")
    """
    pattern = os.path.join(data_dir, '*.csv')
    csv_files = glob.glob(pattern)

    # Filter by instrument name
    if instrument_name:
        csv_files = [f for f in csv_files if instrument_name in os.path.basename(f)]

    # Training mode filters
    if epoch is not None:
        csv_files = [f for f in csv_files if f'epoch{epoch}_' in os.path.basename(f)]
    if batch_idx is not None:
        csv_files = [f for f in csv_files if f'batch{batch_idx}' in os.path.basename(f)]

    # Testing mode filters
    if init_time is not None:
        csv_files = [f for f in csv_files if f'init_{init_time}' in os.path.basename(f)]

    # Forecast hour filter (for legacy mesh-grid predictions)
    # NOTE: new val_csv files contain multiple leads within one file; we filter by lead later.
    if fhr is not None:
        legacy = [f for f in csv_files if f'_f{fhr:03d}' in os.path.basename(f)]
        if legacy:
            csv_files = legacy

    # Generate tags based on what's provided
    filename_tag = ""
    title_tag = ""

    if init_time is not None:
        filename_tag += f"_init_{init_time}"
        title_tag += f" • Init {init_time}"

    if epoch is not None:
        filename_tag += f"_epoch_{epoch}"
        title_tag += f" • Epoch {epoch}"

    if fhr is not None:
        filename_tag += f"_f{fhr:03d}"
        title_tag += f" • F{fhr:03d}"

    return csv_files, filename_tag, title_tag


def _robust_sym_limits(x, q=99.0):
    """Return symmetric limits [-m, m] using the qth percentile of |x|."""
    if x.size == 0 or not np.isfinite(x).any():
        return -1.0, 1.0
    m = float(np.nanpercentile(np.abs(x), q))
    if not np.isfinite(m) or m == 0:
        m = float(np.nanmax(np.abs(x))) if np.isfinite(x).any() else 1.0
    if m == 0:
        m = 1.0
    return -m, m


def _filter_df_by_lead_hours_nominal(df: pd.DataFrame, fhr: int | None) -> pd.DataFrame:
    """Filter val_csv rows to a single lead time when available.

    Newer val_csv artifacts may contain multiple lead times in one CSV.
    When `--fhr` is provided and `lead_hours_nominal` exists, filter rows
    so downstream plots don’t mix leads.
    """
    if fhr is None:
        return df
    if "lead_hours_nominal" not in df.columns:
        return df

    lead = pd.to_numeric(df["lead_hours_nominal"], errors="coerce").to_numpy()
    mask = np.isfinite(lead) & np.isclose(lead.astype(float), float(fhr))
    if mask.all():
        return df
    return df.loc[mask].copy()


def _filter_df_by_lead_hours_set(df: pd.DataFrame, fhrs: list[int] | None) -> pd.DataFrame:
    """Filter val_csv rows to a set of lead times when available.

    Intended for creating a single map that aggregates all observations across
    a 12-hour forecast window (e.g., leads 3/6/9/12) for better global coverage.
    """
    if not fhrs:
        return df
    if "lead_hours_nominal" not in df.columns:
        return df

    lead = pd.to_numeric(df["lead_hours_nominal"], errors="coerce").to_numpy()
    mask = np.zeros(len(df), dtype=bool)
    for fhr in fhrs:
        mask |= np.isfinite(lead) & np.isclose(lead.astype(float), float(fhr))
    if mask.all():
        return df
    return df.loc[mask].copy()


def _infer_latent_step_hours_from_df(df: pd.DataFrame) -> int | None:
    """Infer the target sub-window step (hours) from lead metadata.

    For latent rollout val_csv, `lead_hours_nominal` typically contains [3, 6, 9, 12]
    (or similar). We infer the step as the minimum positive difference.
    """
    if "lead_hours_nominal" not in df.columns:
        return None
    lead = pd.to_numeric(df["lead_hours_nominal"], errors="coerce").to_numpy(dtype=float)
    lead = lead[np.isfinite(lead)]
    if lead.size < 2:
        return None
    uniq = np.unique(np.round(lead, 6))
    uniq.sort()
    if uniq.size < 2:
        return None
    diffs = np.diff(uniq)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return None
    step = float(np.nanmin(diffs))
    if not np.isfinite(step) or step <= 0:
        return None
    return int(round(step))


def _append_target_window_tag(title_tag: str, fhr: int | None, step_hours: int | None) -> str:
    """Append a human-friendly target window tag like 'Window 03–06h after init'."""
    if fhr is None or step_hours is None or step_hours <= 0:
        return title_tag
    start = max(int(fhr) - int(step_hours), 0)
    end = int(fhr)
    return f"{title_tag} • Window {start:02d}–{end:02d}h after init"


def _append_target_window_filename_tag(filename_tag: str, fhr: int | None, step_hours: int | None) -> str:
    """Append a filename-safe tag like '_window_03h_06h' when window metadata is available."""
    if fhr is None or step_hours is None or step_hours <= 0:
        return filename_tag
    start = max(int(fhr) - int(step_hours), 0)
    end = int(fhr)
    return f"{filename_tag}_window_{start:02d}h_{end:02d}h"


def _infer_horizon_window_from_leads(leads: list[int], step_hours: int | None) -> tuple[int, int]:
    """Infer (start_h, end_h) of the aggregated horizon window from nominal lead ends."""
    if not leads:
        return 0, 0
    end_h = int(max(leads))
    if step_hours is None or step_hours <= 0:
        return 0, end_h
    start_h = max(int(min(leads)) - int(step_hours), 0)
    return start_h, end_h


def plot_ocelot_target_diff(
    instrument_name: str,
    epoch: int | None = None,
    batch_idx: int | None = None,
    init_time: str | None = None,
    fhr: int | None = None,
    num_channels: int = 1,
    data_dir: str = "val_csv",
    fig_dir: str = DEFAULT_FIG_DIR,
    units: str | None = None,  # e.g., "K" for ATMS/AMSU-A
    robust_q: float = 99.0,  # robust clipping for Difference panel
    point_size: int = 7,
    projection=ccrs.PlateCarree(),  # try ccrs.Robinson() or ccrs.Mollweide() to match your sample look
):
    """
    Make a 3-panel figure: OCELOT (prediction), Target (truth), Difference (pred - true),
    and annotate RMSE on the Difference panel.

    Args:
        instrument_name: Name of the instrument
        epoch: Epoch number (training mode)
        batch_idx: Batch index (training mode)
        init_time: Initialization time (testing mode, format: YYYYMMDDHH)
        fhr: Forecast hour (testing mode, e.g., 3, 6, 9, 12)
        num_channels: Number of channels for the instrument
        data_dir: Directory containing the CSV files
        fig_dir: Directory to save figures
        units: Units for the colorbar labels
        robust_q: Percentile for robust clipping in difference panel
        point_size: Size of scatter plot points
        projection: Cartopy projection for the map
    """
    _require_plotting()
    csv_files, filename_tag, title_tag = find_csv_files(data_dir, instrument_name, epoch, batch_idx, init_time, fhr)

    if not csv_files:
        print(f"No CSV files found matching criteria in {data_dir}")
        return

    if len(csv_files) != 1:
        print(f"Warning: Expected 1 file, found {len(csv_files)} for {instrument_name}.")
        print(f"  Files found:")
        for f in csv_files:
            print(f"    - {os.path.basename(f)}")
        print(f"  Current filters: epoch={epoch}, batch_idx={batch_idx}, init_time={init_time}, fhr={fhr}")
        return

    filepath = csv_files[0]

    try:
        df = pd.read_csv(filepath)
        print(f"\n--- Processing {instrument_name} from {filepath} ---")
        step_hours = _infer_latent_step_hours_from_df(df) if fhr is not None else None
        # New val_csv format includes multiple lead times per file.
        # If fhr is provided and lead metadata exists, filter rows by lead.
        if fhr is not None:
            before = len(df)
            df = _filter_df_by_lead_hours_nominal(df, fhr)
            after = len(df)
            if after == 0:
                print(f"[WARN] No rows remain after filtering lead_hours_nominal=={int(fhr)}h. Skipping.")
                return
            if after != before:
                title_tag += f" • Nominal lead {int(fhr):d}h"

            title_tag = _append_target_window_tag(title_tag, int(fhr), step_hours)
            filename_tag = _append_target_window_filename_tag(filename_tag, int(fhr), step_hours)

            # Optional diagnostic: show within-window spread if available
            if "obs_time_unix" in df.columns and "init_time_unix" in df.columns:
                try:
                    obs_unix = pd.to_numeric(df["obs_time_unix"], errors="coerce")
                    init_unix = pd.to_numeric(df["init_time_unix"], errors="coerce")
                    m = np.isfinite(obs_unix.to_numpy()) & np.isfinite(init_unix.to_numpy())
                    if np.any(m):
                        dh = (obs_unix.to_numpy()[m] - init_unix.to_numpy()[m]) / 3600.0
                        print(f"  obs_time_unix offsets (hours after init): min={float(np.nanmin(dh)):.3f} max={float(np.nanmax(dh)):.3f}")
                except Exception:
                    pass
    except FileNotFoundError:
        print(f"\nWarning: Could not find data file {filepath}. Skipping.")
        return

    os.makedirs(fig_dir, exist_ok=True)

    feats = _discover_features(df, num_channels)

    for fname in feats:
        true_col = f"true_{fname}"
        pred_col = f"pred_{fname}"
        needed = [true_col, pred_col, "lon", "lat"]
        if not all(c in df.columns for c in needed):
            print(f"Warning: Missing columns for '{fname}'. Skipping.")
            continue

        # valid rows - include mask checking for QC-failed observations
        t = _np(df[true_col])
        p = _np(df[pred_col])
        lon = _np(df["lon"])
        lat = _np(df["lat"])

        # Start with basic finite checks
        valid = np.isfinite(t) & np.isfinite(p) & np.isfinite(lon) & np.isfinite(lat)

        # Optional radiosonde pressure-based filtering (metadata column in CSV)
        if instrument_name == "radiosonde":
            pcol = _first_existing(df, PRESSURE_COL_CANDIDATES)
            if pcol is not None:
                pressure = _np(df[pcol])
                valid &= np.isfinite(pressure) & (pressure >= 10) & (pressure <= 1100)

        # Check for mask columns (QC validity masks)
        mask_col = f"mask_{fname}"
        if mask_col in df.columns:
            mask = df[mask_col].fillna(False).astype(bool).to_numpy()
            valid &= mask
            print(f"  Using QC mask for {fname}: {mask.sum()}/{len(mask)} valid observations")

        # Additional checks for surface observations to exclude extreme/sentinel values
        if instrument_name == "surface_obs":
            # Exclude obvious sentinel/fill values that might have passed through
            if fname == "airTemperature":
                valid &= (t >= -80) & (t <= 60) & (p >= -80) & (p <= 60)
            elif fname == "airPressure":
                valid &= (t >= 300) & (t <= 1200) & (p >= 300) & (p <= 1200)
            elif fname == "dewPointTemperature":
                valid &= (t >= -100) & (t <= 40) & (p >= -100) & (p <= 40)
            elif fname == "relativeHumidity":
                valid &= (t >= 0) & (t <= 100) & (p >= 0) & (p <= 100)
            elif fname in ["wind_u", "wind_v"]:
                valid &= (np.abs(t) <= 75) & (np.abs(p) <= 75)

        if not np.any(valid):
            print(f"Info: No valid rows for '{fname}' after QC filtering. Skipping.")
            continue

        # Apply validity filter and report filtering stats
        total_obs = len(t)
        valid_obs = valid.sum()
        filtered_obs = total_obs - valid_obs
        print(f"  {fname}: {valid_obs}/{total_obs} observations retained ({filtered_obs} filtered by QC)")

        t, p, lon, lat = t[valid], p[valid], lon[valid], lat[valid]
        diff = p - t
        rmse = float(np.sqrt(np.nanmean((diff) ** 2)))

        # shared value limits for the first two panels
        vmin = float(np.nanmin([t.min(), p.min()]))
        vmax = float(np.nanmax([t.max(), p.max()]))

        # symmetric robust limits for Difference
        dmin, dmax = _robust_sym_limits(diff, q=robust_q)
        diff_norm = TwoSlopeNorm(vmin=dmin, vcenter=0.0, vmax=dmax)

        # --- make figure ---
        fig, axes = plt.subplots(1, 3, figsize=(20, 5), subplot_kw={"projection": projection}, sharey=True)

        # Titles above each panel (matching your sample)
        panel_titles = ["OCELOT", "Target", "Difference"]
        for ax, ttl in zip(axes, panel_titles):
            ax.set_title(ttl, fontsize=14)

        # Suptitle with context
        fig.suptitle(f"{instrument_name} • {fname}{title_tag}", fontsize=16, y=1.02)

        # OCELOT (prediction)
        sc0 = axes[0].scatter(lon, lat, c=p, s=point_size, cmap="turbo", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        cb0 = fig.colorbar(sc0, ax=axes[0], orientation="vertical", pad=0.02)
        cb0.set_label(f"Value{f' ({units})' if units else ''}")

        # Target (truth)
        sc1 = axes[1].scatter(lon, lat, c=t, s=point_size, cmap="turbo", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        cb1 = fig.colorbar(sc1, ax=axes[1], orientation="vertical", pad=0.02)
        cb1.set_label(f"Value{f' ({units})' if units else ''}")

        # Difference (pred - true) with symmetric limits
        sc2 = axes[2].scatter(lon, lat, c=diff, s=point_size, cmap="bwr", norm=diff_norm, transform=ccrs.PlateCarree())
        cb2 = fig.colorbar(sc2, ax=axes[2], orientation="vertical", pad=0.02)
        cb2.set_label(f"Pred − True{f' ({units})' if units else ''}")

        # RMSE badge
        rmse_text = f"RMSE = {rmse:.2f}{f' {units}' if units else ''}"
        axes[2].text(
            0.02,
            0.98,
            rmse_text,
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, linewidth=0),
        )

        # Geo styling
        for ax in axes:
            ax.set_global()
            _add_land_boundaries(ax)
            ax.set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")

        plt.tight_layout()
        safe_fname = str(fname).replace(" ", "_")
        out_png = os.path.join(fig_dir, f"{instrument_name}_OCELOT_Target_Diff_{safe_fname}{filename_tag}.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved plot: {out_png}")


def plot_ocelot_target_diff_12h_horizon(
    instrument_name: str,
    epoch: int | None = None,
    batch_idx: int | None = None,
    init_time: str | None = None,
    horizon_fhrs: list[int] | None = None,
    strict_obs_window: bool = False,
    num_channels: int = 1,
    data_dir: str = "val_csv",
    fig_dir: str = DEFAULT_FIG_DIR,
    units: str | None = None,
    robust_q: float = 99.0,
    point_size: int = 7,
    projection=ccrs.PlateCarree(),
):
    """Single 3-panel map aggregating all observations over a multi-step forecast window.

    This is designed for new val_csv artifacts that contain multiple lead times in one CSV.
    For a 12-hour horizon with 3-hour steps, include nominal leads [3, 6, 9, 12] which correspond
    to target sub-windows 00–03, 03–06, 06–09, 09–12 hours after init.
    """
    _require_plotting()

    horizon_fhrs = horizon_fhrs or [3, 6, 9, 12]

    # Find the single CSV file for this instrument/epoch/batch/init (do NOT filter by fhr in filename).
    csv_files, filename_tag, title_tag = find_csv_files(
        data_dir,
        instrument_name,
        epoch,
        batch_idx,
        init_time,
        fhr=None,
    )

    if not csv_files:
        print(f"No CSV files found matching criteria in {data_dir}")
        return

    if len(csv_files) != 1:
        print(f"Warning: Expected 1 file, found {len(csv_files)} for {instrument_name}.")
        print(f"  Files found:")
        for f in csv_files:
            print(f"    - {os.path.basename(f)}")
        print(f"  Current filters: epoch={epoch}, batch_idx={batch_idx}, init_time={init_time}")
        return

    filepath = csv_files[0]
    try:
        df = pd.read_csv(filepath)
        print(f"\n--- Processing {instrument_name} 12h-horizon aggregate from {filepath} ---")
    except FileNotFoundError:
        print(f"\nWarning: Could not find data file {filepath}. Skipping.")
        return

    # Filter to the desired lead set if present.
    df_h = _filter_df_by_lead_hours_set(df, horizon_fhrs)
    if len(df_h) == 0:
        print(f"[WARN] No rows remain after filtering lead_hours_nominal in {horizon_fhrs}. Skipping.")
        return

    # Title/filename tags for horizon plot (relative-to-init window)
    step_hours = _infer_latent_step_hours_from_df(df_h)
    leads_present = []
    if "lead_hours_nominal" in df_h.columns:
        lead_vals = pd.to_numeric(df_h["lead_hours_nominal"], errors="coerce").to_numpy(dtype=float)
        leads_present = sorted({int(round(x)) for x in lead_vals[np.isfinite(lead_vals)]})
    if not leads_present:
        leads_present = [int(h) for h in horizon_fhrs]

    horizon_start_h, horizon_end_h = _infer_horizon_window_from_leads(leads_present, step_hours)
    filename_tag_h = f"{filename_tag}_horizon_{horizon_start_h:02d}h_{horizon_end_h:02d}h"

    leads_str = ", ".join(str(int(h)) for h in leads_present)
    step_str = f"{int(step_hours)}h" if step_hours is not None else "?h"
    title_tag_h = (
        f"{title_tag} • Horizon {horizon_start_h:02d}–{horizon_end_h:02d}h after init"
        f" (aggregated {step_str} windows; nominal leads {leads_str}h)"
    )

    # Optional: summarize actual observation-time coverage within the horizon (if available).
    if {"obs_time_unix", "init_time_unix"}.issubset(df_h.columns):
        try:
            obs_unix = pd.to_numeric(df_h["obs_time_unix"], errors="coerce").to_numpy(dtype=float)
            init_unix = pd.to_numeric(df_h["init_time_unix"], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(obs_unix) & np.isfinite(init_unix)
            # Ignore sentinel / missing obs times if present
            m &= (obs_unix >= 0) & (init_unix >= 0)
            if np.any(m):
                dsec = obs_unix[m] - init_unix[m]
                dh = dsec / 3600.0
                print(
                    f"  obs_time_unix offsets within horizon (hours after init): "
                    f"min={float(np.nanmin(dh)):.3f} max={float(np.nanmax(dh)):.3f} (N={int(dh.size)})"
                )

                start_s = float(horizon_start_h) * 3600.0
                end_s = float(horizon_end_h) * 3600.0
                tol_s = 1.0  # tolerate 1s rounding
                outside = (dsec < (start_s - tol_s)) | (dsec > (end_s + tol_s))
                n_out = int(np.sum(outside))
                if n_out > 0:
                    dh_out = dh[outside]
                    print(
                        f"  [WARN] {n_out}/{int(dh.size)} obs_time_unix offsets outside expected horizon window "
                        f"[{horizon_start_h:02d},{horizon_end_h:02d}]h after init "
                        f"(min_out={float(np.nanmin(dh_out)):.3f}, max_out={float(np.nanmax(dh_out)):.3f})"
                    )
                    if strict_obs_window:
                        raise ValueError(
                            "obs_time_unix outside expected horizon window; "
                            "re-check target binning / metadata export."
                        )
        except Exception as e:
            if strict_obs_window:
                raise
            print(f"  [WARN] Failed to summarize obs_time_unix offsets for horizon plot: {e}")

    # Optional: print counts by lead for quick sanity
    if "lead_hours_nominal" in df_h.columns:
        lead_vals = pd.to_numeric(df_h["lead_hours_nominal"], errors="coerce").to_numpy()
        counts = {
            int(h): int((np.isfinite(lead_vals) & np.isclose(lead_vals.astype(float), float(h))).sum())
            for h in horizon_fhrs
        }
        counts_str = ", ".join([f"{k}h:{v}" for k, v in counts.items()])
        print(f"  Horizon rows by lead: {counts_str} (total={len(df_h)})")

    os.makedirs(fig_dir, exist_ok=True)
    feats = _discover_features(df_h, num_channels)

    for fname in feats:
        true_col = f"true_{fname}"
        pred_col = f"pred_{fname}"
        needed = [true_col, pred_col, "lon", "lat"]
        if not all(c in df_h.columns for c in needed):
            continue

        t = _np(df_h[true_col])
        p = _np(df_h[pred_col])
        lon = _np(df_h["lon"])
        lat = _np(df_h["lat"])

        valid = np.isfinite(t) & np.isfinite(p) & np.isfinite(lon) & np.isfinite(lat)

        if instrument_name == "radiosonde":
            pcol = _first_existing(df_h, PRESSURE_COL_CANDIDATES)
            if pcol is not None:
                pressure = _np(df_h[pcol])
                valid &= np.isfinite(pressure) & (pressure >= 10) & (pressure <= 1100)

        mask_col = f"mask_{fname}"
        if mask_col in df_h.columns:
            mask = df_h[mask_col].fillna(False).astype(bool).to_numpy()
            valid &= mask

        if instrument_name == "surface_obs":
            if fname == "airTemperature":
                valid &= (t >= -80) & (t <= 60) & (p >= -80) & (p <= 60)
            elif fname == "airPressure":
                valid &= (t >= 300) & (t <= 1200) & (p >= 300) & (p <= 1200)
            elif fname == "dewPointTemperature":
                valid &= (t >= -100) & (t <= 40) & (p >= -100) & (p <= 40)
            elif fname == "relativeHumidity":
                valid &= (t >= 0) & (t <= 100) & (p >= 0) & (p <= 100)
            elif fname in ["wind_u", "wind_v"]:
                valid &= (np.abs(t) <= 75) & (np.abs(p) <= 75)

        if not np.any(valid):
            continue

        t, p, lon, lat = t[valid], p[valid], lon[valid], lat[valid]
        diff = p - t
        rmse = float(np.sqrt(np.nanmean((diff) ** 2)))

        vmin = float(np.nanmin([t.min(), p.min()]))
        vmax = float(np.nanmax([t.max(), p.max()]))
        dmin, dmax = _robust_sym_limits(diff, q=robust_q)
        diff_norm = TwoSlopeNorm(vmin=dmin, vcenter=0.0, vmax=dmax)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5), subplot_kw={"projection": projection}, sharey=True)

        panel_titles = ["OCELOT (all leads)", "Target (all leads)", "Difference (all leads)"]
        for ax, ttl in zip(axes, panel_titles):
            ax.set_title(ttl, fontsize=14)

        fig.suptitle(f"{instrument_name} • {fname}{title_tag_h}", fontsize=16, y=1.02)

        sc0 = axes[0].scatter(lon, lat, c=p, s=point_size, cmap="turbo", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        fig.colorbar(sc0, ax=axes[0], orientation="vertical", pad=0.02).set_label(f"Value{f' ({units})' if units else ''}")

        sc1 = axes[1].scatter(lon, lat, c=t, s=point_size, cmap="turbo", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        fig.colorbar(sc1, ax=axes[1], orientation="vertical", pad=0.02).set_label(f"Value{f' ({units})' if units else ''}")

        sc2 = axes[2].scatter(lon, lat, c=diff, s=point_size, cmap="bwr", norm=diff_norm, transform=ccrs.PlateCarree())
        fig.colorbar(sc2, ax=axes[2], orientation="vertical", pad=0.02).set_label(f"Pred − True{f' ({units})' if units else ''}")

        rmse_text = f"RMSE = {rmse:.2f}{f' {units}' if units else ''}"
        axes[2].text(
            0.02,
            0.98,
            rmse_text,
            transform=axes[2].transAxes,
            ha="left",
            va="top",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.8, linewidth=0),
        )

        for ax in axes:
            ax.set_global()
            _add_land_boundaries(ax)
            ax.set_xlabel("Longitude")
        axes[0].set_ylabel("Latitude")

        plt.tight_layout()
        safe_fname = str(fname).replace(" ", "_")
        out_png = os.path.join(fig_dir, f"{instrument_name}_OCELOT_Target_Diff_{safe_fname}{filename_tag_h}.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved 12h-horizon plot: {out_png}")


def plot_mesh_maps(
    instrument_name: str,
    epoch: int | None = None,
    batch_idx: int | None = None,
    init_time: str | None = None,
    fhr: int | None = None,
    num_channels: int = 1,
    data_dir: str = "val_csv",
    fig_dir: str = DEFAULT_FIG_DIR,
    units: str | None = None,
    point_size: int = 7,
    projection=ccrs.PlateCarree(),
):
    """
    Plot mesh-grid maps (no ground truth comparison).
    Used for forecast files that don't have true_ columns.

    Args:
        instrument_name: Name of the instrument
        epoch: Epoch number (training mode)
        batch_idx: Batch index (training mode)
        init_time: Initialization time (testing mode, format: YYYYMMDDHH)
        fhr: Forecast hour (testing mode, e.g., 3, 6, 9, 12)
        num_channels: Number of channels for the instrument
        data_dir: Directory containing the CSV files
        fig_dir: Directory to save figures
        units: Units for the colorbar labels
        point_size: Size of scatter plot points
        projection: Cartopy projection for the map
    """
    csv_files, filename_tag, title_tag = find_csv_files(data_dir, instrument_name, epoch, batch_idx, init_time, fhr)

    if not csv_files:
        print(f"No CSV files found matching criteria in {data_dir}")
        return

    if len(csv_files) != 1:
        print(f"Warning: Expected 1 file, found {len(csv_files)} for {instrument_name}.")
        print(f"  Files found:")
        for f in csv_files:
            print(f"    - {os.path.basename(f)}")
        print(f"  Current filters: epoch={epoch}, batch_idx={batch_idx}, init_time={init_time}, fhr={fhr}")
        return

    filepath = csv_files[0]

    try:
        df = pd.read_csv(filepath)
        print(f"\n--- Processing {instrument_name} from {filepath} ---")
        step_hours = _infer_latent_step_hours_from_df(df) if fhr is not None else None
        if fhr is not None:
            df = _filter_df_by_lead_hours_nominal(df, fhr)
            if len(df) == 0:
                print(f"[WARN] No rows remain after filtering lead_hours_nominal=={int(fhr)}h. Skipping.")
                return
            title_tag = _append_target_window_tag(title_tag, int(fhr), step_hours)
            filename_tag = _append_target_window_filename_tag(filename_tag, int(fhr), step_hours)
    except FileNotFoundError:
        print(f"\nWarning: Could not find data file {filepath}. Skipping.")
        return

    os.makedirs(fig_dir, exist_ok=True)

    feats = _discover_features(df, num_channels)

    for fname in feats:
        pred_col = f"pred_{fname}"
        needed = [pred_col, "lon", "lat"]
        if not all(c in df.columns for c in needed):
            print(f"Warning: Missing columns for '{fname}'. Skipping.")
            continue

        # Get prediction data
        p = _np(df[pred_col])
        lon = _np(df["lon"])
        lat = _np(df["lat"])

        # Basic finite checks
        valid = np.isfinite(p) & np.isfinite(lon) & np.isfinite(lat)

        # Check for mask columns (QC validity masks)
        mask_col = f"mask_{fname}"
        if mask_col in df.columns:
            mask = df[mask_col].fillna(False).astype(bool).to_numpy()
            valid &= mask
            print(f"  Using QC mask for {fname}: {mask.sum()}/{len(mask)} valid observations")

        if not np.any(valid):
            print(f"Info: No valid rows for '{fname}' after QC filtering. Skipping.")
            continue

        # Apply validity filter
        total_obs = len(p)
        valid_obs = valid.sum()
        filtered_obs = total_obs - valid_obs
        print(f"  {fname}: {valid_obs}/{total_obs} observations retained ({filtered_obs} filtered by QC)")

        p, lon, lat = p[valid], lon[valid], lat[valid]

        # Color limits
        vmin = float(np.nanmin(p))
        vmax = float(np.nanmax(p))

        # --- make figure (single panel) ---
        fig, ax = plt.subplots(1, 1, figsize=(8, 6), subplot_kw={"projection": projection})

        # Title
        ax.set_title(f"{instrument_name} • {fname}{title_tag}", fontsize=14)

        # Prediction map
        sc = ax.scatter(lon, lat, c=p, s=point_size, cmap="turbo", vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", pad=0.1)
        cb.set_label(f"Prediction{f' ({units})' if units else ''}")

        # Geo styling
        ax.set_global()
        _add_land_boundaries(ax)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

        plt.tight_layout()
        safe_fname = str(fname).replace(" ", "_")
        out_png = os.path.join(fig_dir, f"{instrument_name}_prediction_{safe_fname}{filename_tag}.png")
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  -> Saved plot: {out_png}")


def _discover_features(df: pd.DataFrame, num_channels: int):
    pred_cols = [c for c in df.columns if c.startswith("pred_")]
    feats = [c[len("pred_"):] for c in pred_cols]
    return feats if feats else [f"ch{i}" for i in range(1, num_channels + 1)]


def _np(x):  # numeric vector
    return pd.to_numeric(x, errors="coerce").to_numpy()


def _smape(p, t, eps=1e-6):
    return 200.0 * np.abs(p - t) / (np.abs(p) + np.abs(t) + eps)


def _shortest_arc_deg(a, b):
    """Absolute shortest angular difference in degrees in [0, 180]."""
    return np.abs(((a - b + 180.0) % 360.0) - 180.0)


def _print_sanity(name, t, p, tiny=None):
    ae = np.abs(p - t)
    sp_all = _smape(p, t)
    med_ae = float(np.nanmedian(ae))
    p95_ae = float(np.nanpercentile(ae, 95))
    med_sp = float(np.nanmedian(sp_all))
    p95_sp = float(np.nanpercentile(sp_all, 95))
    dropped = 0
    if tiny is not None:
        mask_rel = np.abs(t) >= tiny
        dropped = int((~mask_rel).sum())
        if mask_rel.any():
            sp = sp_all[mask_rel]
            med_sp = float(np.nanmedian(sp))
            p95_sp = float(np.nanpercentile(sp, 95))
    print(
        f"{name:20s} | N={t.size:6d} | AbsErr med/95%={med_ae:6.2f}/{p95_ae:6.2f} "
        f"| sMAPE% med/95%={med_sp:6.1f}/{p95_sp:6.1f} | dropped<tiny={dropped}"
    )


def _add_land_boundaries(ax):
    import cartopy.feature as cfeature

    ax.add_feature(cfeature.COASTLINE, linewidth=0.5, alpha=0.8)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, alpha=0.6)


def _make_axes_triple(title):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), subplot_kw={"projection": ccrs.PlateCarree()}, sharey=True)
    fig.suptitle(title, fontsize=16)
    return fig, axes


PRESSURE_COL_CANDIDATES = ["pressure_hPa", "pressure_hpa", "airPressure", "pressure"]
HEIGHT_COL_CANDIDATES = ["log_pressure_height_m", "log_pressure_height"]
PRESSURE_LEVEL_CANDIDATES = ["pressure_level_idx", "pressure_level_index"]  # Categorical level indices
PRESSURE_LABEL_CANDIDATES = ["pressure_level_label", "level_label"]  # Human-readable labels

# Standard pressure levels for radiosonde (matches model embedding)
STANDARD_PRESSURE_LEVELS = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 70, 50, 30, 20, 10]


def _first_existing(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def radiosonde_metrics_by_pressure(
    df,
    feat,
    pcol="pressure_hPa",
    bins_hpa=(1000, 850, 700, 500, 300, 200, 100, 50, 10),
    agg="mean",  # "mean" or "median"
):
    true_col = f"true_{feat}"
    pred_col = f"pred_{feat}"
    mask_col = f"mask_{feat}"

    if pcol not in df.columns or true_col not in df.columns or pred_col not in df.columns:
        return None

    p = _np(df[pcol])
    t = _np(df[true_col])
    y = _np(df[pred_col])

    valid = np.isfinite(p) & np.isfinite(t) & np.isfinite(y)
    if mask_col in df.columns:
        valid &= df[mask_col].fillna(False).astype(bool).to_numpy()

    p = p[valid]
    t = t[valid]
    y = y[valid]
    if p.size == 0:
        return None

    rows = []
    edges = list(bins_hpa)

    for hi, lo in zip(edges[:-1], edges[1:]):
        layer = (p <= hi) & (p > lo)
        if not np.any(layer):
            continue

        tt = t[layer]
        yy = y[layer]
        diff = yy - tt

        if agg == "median":
            t_agg = float(np.nanmedian(tt))
            y_agg = float(np.nanmedian(yy))
        else:
            t_agg = float(np.nanmean(tt))
            y_agg = float(np.nanmean(yy))

        rows.append({
            "p_hi_hPa": float(hi),
            "p_lo_hPa": float(lo),
            "p_mid_hPa": float(0.5 * (hi + lo)),
            "N": int(layer.sum()),
            "mean_true": t_agg,
            "mean_pred": y_agg,
            "bias": float(np.nanmean(diff)),
            "RMSE": float(np.sqrt(np.nanmean(diff ** 2))),
            "MAE": float(np.nanmean(np.abs(diff))),
        })

    return pd.DataFrame(rows)


def radiosonde_metrics_by_pressure_level(
    df,
    feat,
    level_col="pressure_level_idx",
    label_col="pressure_level_label",
    agg="mean",  # "mean" or "median"
):
    """
    Compute metrics stratified by categorical pressure level index.
    This is cleaner than binning continuous pressure values.
    Uses the pressure_level_idx from embeddings (0-15).

    Args:
        df: DataFrame with predictions and truth
        feat: Feature name (e.g., "airTemperature")
        level_col: Column with pressure level indices (0-15)
        label_col: Column with human-readable labels (e.g., "850hPa")
        agg: Aggregation method ("mean" or "median")

    Returns:
        DataFrame with metrics for each pressure level
    """
    true_col = f"true_{feat}"
    pred_col = f"pred_{feat}"
    mask_col = f"mask_{feat}"

    # Check if required columns exist
    if level_col not in df.columns or true_col not in df.columns or pred_col not in df.columns:
        return None

    level_idx = _np(df[level_col])
    t = _np(df[true_col])
    y = _np(df[pred_col])

    # Apply mask
    valid = np.isfinite(level_idx) & np.isfinite(t) & np.isfinite(y) & (level_idx >= 0)
    if mask_col in df.columns:
        valid &= df[mask_col].fillna(False).astype(bool).to_numpy()

    level_idx = level_idx[valid].astype(int)
    t = t[valid]
    y = y[valid]

    if level_idx.size == 0:
        return None

    # Get labels if available
    if label_col in df.columns:
        labels_series = df.loc[df.index[valid], label_col]
    else:
        labels_series = None

    rows = []

    # Process each pressure level (0-15)
    for lvl in range(16):
        mask = (level_idx == lvl)
        if not np.any(mask):
            continue

        tt = t[mask]
        yy = y[mask]
        diff = yy - tt

        if agg == "median":
            t_agg = float(np.nanmedian(tt))
            y_agg = float(np.nanmedian(yy))
        else:
            t_agg = float(np.nanmean(tt))
            y_agg = float(np.nanmean(yy))

        # Get human-readable label
        if labels_series is not None and mask.sum() > 0:
            level_label = labels_series.iloc[np.where(mask)[0][0]]
        else:
            level_label = f"{STANDARD_PRESSURE_LEVELS[lvl]}hPa" if lvl < len(STANDARD_PRESSURE_LEVELS) else f"level_{lvl}"

        # Compute variance ratio (key metric for collapse detection)
        true_var = float(np.nanvar(tt))
        pred_var = float(np.nanvar(yy))
        var_ratio = pred_var / true_var if true_var > 0 else np.nan

        rows.append({
            "pressure_level_idx": int(lvl),
            "pressure_level_label": level_label,
            "pressure_hPa": STANDARD_PRESSURE_LEVELS[lvl] if lvl < len(STANDARD_PRESSURE_LEVELS) else np.nan,
            "N": int(mask.sum()),
            "mean_true": t_agg,
            "mean_pred": y_agg,
            "std_true": float(np.nanstd(tt)),
            "std_pred": float(np.nanstd(yy)),
            "variance_ratio": var_ratio,
            "bias": float(np.nanmean(diff)),
            "RMSE": float(np.sqrt(np.nanmean(diff ** 2))),
            "MAE": float(np.nanmean(np.abs(diff))),
            "R2": float(np.corrcoef(tt, yy)[0, 1] ** 2) if tt.size > 1 else np.nan,
        })

    return pd.DataFrame(rows) if rows else None


def plot_radiosonde_profiles_by_pressure_level_horizon(
    instrument_name: str,
    epoch: int | None = None,
    batch_idx: int | None = None,
    init_time: str | None = None,
    horizon_fhrs: list[int] | None = None,
    strict_obs_window: bool = False,
    data_dir: str = "val_csv",
    fig_dir: str = DEFAULT_FIG_DIR,
    agg="mean",  # or "median"
    min_samples: int = 500,
):
    """Horizon-aggregate radiosonde/aircraft vertical profiles.

    Uses the categorical pressure levels (0..15) and aggregates ALL observations
    across the specified lead set (default: 3/6/9/12) into a single set of
    pressure-level metrics/plots.
    """
    horizon_fhrs = horizon_fhrs or [3, 6, 9, 12]

    # Find the single CSV file for this instrument/epoch/batch/init (do NOT filter by fhr in filename).
    csv_files, filename_tag, title_tag = find_csv_files(
        data_dir,
        instrument_name,
        epoch,
        batch_idx,
        init_time,
        fhr=None,
    )

    if not csv_files:
        print(f"No CSV files found matching criteria in {data_dir}")
        return

    if len(csv_files) != 1:
        print(f"Warning: Expected 1 file, found {len(csv_files)} for {instrument_name}.")
        print(f"  Files found:")
        for f in csv_files:
            print(f"    - {os.path.basename(f)}")
        print(f"  Current filters: epoch={epoch}, batch_idx={batch_idx}, init_time={init_time}")
        return

    filepath = csv_files[0]
    try:
        df = pd.read_csv(filepath)
        print(f"\n--- Processing {instrument_name} horizon profiles from {filepath} ---")
    except FileNotFoundError:
        print(f"\nWarning: Could not find data file {filepath}. Skipping.")
        return

    # Filter to the desired lead set if present.
    df_h = _filter_df_by_lead_hours_set(df, horizon_fhrs)
    if len(df_h) == 0:
        print(f"[WARN] No rows remain after filtering lead_hours_nominal in {horizon_fhrs}. Skipping.")
        return

    # Horizon title/filename tags (relative-to-init window)
    step_hours = _infer_latent_step_hours_from_df(df_h)
    leads_present = []
    if "lead_hours_nominal" in df_h.columns:
        lead_vals = pd.to_numeric(df_h["lead_hours_nominal"], errors="coerce").to_numpy(dtype=float)
        leads_present = sorted({int(round(x)) for x in lead_vals[np.isfinite(lead_vals)]})
    if not leads_present:
        leads_present = [int(h) for h in horizon_fhrs]

    horizon_start_h, horizon_end_h = _infer_horizon_window_from_leads(leads_present, step_hours)
    filename_tag_h = f"{filename_tag}_horizon_{horizon_start_h:02d}h_{horizon_end_h:02d}h"
    leads_str = ", ".join(str(int(h)) for h in leads_present)
    step_str = f"{int(step_hours)}h" if step_hours is not None else "?h"
    title_tag_h = (
        f"{title_tag} • Horizon {horizon_start_h:02d}–{horizon_end_h:02d}h after init"
        f" (aggregated {step_str} windows; nominal leads {leads_str}h)"
    )

    # Optional: summarize actual obs-time coverage (if available)
    if {"obs_time_unix", "init_time_unix"}.issubset(df_h.columns):
        try:
            obs_unix = pd.to_numeric(df_h["obs_time_unix"], errors="coerce").to_numpy(dtype=float)
            init_unix = pd.to_numeric(df_h["init_time_unix"], errors="coerce").to_numpy(dtype=float)
            m = np.isfinite(obs_unix) & np.isfinite(init_unix)
            m &= (obs_unix >= 0) & (init_unix >= 0)
            if np.any(m):
                dsec = obs_unix[m] - init_unix[m]
                dh = dsec / 3600.0
                print(
                    f"  obs_time_unix offsets within horizon (hours after init): "
                    f"min={float(np.nanmin(dh)):.3f} max={float(np.nanmax(dh)):.3f} (N={int(dh.size)})"
                )

                start_s = float(horizon_start_h) * 3600.0
                end_s = float(horizon_end_h) * 3600.0
                tol_s = 1.0
                outside = (dsec < (start_s - tol_s)) | (dsec > (end_s + tol_s))
                n_out = int(np.sum(outside))
                if n_out > 0:
                    dh_out = dh[outside]
                    print(
                        f"  [WARN] {n_out}/{int(dh.size)} obs_time_unix offsets outside expected horizon window "
                        f"[{horizon_start_h:02d},{horizon_end_h:02d}]h after init "
                        f"(min_out={float(np.nanmin(dh_out)):.3f}, max_out={float(np.nanmax(dh_out)):.3f})"
                    )
                    if strict_obs_window:
                        raise ValueError("obs_time_unix outside expected horizon window")
        except Exception as e:
            if strict_obs_window:
                raise
            print(f"  [WARN] Failed to summarize obs_time_unix offsets for horizon profile: {e}")

    os.makedirs(fig_dir, exist_ok=True)

    level_col = _first_existing(df_h, PRESSURE_LEVEL_CANDIDATES)
    label_col = _first_existing(df_h, PRESSURE_LABEL_CANDIDATES)

    if level_col is None:
        print("[WARN] No pressure_level_idx column found; cannot create horizon profiles.")
        return

    feats = _discover_features(df_h, num_channels=9999)

    for feat in feats:
        level_df = radiosonde_metrics_by_pressure_level(
            df_h, feat, level_col=level_col, label_col=label_col, agg=agg
        )
        if level_df is None or level_df.empty:
            continue

        out_layer = os.path.join(fig_dir, f"{instrument_name}_{feat}{filename_tag_h}_level_skill.csv")
        level_df.to_csv(out_layer, index=False)
        print(f"  -> Saved horizon pressure-level skill table: {out_layer}")

        p_hpa = level_df["pressure_hPa"].to_numpy()
        valid_mask = np.isfinite(p_hpa) & (p_hpa > 0)
        if not np.any(valid_mask):
            print(f"  [WARN] No valid pressure values for {feat}, skipping vertical profile plots")
            continue

        sample_counts = level_df["N"].to_numpy()
        sufficient_samples_mask = sample_counts >= min_samples
        plot_mask = valid_mask & sufficient_samples_mask

        if not np.any(plot_mask):
            print(f"  [WARN] No pressure levels with sufficient samples (>={min_samples}) for {feat}, skipping plots")
            continue

        p_hpa = p_hpa[plot_mask]
        level_df_filtered = level_df[plot_mask].reset_index(drop=True)

        if len(p_hpa) < 2:
            print(f"  [WARN] Only {len(p_hpa)} level(s) remaining after filtering for {feat}, need at least 2 for profile plot")
            continue

        p_min, p_max = p_hpa.min(), p_hpa.max()
        if p_max / p_min < 1.5:
            print(f"  [WARN] Pressure range too narrow for {feat} ({p_min:.0f}-{p_max:.0f} hPa, ratio={p_max/p_min:.2f}), skipping plots")
            continue

        # (A) True vs Pred profile
        t_prof = level_df_filtered["mean_true"].to_numpy()
        y_prof = level_df_filtered["mean_pred"].to_numpy()
        labels = level_df_filtered["pressure_level_label"].to_numpy()

        try:
            plt.figure(figsize=(7, 9))
            plt.plot(t_prof, p_hpa, marker="o", markersize=8, linewidth=2, label="True (level avg)")
            plt.plot(y_prof, p_hpa, marker="s", markersize=8, linewidth=2, label="Pred (level avg)")
            plt.gca().invert_yaxis()
            plt.yscale("log")
            plt.yticks(p_hpa, labels)
            plt.xlabel(f"{feat} ({agg})", fontsize=12)
            plt.ylabel("Pressure Level", fontsize=12)
            plt.title(f"{instrument_name} • {feat} • True vs Pred{title_tag_h}\n(by pressure level)", fontsize=13)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            plt.legend(fontsize=11)
            out_png = os.path.join(fig_dir, f"{instrument_name}_{feat}_true_vs_pred_by_level{filename_tag_h}.png")
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> Saved horizon True-vs-Pred-by-level plot: {out_png}")
        except Exception as e:
            plt.close()
            print(f"  [ERROR] Failed to create horizon True-vs-Pred plot for {feat}: {e}")

        # (B) RMSE profile
        rmse = level_df_filtered["RMSE"].to_numpy()
        try:
            plt.figure(figsize=(7, 9))
            plt.plot(rmse, p_hpa, marker="o", markersize=8, linewidth=2, color="red", label="RMSE")
            plt.gca().invert_yaxis()
            plt.yscale("log")
            plt.yticks(p_hpa, labels)
            plt.xlabel("RMSE", fontsize=12)
            plt.ylabel("Pressure Level", fontsize=12)
            plt.title(f"{instrument_name} • {feat} • RMSE{title_tag_h}\n(by pressure level)", fontsize=13)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            plt.legend(fontsize=11)
            out_png = os.path.join(fig_dir, f"{instrument_name}_{feat}_rmse_by_level{filename_tag_h}.png")
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> Saved horizon RMSE-by-level plot: {out_png}")
        except Exception as e:
            plt.close()
            print(f"  [ERROR] Failed to create horizon RMSE plot for {feat}: {e}")

        # (C) Variance Ratio profile
        var_ratio = level_df_filtered["variance_ratio"].to_numpy()
        try:
            plt.figure(figsize=(7, 9))
            plt.plot(var_ratio * 100, p_hpa, marker="o", markersize=8, linewidth=2, color="green", label="Variance Ratio")
            plt.axvline(x=100, color="gray", linestyle="--", linewidth=1.5, label="Perfect (100%)")
            plt.gca().invert_yaxis()
            plt.yscale("log")
            plt.yticks(p_hpa, labels)
            plt.xlabel("Prediction Variance / True Variance (%)", fontsize=12)
            plt.ylabel("Pressure Level", fontsize=12)
            plt.title(f"{instrument_name} • {feat} • Variance Ratio{title_tag_h}\n(by pressure level)", fontsize=13)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            var_ratio_finite = var_ratio[np.isfinite(var_ratio)]
            if len(var_ratio_finite) > 0:
                plt.xlim(0, max(120, np.max(var_ratio_finite) * 105))
            else:
                plt.xlim(0, 120)
            plt.legend(fontsize=11)
            out_png = os.path.join(fig_dir, f"{instrument_name}_{feat}_variance_ratio_by_level{filename_tag_h}.png")
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> Saved horizon Variance-Ratio-by-level plot: {out_png}")
        except Exception as e:
            plt.close()
            print(f"  [ERROR] Failed to create horizon Variance-Ratio plot for {feat}: {e}")


def plot_radiosonde_profiles_by_pressure_level(
    instrument_name: str,
    epoch: int | None = None,
    batch_idx: int | None = None,
    init_time: str | None = None,
    fhr: int | None = None,
    data_dir: str = "val_csv",
    fig_dir: str = DEFAULT_FIG_DIR,
    agg="mean",  # or "median"
    min_samples: int = 500,  # Minimum samples required per level for reliable statistics
):
    """
    Plot radiosonde/aircraft profiles using categorical pressure level indices.
    This is more accurate than binning continuous pressure values.
    Shows metrics stratified by the 16 standard pressure levels.

    Args:
        instrument_name: Name of the instrument
        epoch: Epoch number (training mode)
        batch_idx: Batch index (training mode)
        init_time: Initialization time (testing mode, format: YYYYMMDDHH)
        fhr: Forecast hour (testing mode, e.g., 3, 6, 9, 12)
        min_samples: Minimum number of observations required per pressure level.
                     Levels with fewer samples are excluded from plots (but kept in CSV)
                     to avoid showing unreliable statistics.
                     Default: 500 (sufficient for stable statistics)
    """
    csv_files, filename_tag, title_tag = find_csv_files(data_dir, instrument_name, epoch, batch_idx, init_time, fhr)

    if not csv_files:
        print(f"No CSV files found matching criteria in {data_dir}")
        return

    if len(csv_files) != 1:
        print(f"Warning: Expected 1 file, found {len(csv_files)} for {instrument_name}.")
        print(f"  Files found:")
        for f in csv_files:
            print(f"    - {os.path.basename(f)}")
        print(f"  Current filters: epoch={epoch}, batch_idx={batch_idx}, init_time={init_time}, fhr={fhr}")
        return

    filepath = csv_files[0]

    try:
        df = pd.read_csv(filepath)
        print(f"\n--- Processing {instrument_name} from {filepath} ---")
        if fhr is not None:
            step_hours = _infer_latent_step_hours_from_df(df)
            df = _filter_df_by_lead_hours_nominal(df, fhr)
            if len(df) == 0:
                print(f"[WARN] No rows remain after filtering lead_hours_nominal=={int(fhr)}h. Skipping.")
                return
            title_tag = _append_target_window_tag(title_tag, int(fhr), step_hours)
            filename_tag = _append_target_window_filename_tag(filename_tag, int(fhr), step_hours)
    except FileNotFoundError:
        print(f"\nWarning: Could not find data file {filepath}. Skipping.")
        return

    os.makedirs(fig_dir, exist_ok=True)

    # Check if we have pressure_level_idx column
    level_col = _first_existing(df, PRESSURE_LEVEL_CANDIDATES)
    label_col = _first_existing(df, PRESSURE_LABEL_CANDIDATES)

    if level_col is None:
        print("[WARN] No pressure_level_idx column found; use plot_radiosonde_profiles_by_pressure instead.")
        return

    feats = _discover_features(df, num_channels=9999)

    for feat in feats:
        level_df = radiosonde_metrics_by_pressure_level(
            df, feat, level_col=level_col, label_col=label_col, agg=agg
        )
        if level_df is None or level_df.empty:
            continue

        # Save table (with all levels, including those with few samples)
        out_layer = os.path.join(fig_dir, f"radiosonde_{feat}{filename_tag}_level_skill.csv")
        level_df.to_csv(out_layer, index=False)
        print(f"  -> Saved pressure-level skill table: {out_layer}")

        p_hpa = level_df["pressure_hPa"].to_numpy()

        # Filter out invalid pressure values (NaN, zero, negative) for log scale
        valid_mask = np.isfinite(p_hpa) & (p_hpa > 0)
        if not np.any(valid_mask):
            print(f"  [WARN] No valid pressure values for {feat}, skipping vertical profile plots")
            continue

        # Filter out levels with insufficient samples for reliable statistics
        sample_counts = level_df["N"].to_numpy()
        sufficient_samples_mask = sample_counts >= min_samples
        # Combine both filters
        plot_mask = valid_mask & sufficient_samples_mask

        if not np.any(plot_mask):
            print(f"  [WARN] No pressure levels with sufficient samples (>={min_samples}) for {feat}, skipping plots")
            continue

        # Count how many levels were excluded
        excluded_count = valid_mask.sum() - plot_mask.sum()
        if excluded_count > 0:
            excluded_levels = level_df[valid_mask & ~sufficient_samples_mask]
            excluded_info = ", ".join([f"{row['pressure_level_label']} (N={row['N']})"
                                      for _, row in excluded_levels.iterrows()])
            print(f"  [INFO] Excluding {excluded_count} level(s) with insufficient data: {excluded_info}")

        # Apply mask to all arrays
        p_hpa = p_hpa[plot_mask]
        level_df_filtered = level_df[plot_mask].reset_index(drop=True)

        # Final safety check: need at least 2 points for a meaningful profile plot
        if len(p_hpa) < 2:
            print(f"  [WARN] Only {len(p_hpa)} level(s) remaining after filtering for {feat}, need at least 2 for profile plot")
            continue

        # Check if pressure range is sufficient for log scale (need at least 2x ratio)
        p_min, p_max = p_hpa.min(), p_hpa.max()
        if p_max / p_min < 1.5:
            print(f"  [WARN] Pressure range too narrow for {feat} ({p_min:.0f}-{p_max:.0f} hPa, ratio={p_max/p_min:.2f}), skipping plots")
            continue

        # -------------------------
        # (A) True vs Pred profile
        # -------------------------
        t_prof = level_df_filtered["mean_true"].to_numpy()
        y_prof = level_df_filtered["mean_pred"].to_numpy()
        labels = level_df_filtered["pressure_level_label"].to_numpy()

        try:
            plt.figure(figsize=(7, 9))
            plt.plot(t_prof, p_hpa, marker="o", markersize=8, linewidth=2, label="True (level avg)")
            plt.plot(y_prof, p_hpa, marker="s", markersize=8, linewidth=2, label="Pred (level avg)")
            plt.gca().invert_yaxis()
            plt.yscale("log")
            plt.yticks(p_hpa, labels)
            plt.xlabel(f"{feat} ({agg})", fontsize=12)
            plt.ylabel("Pressure Level", fontsize=12)
            plt.title(f"{instrument_name} • {feat} • True vs Pred{title_tag}\n(by pressure level)", fontsize=13)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            plt.legend(fontsize=11)
            out_png = os.path.join(fig_dir, f"{instrument_name}_{feat}_true_vs_pred_by_level{filename_tag}.png")
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> Saved True-vs-Pred-by-level plot: {out_png}")
        except Exception as e:
            plt.close()
            print(f"  [ERROR] Failed to create True-vs-Pred plot for {feat}: {e}")

        # -------------------------
        # (B) RMSE profile
        # -------------------------
        rmse = level_df_filtered["RMSE"].to_numpy()

        try:
            plt.figure(figsize=(7, 9))
            plt.plot(rmse, p_hpa, marker="o", markersize=8, linewidth=2, color="red", label="RMSE")
            plt.gca().invert_yaxis()
            plt.yscale("log")
            plt.yticks(p_hpa, labels)
            plt.xlabel("RMSE", fontsize=12)
            plt.ylabel("Pressure Level", fontsize=12)
            plt.title(f"{instrument_name} • {feat} • RMSE{title_tag}\n(by pressure level)", fontsize=13)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            plt.legend(fontsize=11)
            out_png = os.path.join(fig_dir, f"{instrument_name}_{feat}_rmse_by_level{filename_tag}.png")
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> Saved RMSE-by-level plot: {out_png}")
        except Exception as e:
            plt.close()
            print(f"  [ERROR] Failed to create RMSE plot for {feat}: {e}")

        # -------------------------
        # (C) Variance Ratio profile (KEY METRIC!)
        # -------------------------
        var_ratio = level_df_filtered["variance_ratio"].to_numpy()

        try:
            plt.figure(figsize=(7, 9))
            plt.plot(var_ratio * 100, p_hpa, marker="o", markersize=8, linewidth=2, color="green", label="Variance Ratio")
            plt.axvline(x=100, color="gray", linestyle="--", linewidth=1.5, label="Perfect (100%)")
            plt.gca().invert_yaxis()
            plt.yscale("log")
            plt.yticks(p_hpa, labels)
            plt.xlabel("Prediction Variance / True Variance (%)", fontsize=12)
            plt.ylabel("Pressure Level", fontsize=12)
            plt.title(f"{instrument_name} • {feat} • Variance Ratio\nEpoch {epoch} (by pressure level)", fontsize=13)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            # Handle potential NaN values in variance ratio
            var_ratio_finite = var_ratio[np.isfinite(var_ratio)]
            if len(var_ratio_finite) > 0:
                plt.xlim(0, max(120, np.max(var_ratio_finite) * 105))
            else:
                plt.xlim(0, 120)
            plt.legend(fontsize=11)
            out_png = os.path.join(fig_dir, f"{instrument_name}_{feat}_variance_ratio_by_level_epoch_{epoch}.png")
            plt.savefig(out_png, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  -> Saved Variance-Ratio-by-level plot: {out_png}")
        except Exception as e:
            plt.close()
            print(f"  [ERROR] Failed to create Variance-Ratio plot for {feat}: {e}")


def plot_instrument_maps(
    instrument_name: str,
    epoch: int | None = None,
    batch_idx: int | None = None,
    init_time: str | None = None,
    fhr: int | None = None,
    num_channels: int = 1,
    data_dir: str = "val_csv",
    fig_dir: str = DEFAULT_FIG_DIR,
    error_metric: str = "auto",  # "auto" | "absolute" | "percent" | "smape"
    drop_small_truth: bool = True,  # for percent/sMAPE
):
    """
    Load prediction CSV and generate maps for each feature with robust errors.

    Args:
        instrument_name: Name of the instrument
        epoch: Epoch number (training mode)
        batch_idx: Batch index (training mode)
        init_time: Initialization time (testing mode, format: YYYYMMDDHH)
        fhr: Forecast hour (testing mode, e.g., 3, 6, 9, 12)
        num_channels: Number of channels for the instrument
        data_dir: Directory containing the CSV files
        fig_dir: Directory to save figures
        error_metric: Error metric to use (auto, absolute, percent, smape)
        drop_small_truth: Drop small truth values for relative metrics
    """
    csv_files, filename_tag, title_tag = find_csv_files(data_dir, instrument_name, epoch, batch_idx, init_time, fhr)

    if not csv_files:
        print(f"No CSV files found matching criteria in {data_dir}")
        return

    if len(csv_files) != 1:
        print(f"Warning: Expected 1 file, found {len(csv_files)} for {instrument_name}.")
        print(f"  Files found:")
        for f in csv_files:
            print(f"    - {os.path.basename(f)}")
        print(f"  Current filters: epoch={epoch}, batch_idx={batch_idx}, init_time={init_time}, fhr={fhr}")
        return

    filepath = csv_files[0]

    try:
        df = pd.read_csv(filepath)
        print(f"\n--- Processing {instrument_name} from {filepath} ---")
        if fhr is not None:
            step_hours = _infer_latent_step_hours_from_df(df)
            df = _filter_df_by_lead_hours_nominal(df, fhr)
            if len(df) == 0:
                print(f"[WARN] No rows remain after filtering lead_hours_nominal=={int(fhr)}h. Skipping.")
                return
            title_tag = _append_target_window_tag(title_tag, int(fhr), step_hours)
            filename_tag = _append_target_window_filename_tag(filename_tag, int(fhr), step_hours)
    except FileNotFoundError:
        print(f"\nWarning: Could not find data file {filepath}. Skipping.")
        return

    os.makedirs(fig_dir, exist_ok=True)

    feats = _discover_features(df, num_channels)

    for fname in feats:
        true_col = f"true_{fname}"
        pred_col = f"pred_{fname}"
        mask_col = f"mask_{fname}"
        needed = [true_col, pred_col, "lon", "lat"]
        if not all(col in df.columns for col in needed):
            print(f"Warning: Missing columns for '{fname}'. Skipping.")
            continue

        # validity mask
        valid = np.ones(len(df), dtype=bool)
        if mask_col in df.columns:
            valid &= df[mask_col].fillna(False).astype(bool).to_numpy()
        t = _np(df[true_col])
        p = _np(df[pred_col])
        lon = _np(df["lon"])
        lat = _np(df["lat"])
        pcol = _first_existing(df, PRESSURE_COL_CANDIDATES)
        pressure = _np(df[pcol]) if pcol else None
        if instrument_name == "radiosonde" and pressure is not None:
            valid &= np.isfinite(pressure) & (pressure >= 10) & (pressure <= 1100)

        valid &= np.isfinite(t) & np.isfinite(p) & np.isfinite(lon) & np.isfinite(lat)

        if instrument_name == "radiosonde":
            # Try using pressure_level_idx first (more accurate)
            level_col = _first_existing(df, PRESSURE_LEVEL_CANDIDATES)
            label_col = _first_existing(df, PRESSURE_LABEL_CANDIDATES)

            if level_col is not None:
                # Use categorical pressure levels (preferred method)
                level_df = radiosonde_metrics_by_pressure_level(df, fname, level_col=level_col, label_col=label_col)
                if level_df is not None and len(level_df) > 0:
                    out_layer = os.path.join(fig_dir, f"radiosonde_{fname}{filename_tag}_level_skill.csv")
                    level_df.to_csv(out_layer, index=False)
                    print(f"  -> Saved pressure-level skill (categorical): {out_layer}")
            else:
                # Fallback to binning continuous pressure
                pcol = _first_existing(df, PRESSURE_COL_CANDIDATES)
                if pcol is not None:
                    layer_df = radiosonde_metrics_by_pressure(df, fname, pcol=pcol)
                    if layer_df is not None and len(layer_df) > 0:
                        out_layer = os.path.join(fig_dir, f"radiosonde_{fname}{filename_tag}_pressure_skill.csv")
                        layer_df.to_csv(out_layer, index=False)
                        print(f"  -> Saved pressure-layer skill (binned): {out_layer}")

        # resolve metric
        metric = "absolute" if (error_metric == "auto" and fname in AUTO_ABS) else ("smape" if (error_metric == "auto") else error_metric)

        # drop tiny truth for relative metrics
        tiny = TINY_THRESH.get(fname, 0.0)
        if drop_small_truth and metric in ("percent", "smape"):
            valid &= np.abs(t) >= tiny

        if not np.any(valid):
            print(f"Info: No valid rows for '{fname}'. Skipping.")
            continue

        t, p, lon, lat = t[valid], p[valid], lon[valid], lat[valid]

        # sanity to console
        _print_sanity(fname, t, p, tiny if drop_small_truth else None)

        # shared color limits for true/pred
        vmin = float(np.nanmin([t.min(), p.min()]))
        vmax = float(np.nanmax([t.max(), p.max()]))

        fig, axes = _make_axes_triple(f"Instrument: {instrument_name} • {fname}{title_tag}")

        # Ground Truth
        sc1 = axes[0].scatter(lon, lat, c=t, cmap="jet", s=7, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        fig.colorbar(sc1, ax=axes[0], orientation="horizontal", pad=0.1).set_label("Value")
        axes[0].set_title("Ground Truth")

        # Prediction
        sc2 = axes[1].scatter(lon, lat, c=p, cmap="jet", s=7, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        fig.colorbar(sc2, ax=axes[1], orientation="horizontal", pad=0.1).set_label("Value")
        axes[1].set_title("Prediction")

        # Error panel
        if metric == "absolute":
            err = np.abs(p - t)
            lo, hi = np.nanpercentile(err, [1, 99])
            err = np.clip(err, lo, hi)
            label, cmap, norm = "Abs Error", "jet", None
        elif metric == "percent":
            denom = np.clip(np.abs(t), 1e-6, None)
            err = 100.0 * (p - t) / denom
            err = np.clip(err, -200, 200)
            m = float(np.nanmax(np.abs(err))) if np.isfinite(err).any() else 1.0
            label, cmap, norm = "% Error", "bwr", TwoSlopeNorm(vmin=-m, vcenter=0.0, vmax=m)
        else:  # smape
            err = _smape(p, t)
            lo, hi = np.nanpercentile(err, [1, 99])
            err = np.clip(err, lo, hi)
            label, cmap, norm = "sMAPE (%)", "jet", None

        sc3 = axes[2].scatter(lon, lat, c=err, cmap=cmap, norm=norm, s=7, transform=ccrs.PlateCarree())
        fig.colorbar(sc3, ax=axes[2], orientation="horizontal", pad=0.1).set_label(label)
        axes[2].set_title(label)

        for ax in axes:
            ax.set_xlabel("Longitude")
            _add_land_boundaries(ax)
            ax.set_global()
        axes[0].set_ylabel("Latitude")

        safe_fname = str(fname).replace(" ", "_")
        out_png = os.path.join(fig_dir, f"{instrument_name}_map_{safe_fname}{filename_tag}_{metric}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"  -> Saved plot: {out_png}")

    # -------- optional vector wind diagnostics --------
    cols_needed = {"true_wind_u", "true_wind_v", "pred_wind_u", "pred_wind_v", "lon", "lat"}
    if cols_needed.issubset(df.columns):
        tu = _np(df["true_wind_u"])
        tv = _np(df["true_wind_v"])
        pu = _np(df["pred_wind_u"])
        pv = _np(df["pred_wind_v"])
        lon_all = _np(df["lon"])
        lat_all = _np(df["lat"])

        valid = np.isfinite(tu) & np.isfinite(tv) & np.isfinite(pu) & np.isfinite(pv) & np.isfinite(lon_all) & np.isfinite(lat_all)
        tu, tv, pu, pv = tu[valid], tv[valid], pu[valid], pv[valid]
        lon_all, lat_all = lon_all[valid], lat_all[valid]

        ts = np.hypot(tu, tv)
        ps = np.hypot(pu, pv)
        # meteorological direction in deg [0,360)
        tdir = (np.degrees(np.arctan2(-tu, -tv)) + 360.0) % 360.0
        pdir = (np.degrees(np.arctan2(-pu, -pv)) + 360.0) % 360.0
        ang = _shortest_arc_deg(pdir, tdir)
        se = np.abs(ps - ts)

        # mask calm winds for direction
        calm = ts < CALM_WIND_THRESHOLD
        tdir_c = tdir.copy()
        pdir_c = pdir.copy()
        ang_c = ang.copy()
        tdir_c[calm] = np.nan
        pdir_c[calm] = np.nan
        ang_c[calm] = np.nan

        # ---------- wind speed triple (ALL valid points) ----------
        vmin = float(np.nanmin([ts.min(), ps.min()]))
        vmax = float(np.nanmax([ts.max(), ps.max()]))

        fig, axes = _make_axes_triple(f"Instrument: {instrument_name} • wind_speed{title_tag}")

        sc1 = axes[0].scatter(lon_all, lat_all, c=ts, cmap="jet", s=7, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        fig.colorbar(sc1, ax=axes[0], orientation="horizontal", pad=0.1).set_label("Value")
        axes[0].set_title("Ground Truth")

        sc2 = axes[1].scatter(lon_all, lat_all, c=ps, cmap="jet", s=7, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
        fig.colorbar(sc2, ax=axes[1], orientation="horizontal", pad=0.1).set_label("Value")
        axes[1].set_title("Prediction")

        lo, hi = np.nanpercentile(se, [1, 99])
        sc3 = axes[2].scatter(lon_all, lat_all, c=np.clip(se, lo, hi), cmap="jet", s=7, transform=ccrs.PlateCarree())
        fig.colorbar(sc3, ax=axes[2], orientation="horizontal", pad=0.1).set_label("Abs Error (m/s)")
        axes[2].set_title("Abs Error (m/s)")

        for ax in axes:
            ax.set_xlabel("Longitude")
            _add_land_boundaries(ax)
            ax.set_global()
        axes[0].set_ylabel("Latitude")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        out_png = os.path.join(fig_dir, f"{instrument_name}_map_wind_speed{filename_tag}.png")
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"  -> Saved plot: {out_png}")

        # ---------- wind direction triple (subset to non-calm) ----------
        keep = ~np.isnan(ang_c)
        if keep.any():
            lon_dir, lat_dir = lon_all[keep], lat_all[keep]
            tdir_plot, pdir_plot, ang_plot = tdir_c[keep], pdir_c[keep], ang_c[keep]

            vmin = float(np.nanmin([tdir_plot.min(), pdir_plot.min()]))
            vmax = float(np.nanmax([tdir_plot.max(), pdir_plot.max()]))

            fig, axes = _make_axes_triple(f"Instrument: {instrument_name} • wind_direction{title_tag}")

            sc1 = axes[0].scatter(lon_dir, lat_dir, c=tdir_plot, cmap="jet", s=7, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            fig.colorbar(sc1, ax=axes[0], orientation="horizontal", pad=0.1).set_label("Value")
            axes[0].set_title("Ground Truth")

            sc2 = axes[1].scatter(lon_dir, lat_dir, c=pdir_plot, cmap="jet", s=7, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            fig.colorbar(sc2, ax=axes[1], orientation="horizontal", pad=0.1).set_label("Value")
            axes[1].set_title("Prediction")

            lo, hi = np.nanpercentile(ang_plot, [1, 99])
            sc3 = axes[2].scatter(lon_dir, lat_dir, c=np.clip(ang_plot, lo, hi), cmap="jet", s=7, transform=ccrs.PlateCarree())
            fig.colorbar(sc3, ax=axes[2], orientation="horizontal", pad=0.1).set_label("Abs Error (deg)")
            axes[2].set_title("Abs Error (deg)")

            for ax in axes:
                ax.set_xlabel("Longitude")
                _add_land_boundaries(ax)
                ax.set_global()
            axes[0].set_ylabel("Latitude")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            out_png = os.path.join(fig_dir, f"{instrument_name}_map_wind_direction{filename_tag}.png")
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"  -> Saved plot: {out_png}")


# ----------------- main -----------------
if __name__ == "__main__":

    # Configuration - command-line args override these defaults
    USE_ARGS = True  # Set to True for training mode, False for testing mode

    if USE_ARGS:  # Recommend to use run_evaluation.py
        # Parse command-line arguments
        args = parse_args()

        # Metrics-only mode: compute and exit early
        if str(args.mode).lower() == "metrics":
            keys = _parse_groupby_keys(args.metrics_groupby)
            compute_pointwise_metrics_from_val_csv(
                data_dir=args.data_dir,
                out_path=args.metrics_out,
                groupby_keys=keys,
                pattern=args.metrics_pattern,
                recursive=bool(args.metrics_recursive),
                min_count=int(args.metrics_min_count),
            )
            raise SystemExit(0)

        # Otherwise: plotting mode (existing behavior)
        EPOCH_TO_PLOT = args.epoch
        BATCH_IDX_TO_PLOT = args.batch_idx
        INIT_TIME = args.init_time
        FHR = args.fhr
        PLOT_ALL_FHRS = bool(args.plot_all_fhrs)
        PLOT_HORIZON_12H = bool(args.plot_horizon_12h)
        STRICT_OBS_WINDOW = bool(args.strict_obs_window)
        DATA_DIR = args.data_dir
        PLOT_DIR = args.plot_dir
        HAS_GROUND_TRUTH = args.has_ground_truth
    else:  # Manually configure arguments, see examples below
        # --- Example[1] Training mode - Obs-location ourputs (Original outputs) ---
        HAS_GROUND_TRUTH = True  # Set True if have ground truth for comparison
        EPOCH_TO_PLOT = 159
        BATCH_IDX_TO_PLOT = 0
        INIT_TIME = "2024112500"
        FHR = None               # No forecast hours in training
        DATA_DIR = "val_csv"
        PLOT_DIR = os.path.join(DEFAULT_FIG_DIR, "val", "obs")
        PLOT_ALL_FHRS = False
        PLOT_HORIZON_12H = True
        STRICT_OBS_WINDOW = False
        # --- Example[2] Training mode - Mesh Prediction ---
        # HAS_GROUND_TRUTH = False
        # EPOCH_TO_PLOT = 159
        # BATCH_IDX_TO_PLOT = 0
        # INIT_TIME = "2024112500"
        # FHR = 12
        # DATA_DIR = "val_mesh_csv"
        # PLOT_DIR = os.path.join(DEFAULT_FIG_DIR, "val", "mesh")
        # --- Example[3] Testing mode - Evaluation ---
        # HAS_GROUND_TRUTH = True
        # EPOCH_TO_PLOT = None
        # BATCH_IDX_TO_PLOT = None
        # INIT_TIME = "2025030100"
        # FHR = None
        # DATA_DIR = "predictions/pred_csv/obs-space/"
        # PLOT_DIR = os.path.join(DEFAULT_FIG_DIR, "test", "obs")
        # --- Example[4] Testing mode - Inference ---
        # HAS_GROUND_TRUTH = False
        # EPOCH_TO_PLOT = None
        # BATCH_IDX_TO_PLOT = None
        # INIT_TIME = "2025030100"
        # FHR = 12               # Forecast hour: 3, 6, 9, or 12
        # DATA_DIR = "predictions/pred_csv/mesh-grid/"
        # PLOT_DIR = os.path.join(DEFAULT_FIG_DIR, "test", "mesh")

    plot_dir = os.path.abspath(PLOT_DIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Choose plotting approach based on HAS_GROUND_TRUTH flag
    if not HAS_GROUND_TRUTH:
        # Forecast-only visualization (target files)
        print("\n=== Plotting prediction-only maps (no ground truth) ===\n")

        plot_mesh_maps("radiosonde", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, FHR, num_channels=5, data_dir=DATA_DIR, fig_dir=plot_dir)
        plot_mesh_maps("surface_obs", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, FHR, num_channels=6, data_dir=DATA_DIR, fig_dir=plot_dir)

    else:
        # Model evaluation with ground truth
        print("\n=== Plotting OCELOT vs Target comparison ===\n")

        # Always generate an additional 12-hour horizon aggregate plot that uses all observations
        # from leads 3/6/9/12 in a *single* global map (better coverage), in addition to the
        # user-selected single-lead plot (--fhr, default 3).
        HORIZON_FHRS = [3, 6, 9, 12]

        # Decide which lead(s) to generate per-lead plots for.
        # - Default: only the user-selected single lead (FHR)
        # - With --plot_all_fhrs: generate all 4 steps (3/6/9/12)
        if PLOT_ALL_FHRS:
            fhrs_to_plot = list(HORIZON_FHRS)
            if FHR is not None and int(FHR) not in fhrs_to_plot:
                fhrs_to_plot.append(int(FHR))
        else:
            fhrs_to_plot = [int(FHR)] if FHR is not None else []

        # Plot radiosonde profiles by categorical pressure level (more accurate)
        plot_radiosonde_profiles_by_pressure_level(
            "radiosonde",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
        )
        if PLOT_HORIZON_12H:
            plot_radiosonde_profiles_by_pressure_level_horizon(
                "radiosonde",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
            )

        # Plot aircraft profiles by categorical pressure level (similar to radiosonde)
        # Use lower min_samples threshold for aircraft due to sparser data distribution
        plot_radiosonde_profiles_by_pressure_level(
            "aircraft",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            min_samples=1000,  # Exclude very sparse levels (e.g., 150 hPa with N=240)
        )
        if PLOT_HORIZON_12H:
            plot_radiosonde_profiles_by_pressure_level_horizon(
                "aircraft",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                min_samples=1000,
            )

        # Aircraft conventional observations (temperature, humidity, winds)
        for fhr_i in fhrs_to_plot:
            plot_ocelot_target_diff("aircraft", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, fhr_i,
                                    num_channels=4, data_dir=DATA_DIR, fig_dir=plot_dir, units="various")
        if PLOT_HORIZON_12H:
            plot_ocelot_target_diff_12h_horizon(
                "aircraft",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                num_channels=4,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                units="various",
            )

        # ASCAT backscatter: add units for sigma0
        for fhr_i in fhrs_to_plot:
            plot_ocelot_target_diff("ascat", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, fhr_i,
                                    num_channels=3, data_dir=DATA_DIR, fig_dir=plot_dir, units="dB")
        if PLOT_HORIZON_12H:
            plot_ocelot_target_diff_12h_horizon(
                "ascat",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                num_channels=3,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                units="dB",
            )

        # brightness temperature instruments (add units to annotate RMSE like your sample)
        for fhr_i in fhrs_to_plot:
            plot_ocelot_target_diff("atms", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, fhr_i,
                                    num_channels=22, data_dir=DATA_DIR, fig_dir=plot_dir, units="K")
        if PLOT_HORIZON_12H:
            plot_ocelot_target_diff_12h_horizon(
                "atms",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                num_channels=22,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                units="K",
            )

         # brightness temperature instruments (add units to annotate RMSE like your sample)
        for fhr_i in fhrs_to_plot:
            plot_ocelot_target_diff("cris_pca", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, fhr_i,
                                    num_channels=22, data_dir=DATA_DIR, fig_dir=plot_dir, units="K")
        if PLOT_HORIZON_12H:
            plot_ocelot_target_diff_12h_horizon(
                "cris_pca",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                num_channels=22,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                units="K",
            )

        for fhr_i in fhrs_to_plot:
            plot_ocelot_target_diff("amsua", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, fhr_i,
                                    num_channels=15, data_dir=DATA_DIR, fig_dir=plot_dir, units="K")
        if PLOT_HORIZON_12H:
            plot_ocelot_target_diff_12h_horizon(
                "amsua",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                num_channels=15,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                units="K",
            )

        for fhr_i in fhrs_to_plot:
            plot_ocelot_target_diff("ssmis", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, fhr_i,
                                    num_channels=24, data_dir=DATA_DIR, fig_dir=plot_dir, units="K")
        if PLOT_HORIZON_12H:
            plot_ocelot_target_diff_12h_horizon(
                "ssmis",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                num_channels=24,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                units="K",
            )

        for fhr_i in fhrs_to_plot:
            plot_ocelot_target_diff("seviri", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, fhr_i,
                                    num_channels=16, data_dir=DATA_DIR, fig_dir=plot_dir, units="K")
        if PLOT_HORIZON_12H:
            plot_ocelot_target_diff_12h_horizon(
                "seviri",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                num_channels=16,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                units="K",
            )

        # AVHRR reflectance/albedo: omit units or add as needed
        for fhr_i in fhrs_to_plot:
            plot_ocelot_target_diff(
                "avhrr",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                fhr_i,
                num_channels=3,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
            )
        if PLOT_HORIZON_12H:
            plot_ocelot_target_diff_12h_horizon(
                "avhrr",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                num_channels=3,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                units=None,
            )
        # Surface obs and snow cover: omit units or add as needed
        for fhr_i in fhrs_to_plot:
            plot_ocelot_target_diff(
                "surface_obs",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                fhr_i,
                num_channels=6,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
            )
        if PLOT_HORIZON_12H:
            plot_ocelot_target_diff_12h_horizon(
                "surface_obs",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                num_channels=6,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                units=None,
            )

        for fhr_i in fhrs_to_plot:
            plot_ocelot_target_diff(
                "snow_cover",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                fhr_i,
                num_channels=2,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
            )
        if PLOT_HORIZON_12H:
            plot_ocelot_target_diff_12h_horizon(
                "snow_cover",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                num_channels=2,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                units=None,
            )

        # plot_instrument_maps also requires ground truth

        plot_instrument_maps(
            "radiosonde",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            num_channels=5,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            error_metric="auto",  # ABS for most, sMAPE for pressure
            drop_small_truth=True,
        )

        # Aircraft: similar to radiosonde with 4 features (T, q, u, v)
        plot_instrument_maps(
            "aircraft",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            num_channels=4,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            error_metric="auto",  # ABS for temperature and winds
            drop_small_truth=True,
        )

        # ASCAT backscatter: use absolute error for sigma0 measurements
        plot_instrument_maps(
            "ascat",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            num_channels=3,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            error_metric="absolute",  # Absolute error for backscatter coefficients
            drop_small_truth=False,
        )

        # Surface obs: ABS for thermo/u/v, sMAPE for pressure
        plot_instrument_maps(
            "surface_obs",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            num_channels=6,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            error_metric="auto",  # ABS for most, sMAPE for pressure
            drop_small_truth=True,
        )

        plot_instrument_maps(
            "snow_cover",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            num_channels=2,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            error_metric="auto",
            drop_small_truth=True,
        )

        plot_instrument_maps(
            "avhrr",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            num_channels=3,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            error_metric="percent",
            drop_small_truth=False,
        )

        plot_instrument_maps(
            "atms",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            num_channels=22,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            error_metric="percent",
            drop_small_truth=False,
        )

        plot_instrument_maps(
            "cris_pca",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            num_channels=22,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            error_metric="percent",
            drop_small_truth=False,
        )

        plot_instrument_maps(
            "amsua",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            num_channels=15,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            error_metric="percent",
            drop_small_truth=False,
        )

        plot_instrument_maps(
            "ssmis",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            num_channels=24,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            error_metric="percent",
            drop_small_truth=False,
        )

        plot_instrument_maps(
            "seviri",
            EPOCH_TO_PLOT,
            BATCH_IDX_TO_PLOT,
            INIT_TIME,
            FHR,
            num_channels=16,
            data_dir=DATA_DIR,
            fig_dir=plot_dir,
            error_metric="percent",
            drop_small_truth=False,
        )
