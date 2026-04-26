#!/usr/bin/env python
"""evaluations.py

Evaluation utilities for OCELOT weather prediction model.

Author: Azadeh Gholoubi

This file historically focused on plotting diagnostics from CSV artifacts.
We now also support *pointwise* verification metrics directly from the new
`val_csv` format that includes init/valid timestamps.
"""

import os
import sys
import glob
import argparse

import numpy as np
import pandas as pd


def _reexec_under_micromamba_if_needed() -> None:
    """Re-exec this script under the stable micromamba env if needed.

    Avoids failures when the user's current `python` is from a broken conda env.
    """

    if os.environ.get("OCELOT_SKIP_MICROMAMBA_REEXEC") == "1":
        return
    if os.environ.get("OCELOT_IN_MICROMAMBA") == "1":
        return

    env_home = os.environ.get(
        "OCELOT_ENV_HOME",
        "/scratch4/NAGAPE/gpu-ai4wp/Azadeh.Gholoubi/ocelot_env",
    )
    mm = os.environ.get(
        "OCELOT_MM",
        os.path.join(env_home, "micromamba", "bin", "micromamba"),
    )
    root_prefix = os.environ.get(
        "MAMBA_ROOT_PREFIX",
        os.path.join(env_home, "micromamba_root"),
    )
    env_name = os.environ.get("OCELOT_ENV_NAME", "ocelot-cu121")

    if not (os.path.exists(mm) and os.access(mm, os.X_OK)):
        return

    new_env = os.environ.copy()
    new_env["MAMBA_ROOT_PREFIX"] = root_prefix
    new_env["OCELOT_IN_MICROMAMBA"] = "1"
    cmd = [mm, "run", "-n", env_name, "python"] + sys.argv
    os.execvpe(mm, cmd, new_env)


_reexec_under_micromamba_if_needed()

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
        "--plot_pressure_level_maps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Generate radiosonde/aircraft maps for each pressure level (e.g., level_850hPa/...). "
            "Default is enabled. Use --no-plot_pressure_level_maps to disable (note: mixed-level maps are not generated for radiosonde/aircraft)."
        ),
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

    parser.add_argument(
        "--instruments",
        nargs="*",
        default=None,
        help=(
            "Optional subset of instruments to plot (space-separated). "
            "Example: --instruments radiosonde aircraft surface_obs. "
            "If omitted, runs the default full plotting suite."
        ),
    )
    return parser.parse_args()


def _require_plotting():
    if ccrs is None or plt is None or TwoSlopeNorm is None:
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
    "pressureMeanSeaLevel_prepbufr": 5.0,  # hPa
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
    projection=None,  # cartopy CRS; defaults to PlateCarree when plotting deps are available
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
    if projection is None:
        projection = ccrs.PlateCarree()
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
            elif _is_surface_pressure_feature(fname):
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
    projection=None,
):
    """Single 3-panel map aggregating all observations over a multi-step forecast window.

    This is designed for new val_csv artifacts that contain multiple lead times in one CSV.
    For a 12-hour horizon with 3-hour steps, include nominal leads [3, 6, 9, 12] which correspond
    to target sub-windows 00–03, 03–06, 06–09, 09–12 hours after init.
    """
    _require_plotting()
    if projection is None:
        projection = ccrs.PlateCarree()

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
            elif _is_surface_pressure_feature(fname):
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
    projection=None,
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
    _require_plotting()
    if projection is None:
        projection = ccrs.PlateCarree()

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


PRESSURE_COL_CANDIDATES = ["pressure_hPa", "pressure_hpa", "pressureMeanSeaLevel", "airPressure", "pressure"]
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


def _is_surface_pressure_feature(name: str) -> bool:
    return name in {"airPressure", "pressureMeanSeaLevel_prepbufr"}


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
    horizon_fhrs: list[int] | None = None,
    strict_obs_window: bool = False,
    num_channels: int = 1,
    data_dir: str = "val_csv",
    fig_dir: str = DEFAULT_FIG_DIR,
    error_metric: str = "auto",  # "auto" | "absolute" | "percent" | "smape"
    drop_small_truth: bool = True,  # for percent/sMAPE
    plot_pressure_level_maps: bool = False,
    min_points_per_level: int = 200,
):
    """
    Load prediction CSV and generate maps for each feature with robust errors.

    Args:
        instrument_name: Name of the instrument
        epoch: Epoch number (training mode)
        batch_idx: Batch index (training mode)
        init_time: Initialization time (testing mode, format: YYYYMMDDHH)
        fhr: Forecast hour (testing mode, e.g., 3, 6, 9, 12)
        horizon_fhrs: If set, aggregate multiple leads into a single plot set (e.g., [3,6,9,12]).
            For radiosonde/aircraft with plot_pressure_level_maps enabled, this produces per-level horizon maps.
        strict_obs_window: If True, raise if obs_time_unix falls outside the expected horizon window.
        num_channels: Number of channels for the instrument
        data_dir: Directory containing the CSV files
        fig_dir: Directory to save figures
        error_metric: Error metric to use (auto, absolute, percent, smape)
        drop_small_truth: Drop small truth values for relative metrics
    """
    _require_plotting()

    # Horizon plots must not filter by fhr in the filename.
    fhr_for_filename = None if horizon_fhrs is not None else fhr
    csv_files, filename_tag, title_tag = find_csv_files(
        data_dir,
        instrument_name,
        epoch,
        batch_idx,
        init_time,
        fhr_for_filename,
    )

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
        step_hours = _infer_latent_step_hours_from_df(df)

        if horizon_fhrs is not None:
            horizon_fhrs = [int(x) for x in (horizon_fhrs or [3, 6, 9, 12])]
            df = _filter_df_by_lead_hours_set(df, horizon_fhrs)
            if len(df) == 0:
                print(f"[WARN] No rows remain after filtering lead_hours_nominal in {horizon_fhrs}. Skipping.")
                return

            leads_present = []
            if "lead_hours_nominal" in df.columns:
                lead_vals = pd.to_numeric(df["lead_hours_nominal"], errors="coerce").to_numpy(dtype=float)
                leads_present = sorted({int(round(x)) for x in lead_vals[np.isfinite(lead_vals)]})
            if not leads_present:
                leads_present = list(horizon_fhrs)

            horizon_start_h, horizon_end_h = _infer_horizon_window_from_leads(leads_present, step_hours)
            filename_tag = f"{filename_tag}_horizon_{horizon_start_h:02d}h_{horizon_end_h:02d}h"
            leads_str = ", ".join(str(int(h)) for h in leads_present)
            step_str = f"{int(step_hours)}h" if step_hours is not None else "?h"
            title_tag = (
                f"{title_tag} • Horizon {horizon_start_h:02d}–{horizon_end_h:02d}h after init"
                f" (aggregated {step_str} windows; nominal leads {leads_str}h)"
            )

            if {"obs_time_unix", "init_time_unix"}.issubset(df.columns):
                try:
                    obs_unix = pd.to_numeric(df["obs_time_unix"], errors="coerce").to_numpy(dtype=float)
                    init_unix = pd.to_numeric(df["init_time_unix"], errors="coerce").to_numpy(dtype=float)
                    m = np.isfinite(obs_unix) & np.isfinite(init_unix) & (obs_unix >= 0) & (init_unix >= 0)
                    if np.any(m):
                        dsec = obs_unix[m] - init_unix[m]
                        start_s = float(horizon_start_h) * 3600.0
                        end_s = float(horizon_end_h) * 3600.0
                        tol_s = 1.0
                        outside = (dsec < (start_s - tol_s)) | (dsec > (end_s + tol_s))
                        n_out = int(np.sum(outside))
                        if n_out > 0:
                            if strict_obs_window:
                                raise ValueError("obs_time_unix outside expected horizon window")
                            print(f"  [WARN] {n_out} obs_time_unix offsets outside expected horizon window")
                except Exception as e:
                    if strict_obs_window:
                        raise
                    print(f"  [WARN] Failed to validate obs_time_unix offsets for horizon maps: {e}")

        elif fhr is not None:
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

    # Optional: for radiosonde/aircraft, also create per-pressure-level maps.
    is_pressure_instrument = instrument_name in ("radiosonde", "aircraft")
    do_levels = bool(plot_pressure_level_maps) and is_pressure_instrument
    level_col = _first_existing(df, PRESSURE_LEVEL_CANDIDATES) if do_levels else None
    label_col = _first_existing(df, PRESSURE_LABEL_CANDIDATES) if do_levels else None
    if do_levels and (level_col is None) and (label_col is None) and ("pressure_hPa" not in df.columns):
        print("[WARN] Requested per-level maps but no pressure_level_label/idx/pressure_hPa columns found; skipping per-level maps.")
        do_levels = False

    def _level_series(frame: pd.DataFrame) -> pd.Series | None:
        if not do_levels:
            return None
        if label_col is not None and label_col in frame.columns:
            s = frame[label_col].astype(str)
            # Normalize labels like '850hPa'
            return s
        if level_col is not None and level_col in frame.columns:
            lvl = pd.to_numeric(frame[level_col], errors="coerce")
            # Map known standard indices to labels
            labels = []
            for x in lvl.to_numpy(dtype=float, na_value=np.nan):
                if not np.isfinite(x):
                    labels.append(np.nan)
                    continue
                i = int(x)
                if 0 <= i < len(STANDARD_PRESSURE_LEVELS):
                    labels.append(f"{STANDARD_PRESSURE_LEVELS[i]}hPa")
                else:
                    labels.append(f"level_{i}")
            return pd.Series(labels, index=frame.index)
        if "pressure_hPa" in frame.columns:
            p = pd.to_numeric(frame["pressure_hPa"], errors="coerce")
            # Snap to nearest standard level
            std = np.asarray(STANDARD_PRESSURE_LEVELS, dtype=float)
            out = []
            for x in p.to_numpy(dtype=float, na_value=np.nan):
                if not np.isfinite(x):
                    out.append(np.nan)
                    continue
                j = int(np.argmin(np.abs(std - float(x))))
                out.append(f"{int(std[j])}hPa")
            return pd.Series(out, index=frame.index)
        return None

    lvl_s = _level_series(df)
    if do_levels and lvl_s is not None:
        df = df.copy()
        df["_pressure_level_label_eval"] = lvl_s.astype(object)

    def _sorted_level_labels(series: pd.Series) -> list[str]:
        vals = [str(x) for x in series.dropna().unique().tolist() if str(x) and str(x) != "nan"]

        def _parse_hpa(label: str) -> float | None:
            s = str(label).lower().replace(" ", "")
            # common variants: '850hpa', 'level_850hpa', 'p=850'
            import re

            m = re.search(r"(\d{2,4})", s)
            if not m:
                return None
            try:
                return float(m.group(1))
            except Exception:
                return None

        parsed = [(v, _parse_hpa(v)) for v in vals]
        if any(p is not None for _, p in parsed):
            # sort high->low pressure
            parsed.sort(key=lambda x: (x[1] is None, -(x[1] or 0.0), x[0]))
            return [v for v, _ in parsed]
        return sorted(vals)

    for fname in feats:
        true_col = f"true_{fname}"
        pred_col = f"pred_{fname}"
        mask_col = f"mask_{fname}"
        needed = [true_col, pred_col, "lon", "lat"]
        if not all(col in df.columns for col in needed):
            print(f"Warning: Missing columns for '{fname}'. Skipping.")
            continue

        # base validity mask
        base_valid = np.ones(len(df), dtype=bool)
        if mask_col in df.columns:
            base_valid &= df[mask_col].fillna(False).astype(bool).to_numpy()
        t_all = _np(df[true_col])
        p_all = _np(df[pred_col])
        lon_all = _np(df["lon"])
        lat_all = _np(df["lat"])
        pcol = _first_existing(df, PRESSURE_COL_CANDIDATES)
        pressure = _np(df[pcol]) if pcol else None
        if instrument_name in ("radiosonde", "aircraft") and pressure is not None:
            base_valid &= np.isfinite(pressure) & (pressure >= 10) & (pressure <= 1100)

        base_valid &= np.isfinite(t_all) & np.isfinite(p_all) & np.isfinite(lon_all) & np.isfinite(lat_all)

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

        def _plot_one(tag_dir: str, tag_suffix: str, valid_mask: np.ndarray, *, min_points: int) -> None:
            if drop_small_truth and metric in ("percent", "smape"):
                valid_mask = valid_mask & (np.abs(t_all) >= tiny)
            if not np.any(valid_mask):
                return

            t = t_all[valid_mask]
            p = p_all[valid_mask]
            lon = lon_all[valid_mask]
            lat = lat_all[valid_mask]

            if int(t.size) < int(min_points):
                return

            _print_sanity(fname, t, p, tiny if drop_small_truth else None)

            vmin = float(np.nanmin([t.min(), p.min()]))
            vmax = float(np.nanmax([t.max(), p.max()]))

            fig, axes = _make_axes_triple(f"Instrument: {instrument_name} • {fname}{title_tag}{tag_suffix}")

            sc1 = axes[0].scatter(lon, lat, c=t, cmap="jet", s=7, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            fig.colorbar(sc1, ax=axes[0], orientation="horizontal", pad=0.1).set_label("Value")
            axes[0].set_title("Ground Truth")

            sc2 = axes[1].scatter(lon, lat, c=p, cmap="jet", s=7, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            fig.colorbar(sc2, ax=axes[1], orientation="horizontal", pad=0.1).set_label("Value")
            axes[1].set_title("Prediction")

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
            os.makedirs(tag_dir, exist_ok=True)
            out_png = os.path.join(tag_dir, f"{instrument_name}_map_{safe_fname}{filename_tag}{tag_suffix}_{metric}.png")
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"  -> Saved plot: {out_png}")

        # (1) aggregate (existing behavior)
        if is_pressure_instrument:
            # For radiosonde/aircraft, do not generate mixed-pressure maps.
            if not do_levels:
                print(
                    f"Info: Skipping {instrument_name} mixed-level map for '{fname}'. "
                    "Enable per-level maps with --plot_pressure_level_maps (default: enabled)."
                )
        else:
            if np.any(base_valid):
                _plot_one(fig_dir, "", base_valid, min_points=1)
            else:
                print(f"Info: No valid rows for '{fname}'. Skipping.")

        # (2) per-level
        if do_levels and "_pressure_level_label_eval" in df.columns:
            levels = _sorted_level_labels(df["_pressure_level_label_eval"])
            for lvl in levels:
                lvl_mask = base_valid & (df["_pressure_level_label_eval"].astype(str) == str(lvl)).to_numpy()
                lvl_dir = os.path.join(fig_dir, f"level_{str(lvl)}")
                _plot_one(lvl_dir, f"_{str(lvl)}", lvl_mask, min_points=int(min_points_per_level))

    # -------- optional vector wind diagnostics --------
    cols_needed = {"true_wind_u", "true_wind_v", "pred_wind_u", "pred_wind_v", "lon", "lat"}
    if cols_needed.issubset(df.columns):
        tu_all = _np(df["true_wind_u"])
        tv_all = _np(df["true_wind_v"])
        pu_all = _np(df["pred_wind_u"])
        pv_all = _np(df["pred_wind_v"])
        lon_all = _np(df["lon"])
        lat_all = _np(df["lat"])

        valid_wind = (
            np.isfinite(tu_all)
            & np.isfinite(tv_all)
            & np.isfinite(pu_all)
            & np.isfinite(pv_all)
            & np.isfinite(lon_all)
            & np.isfinite(lat_all)
        )

        def _plot_wind_maps(tag_dir: str, tag_suffix: str, valid_mask: np.ndarray) -> None:
            if valid_mask is None:
                return
            valid_mask = valid_wind & valid_mask
            if not np.any(valid_mask):
                return
            tu_i, tv_i, pu_i, pv_i = tu_all[valid_mask], tv_all[valid_mask], pu_all[valid_mask], pv_all[valid_mask]
            lon_i, lat_i = lon_all[valid_mask], lat_all[valid_mask]

            ts_i = np.hypot(tu_i, tv_i)
            ps_i = np.hypot(pu_i, pv_i)
            tdir_i = (np.degrees(np.arctan2(-tu_i, -tv_i)) + 360.0) % 360.0
            pdir_i = (np.degrees(np.arctan2(-pu_i, -pv_i)) + 360.0) % 360.0
            ang_i = _shortest_arc_deg(pdir_i, tdir_i)
            se_i = np.abs(ps_i - ts_i)

            # ---------- wind speed triple ----------
            vmin = float(np.nanmin([ts_i.min(), ps_i.min()]))
            vmax = float(np.nanmax([ts_i.max(), ps_i.max()]))

            fig, axes = _make_axes_triple(f"Instrument: {instrument_name} • wind_speed{title_tag}{tag_suffix}")

            sc1 = axes[0].scatter(lon_i, lat_i, c=ts_i, cmap="jet", s=7, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            fig.colorbar(sc1, ax=axes[0], orientation="horizontal", pad=0.1).set_label("Value")
            axes[0].set_title("Ground Truth")

            sc2 = axes[1].scatter(lon_i, lat_i, c=ps_i, cmap="jet", s=7, vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())
            fig.colorbar(sc2, ax=axes[1], orientation="horizontal", pad=0.1).set_label("Value")
            axes[1].set_title("Prediction")

            lo, hi = np.nanpercentile(se_i, [1, 99])
            sc3 = axes[2].scatter(lon_i, lat_i, c=np.clip(se_i, lo, hi), cmap="jet", s=7, transform=ccrs.PlateCarree())
            fig.colorbar(sc3, ax=axes[2], orientation="horizontal", pad=0.1).set_label("Abs Error (m/s)")
            axes[2].set_title("Abs Error (m/s)")

            for ax in axes:
                ax.set_xlabel("Longitude")
                _add_land_boundaries(ax)
                ax.set_global()
            axes[0].set_ylabel("Latitude")

            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            os.makedirs(tag_dir, exist_ok=True)
            out_png = os.path.join(tag_dir, f"{instrument_name}_map_wind_speed{filename_tag}{tag_suffix}.png")
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"  -> Saved plot: {out_png}")

            # ---------- wind direction triple (subset to non-calm) ----------
            calm_i = ts_i < CALM_WIND_THRESHOLD
            ang_i = ang_i.copy()
            ang_i[calm_i] = np.nan

            keep = ~np.isnan(ang_i)
            if keep.any():
                lon_dir, lat_dir = lon_i[keep], lat_i[keep]
                tdir_plot, pdir_plot, ang_plot = tdir_i[keep], pdir_i[keep], ang_i[keep]

                vmin = float(np.nanmin([tdir_plot.min(), pdir_plot.min()]))
                vmax = float(np.nanmax([tdir_plot.max(), pdir_plot.max()]))

                fig, axes = _make_axes_triple(f"Instrument: {instrument_name} • wind_direction{title_tag}{tag_suffix}")

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
                os.makedirs(tag_dir, exist_ok=True)
                out_png = os.path.join(tag_dir, f"{instrument_name}_map_wind_direction{filename_tag}{tag_suffix}.png")
                plt.savefig(out_png, dpi=150)
                plt.close()
                print(f"  -> Saved plot: {out_png}")

        if is_pressure_instrument:
            if do_levels and ("_pressure_level_label_eval" in df.columns):
                levels = _sorted_level_labels(df["_pressure_level_label_eval"])
                for lvl in levels:
                    lvl_mask = (df["_pressure_level_label_eval"].astype(str) == str(lvl)).to_numpy()
                    lvl_dir = os.path.join(fig_dir, f"level_{str(lvl)}")
                    _plot_wind_maps(lvl_dir, f"_{str(lvl)}", lvl_mask)
            else:
                print(
                    f"Info: Skipping {instrument_name} mixed-level wind maps. "
                    "Enable per-level maps with --plot_pressure_level_maps (default: enabled)."
                )
        else:
            _plot_wind_maps(fig_dir, "", np.ones(len(df), dtype=bool))


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
        PLOT_PRESSURE_LEVEL_MAPS = bool(args.plot_pressure_level_maps)
        STRICT_OBS_WINDOW = bool(args.strict_obs_window)
        DATA_DIR = args.data_dir
        PLOT_DIR = args.plot_dir
        HAS_GROUND_TRUTH = args.has_ground_truth
        INSTRUMENTS = args.instruments
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
        PLOT_PRESSURE_LEVEL_MAPS = False
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

    selected = None
    if 'INSTRUMENTS' in globals() and INSTRUMENTS:
        selected = {str(x).strip() for x in INSTRUMENTS if str(x).strip()}

    def _want(name: str) -> bool:
        return True if selected is None else (str(name) in selected)

    # Choose plotting approach based on HAS_GROUND_TRUTH flag
    if not HAS_GROUND_TRUTH:
        # Forecast-only visualization (target files)
        print("\n=== Plotting prediction-only maps (no ground truth) ===\n")

        if _want("radiosonde"):
            plot_mesh_maps("radiosonde", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, FHR, num_channels=5, data_dir=DATA_DIR, fig_dir=plot_dir)
        if _want("surface_obs"):
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
        if _want("radiosonde"):
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
        if _want("aircraft"):
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

        # Aircraft map plots are generated below via plot_instrument_maps().
        # We intentionally do not generate mixed-level aircraft maps (all pressures together).

        # ASCAT backscatter: add units for sigma0
        if _want("ascat"):
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
        if _want("atms"):
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
                                    num_channels=10, data_dir=DATA_DIR, fig_dir=plot_dir, units=None)
        if PLOT_HORIZON_12H:
            plot_ocelot_target_diff_12h_horizon(
                "cris_pca",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                horizon_fhrs=HORIZON_FHRS,
                strict_obs_window=STRICT_OBS_WINDOW,
                num_channels=10,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                units=None,
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

        if _want("ssmis"):
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

        if _want("seviri_asr"):
            for fhr_i in fhrs_to_plot:
                plot_ocelot_target_diff("seviri_asr", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, fhr_i,
                                        num_channels=8, data_dir=DATA_DIR, fig_dir=plot_dir, units="K")
            if PLOT_HORIZON_12H:
                plot_ocelot_target_diff_12h_horizon(
                    "seviri_asr",
                    EPOCH_TO_PLOT,
                    BATCH_IDX_TO_PLOT,
                    INIT_TIME,
                    horizon_fhrs=HORIZON_FHRS,
                    strict_obs_window=STRICT_OBS_WINDOW,
                    num_channels=8,
                    data_dir=DATA_DIR,
                    fig_dir=plot_dir,
                    units="K",
                )
        if _want("seviri_csr"):
            for fhr_i in fhrs_to_plot:
                plot_ocelot_target_diff("seviri_csr", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, INIT_TIME, fhr_i,
                                        num_channels=8, data_dir=DATA_DIR, fig_dir=plot_dir, units="K")
            if PLOT_HORIZON_12H:
                plot_ocelot_target_diff_12h_horizon(
                    "seviri_csr",
                    EPOCH_TO_PLOT,
                    BATCH_IDX_TO_PLOT,
                    INIT_TIME,
                    horizon_fhrs=HORIZON_FHRS,
                    strict_obs_window=STRICT_OBS_WINDOW,
                    num_channels=8,
                    data_dir=DATA_DIR,
                    fig_dir=plot_dir,
                    units="K",
                )

        # AVHRR reflectance/albedo: omit units or add as needed
        if _want("avhrr"):
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
        if _want("surface_obs"):
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

        if _want("snow_cover"):
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

        if _want("radiosonde"):
            for fhr_i in fhrs_to_plot:
                plot_instrument_maps(
                    "radiosonde",
                    EPOCH_TO_PLOT,
                    BATCH_IDX_TO_PLOT,
                    INIT_TIME,
                    fhr_i,
                    num_channels=5,
                    data_dir=DATA_DIR,
                    fig_dir=plot_dir,
                    error_metric="auto",
                    drop_small_truth=True,
                    plot_pressure_level_maps=PLOT_PRESSURE_LEVEL_MAPS,
                )
            if PLOT_HORIZON_12H:
                plot_instrument_maps(
                    "radiosonde",
                    EPOCH_TO_PLOT,
                    BATCH_IDX_TO_PLOT,
                    INIT_TIME,
                    fhr=None,
                    horizon_fhrs=HORIZON_FHRS,
                    strict_obs_window=STRICT_OBS_WINDOW,
                    num_channels=5,
                    data_dir=DATA_DIR,
                    fig_dir=plot_dir,
                    error_metric="auto",
                    drop_small_truth=True,
                    plot_pressure_level_maps=PLOT_PRESSURE_LEVEL_MAPS,
                )

        if _want("aircraft"):
            for fhr_i in fhrs_to_plot:
                plot_instrument_maps(
                    "aircraft",
                    EPOCH_TO_PLOT,
                    BATCH_IDX_TO_PLOT,
                    INIT_TIME,
                    fhr_i,
                    num_channels=4,
                    data_dir=DATA_DIR,
                    fig_dir=plot_dir,
                    error_metric="auto",
                    drop_small_truth=True,
                    plot_pressure_level_maps=PLOT_PRESSURE_LEVEL_MAPS,
                )
            if PLOT_HORIZON_12H:
                plot_instrument_maps(
                    "aircraft",
                    EPOCH_TO_PLOT,
                    BATCH_IDX_TO_PLOT,
                    INIT_TIME,
                    fhr=None,
                    horizon_fhrs=HORIZON_FHRS,
                    strict_obs_window=STRICT_OBS_WINDOW,
                    num_channels=4,
                    data_dir=DATA_DIR,
                    fig_dir=plot_dir,
                    error_metric="auto",
                    drop_small_truth=True,
                    plot_pressure_level_maps=PLOT_PRESSURE_LEVEL_MAPS,
                )

        if _want("ascat"):
            plot_instrument_maps(
                "ascat",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                FHR,
                num_channels=3,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                error_metric="absolute",
                drop_small_truth=False,
            )

        if _want("surface_obs"):
            plot_instrument_maps(
                "surface_obs",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                FHR,
                num_channels=6,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                error_metric="auto",
                drop_small_truth=True,
            )

        # plot_instrument_maps(
        #     "snow_cover",
        #     EPOCH_TO_PLOT,
        #     BATCH_IDX_TO_PLOT,
        #     INIT_TIME,
        #     FHR,
        #     num_channels=2,
        #     data_dir=DATA_DIR,
        #     fig_dir=plot_dir,
        #     error_metric="auto",
        #     drop_small_truth=True,
        # )

        if _want("avhrr"):
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

        if _want("atms"):
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

        if _want("cris_pca"):
            plot_instrument_maps(
                "cris_pca",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                FHR,
                num_channels=10,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                error_metric="percent",
                drop_small_truth=False,
            )

        if _want("amsua"):
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

        if _want("ssmis"):
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

        if _want("seviri_asr"):
            plot_instrument_maps(
                "seviri_asr",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                FHR,
                num_channels=8,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                error_metric="percent",
                drop_small_truth=False,
            )

        if _want("seviri_csr"):
            plot_instrument_maps(
                "seviri_csr",
                EPOCH_TO_PLOT,
                BATCH_IDX_TO_PLOT,
                INIT_TIME,
                FHR,
                num_channels=8,
                data_dir=DATA_DIR,
                fig_dir=plot_dir,
                error_metric="percent",
                drop_small_truth=False,
            )
