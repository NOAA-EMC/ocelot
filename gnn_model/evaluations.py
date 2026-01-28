import os
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

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

PLOT_DIR = "figures"


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


def plot_ocelot_target_diff(
    instrument_name: str,
    epoch: int,
    batch_idx: int,
    num_channels: int = 1,
    data_dir: str = "val_csv",
    fig_dir: str = PLOT_DIR,
    units: str | None = None,  # e.g., "K" for ATMS/AMSU-A
    robust_q: float = 99.0,  # robust clipping for Difference panel
    point_size: int = 7,
    projection=ccrs.PlateCarree(),  # try ccrs.Robinson() or ccrs.Mollweide() to match your sample look
):
    """
    Make a 3-panel figure: OCELOT (prediction), Target (truth), Difference (pred - true),
    and annotate RMSE on the Difference panel.
    """
    filepath = f"{data_dir}/val_{instrument_name}_target_epoch{epoch}_batch{batch_idx}_step0.csv"
    try:
        df = pd.read_csv(filepath)
        print(f"\n--- OCELOT/Target/Difference for {instrument_name} from {filepath} ---")
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
        fig.suptitle(f"{instrument_name} • {fname} • Epoch {epoch}", fontsize=16, y=1.02)

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
        out_png = os.path.join(fig_dir, f"{instrument_name}_OCELOT_Target_Diff_{safe_fname}_epoch_{epoch}.png")
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


def plot_radiosonde_profiles_by_pressure_level(
    instrument_name: str,
    epoch: int,
    batch_idx: int,
    data_dir: str = "val_csv",
    fig_dir: str = PLOT_DIR,
    agg="mean",  # or "median"
    min_samples: int = 500,  # Minimum samples required per level for reliable statistics
):
    """
    Plot radiosonde/aircraft profiles using categorical pressure level indices.
    This is more accurate than binning continuous pressure values.
    Shows metrics stratified by the 16 standard pressure levels.

    Args:
        min_samples: Minimum number of observations required per pressure level.
                     Levels with fewer samples are excluded from plots (but kept in CSV)
                     to avoid showing unreliable statistics.
                     Default: 500 (sufficient for stable statistics)
    """
    filepath = f"{data_dir}/val_{instrument_name}_target_epoch{epoch}_batch{batch_idx}_step0.csv"
    df = pd.read_csv(filepath)

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
        out_layer = os.path.join(fig_dir, f"radiosonde_{feat}_epoch_{epoch}_level_skill.csv")
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
            plt.title(f"{instrument_name} • {feat} • True vs Pred\nEpoch {epoch} (by pressure level)", fontsize=13)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            plt.legend(fontsize=11)
            out_png = os.path.join(fig_dir, f"{instrument_name}_{feat}_true_vs_pred_by_level_epoch_{epoch}.png")
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
            plt.title(f"{instrument_name} • {feat} • RMSE\nEpoch {epoch} (by pressure level)", fontsize=13)
            plt.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.6)
            plt.legend(fontsize=11)
            out_png = os.path.join(fig_dir, f"{instrument_name}_{feat}_rmse_by_level_epoch_{epoch}.png")
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
    epoch: int,
    batch_idx: int,
    num_channels: int = 1,
    data_dir: str = "val_csv",
    fig_dir: str = PLOT_DIR,
    error_metric: str = "auto",  # "auto" | "absolute" | "percent" | "smape"
    drop_small_truth: bool = True,  # for percent/sMAPE
):
    """
    Load prediction CSV and generate maps for each feature with robust errors.
    """
    filepath = f"{data_dir}/val_{instrument_name}_target_epoch{epoch}_batch{batch_idx}_step0.csv"
    try:
        df = pd.read_csv(filepath)
        print(f"\n--- Processing {instrument_name} from {filepath} ---")
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
                    out_layer = os.path.join(fig_dir, f"radiosonde_{fname}_epoch_{epoch}_level_skill.csv")
                    level_df.to_csv(out_layer, index=False)
                    print(f"  -> Saved pressure-level skill (categorical): {out_layer}")
            else:
                # Fallback to binning continuous pressure
                pcol = _first_existing(df, PRESSURE_COL_CANDIDATES)
                if pcol is not None:
                    layer_df = radiosonde_metrics_by_pressure(df, fname, pcol=pcol)
                    if layer_df is not None and len(layer_df) > 0:
                        out_layer = os.path.join(fig_dir, f"radiosonde_{fname}_epoch_{epoch}_pressure_skill.csv")
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

        fig, axes = _make_axes_triple(f"Instrument: {instrument_name} • {fname} • Epoch: {epoch}")

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
        out_png = os.path.join(fig_dir, f"{instrument_name}_map_{safe_fname}_epoch_{epoch}_{metric}.png")
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

        fig, axes = _make_axes_triple(f"Instrument: {instrument_name} • wind_speed • Epoch: {epoch}")

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
        out_png = os.path.join(fig_dir, f"{instrument_name}_map_wind_speed_epoch_{epoch}.png")
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

            fig, axes = _make_axes_triple(f"Instrument: {instrument_name} • wind_direction • Epoch: {epoch}")

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
            out_png = os.path.join(fig_dir, f"{instrument_name}_map_wind_direction_epoch_{epoch}.png")
            plt.savefig(out_png, dpi=150)
            plt.close()
            print(f"  -> Saved plot: {out_png}")


# ----------------- main -----------------
if __name__ == "__main__":
    EPOCH_TO_PLOT = 58
    BATCH_IDX_TO_PLOT = 0
    DATA_DIR = "val_csv"

    plot_dir = os.path.abspath(PLOT_DIR)
    os.makedirs(plot_dir, exist_ok=True)

    # Plot radiosonde profiles by categorical pressure level (more accurate)
    plot_radiosonde_profiles_by_pressure_level(
        "radiosonde",
        EPOCH_TO_PLOT,
        BATCH_IDX_TO_PLOT,
        data_dir=DATA_DIR,
        fig_dir=plot_dir,
    )

    # Plot aircraft profiles by categorical pressure level (similar to radiosonde)
    # Use lower min_samples threshold for aircraft due to sparser data distribution
    plot_radiosonde_profiles_by_pressure_level(
        "aircraft",
        EPOCH_TO_PLOT,
        BATCH_IDX_TO_PLOT,
        data_dir=DATA_DIR,
        fig_dir=plot_dir,
        min_samples=1000,  # Exclude very sparse levels (e.g., 150 hPa with N=240)
    )

    # add the OCELOT | Target | Difference + RMSE figures
    # Aircraft conventional observations (temperature, humidity, winds)
    plot_ocelot_target_diff("aircraft", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, num_channels=4, data_dir=DATA_DIR, fig_dir=plot_dir, units="various")

    # ASCAT backscatter: add units for sigma0
    plot_ocelot_target_diff("ascat", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, num_channels=3, data_dir=DATA_DIR, fig_dir=plot_dir, units="dB")

    # brightness temperature instruments (add units to annotate RMSE like your sample)
    plot_ocelot_target_diff("atms", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, num_channels=22, data_dir=DATA_DIR, fig_dir=plot_dir, units="K")
    plot_ocelot_target_diff("amsua", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, num_channels=15, data_dir=DATA_DIR, fig_dir=plot_dir, units="K")
    plot_ocelot_target_diff("ssmis", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, num_channels=24, data_dir=DATA_DIR, fig_dir=plot_dir, units="K")
    plot_ocelot_target_diff("seviri", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, num_channels=16, data_dir=DATA_DIR, fig_dir=plot_dir, units="K")

    # AVHRR reflectance/albedo: omit units or add as needed
    plot_ocelot_target_diff("avhrr", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, num_channels=3, data_dir=DATA_DIR, fig_dir=plot_dir)
    # Surface obs and snow cover: omit units or add as needed
    plot_ocelot_target_diff("surface_obs", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, num_channels=6, data_dir=DATA_DIR, fig_dir=plot_dir)
    plot_ocelot_target_diff("snow_cover", EPOCH_TO_PLOT, BATCH_IDX_TO_PLOT, num_channels=2, data_dir=DATA_DIR, fig_dir=plot_dir)

    plot_instrument_maps(
        "radiosonde",
        EPOCH_TO_PLOT,
        BATCH_IDX_TO_PLOT,
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
        num_channels=16,
        data_dir=DATA_DIR,
        fig_dir=plot_dir,
        error_metric="percent",
        drop_small_truth=False,
    )
