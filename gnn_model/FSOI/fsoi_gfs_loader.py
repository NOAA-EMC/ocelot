"""
GFS analysis loader for mesh-based FSOI verification.

Loads GFS analysis at OCELOT mesh node locations and returns normalized
tensors in the same feature space as the model's training data.

Scientific purpose
------------------
Traditional obs-space FSOI verifies against radiosonde observations, which
are geographically concentrated (NH mid-latitudes) and whose innovations
(xa - xb) can be biased when the background field quality is poor.

Mesh-space FSOI uses an INDEPENDENT reference (GFS analysis) interpolated
to the 40,962-node global icosahedral mesh. This:
  - Covers the globe uniformly (no geographic concentration bias)
  - Is independent of any instrument's background field bias
  - Allows direct comparison to traditional NWP FSOI metrics

Channel mapping (in model target channel order)
------------------------------------------------
radiosonde (target_dim=4):
  ch 0: airTemperature    → GFS t  at pressure level (K → °C)
  ch 1: dewPointTemperature → NOT in GFS isobaric GRIB  [set to NaN → excluded]
  ch 2: wind_u            → GFS u  at pressure level (m/s)
  ch 3: wind_v            → GFS v  at pressure level (m/s)

surface_obs (target_dim=5):
  ch 0: pressureMeanSeaLevel_prepbufr → GFS MSLP (Pa → hPa)
  ch 1: airTemperature               → GFS 2m temperature (K → °C)
  ch 2: dewPointTemperature          → NOT available directly [set to NaN]
  ch 3: wind_u                       → GFS 10m u wind (m/s)
  ch 4: wind_v                       → GFS 10m v wind (m/s)

Channels set to NaN are excluded from the MSE automatically.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

# Standard OCELOT pressure levels (hPa) indexed 0–15
STANDARD_PRESSURE_LEVELS_HPA = [
    1000, 925, 850, 700, 500, 400, 300, 250,
    200, 150, 100, 70, 50, 30, 20, 10,
]


def _nearest_gfs_cycle(dt: datetime) -> datetime:
    """Floor to the nearest 6-hourly GFS analysis cycle (00/06/12/18Z)."""
    h = (dt.hour // 6) * 6
    return dt.replace(hour=h, minute=0, second=0, microsecond=0)


def _gfs_analysis_path(gfs_root: str, valid_dt: datetime) -> str:
    ymd = valid_dt.strftime("%Y%m%d")
    hh = valid_dt.strftime("%H")
    return os.path.join(gfs_root, ymd, f"gfs.{ymd}.t{hh}z.pgrb2.0p25.f000")


def _lon_to_360(lon: np.ndarray) -> np.ndarray:
    """Convert [-180,180] longitudes to [0,360]."""
    return np.mod(lon, 360.0)


def _interp_grib(
    path: str,
    type_of_level: str,
    short_name: str,
    level: int | None,
    preferred_var: str,
    lat: np.ndarray,
    lon360: np.ndarray,
    scale: float = 1.0,
    offset: float = 0.0,
) -> np.ndarray:
    """Open GRIB2 field and interpolate to scattered (lat, lon) points."""
    import cfgrib
    import xarray as xr

    keys = {"typeOfLevel": type_of_level, "shortName": short_name}
    if level is not None:
        keys["level"] = int(level)

    ds = cfgrib.open_dataset(path, filter_by_keys=keys, indexpath="")
    try:
        var = preferred_var if preferred_var in ds else list(ds.data_vars)[0]
        lat_da = xr.DataArray(lat.astype(np.float64), dims="points")
        lon_da = xr.DataArray(lon360.astype(np.float64), dims="points")
        values = ds[var].interp(latitude=lat_da, longitude=lon_da,
                                method="linear").values.astype(np.float64)
    finally:
        getattr(ds, "close", lambda: None)()

    return values * scale + offset


def _load_gfs_radiosonde(
    path: str,
    pressure_hpa: int,
    lat: np.ndarray,
    lon360: np.ndarray,
) -> dict[str, np.ndarray | None]:
    """Load temperature, u-wind, v-wind from GFS at one isobaric pressure level.

    dewPointTemperature is not directly available in GFS isobaric GRIB — returned
    as None so it is excluded from the loss.
    """
    T = _interp_grib(path, "isobaricInhPa", "t", pressure_hpa, "t",
                     lat, lon360, scale=1.0, offset=-273.15)  # K → °C
    u = _interp_grib(path, "isobaricInhPa", "u", pressure_hpa, "u",
                     lat, lon360)
    v = _interp_grib(path, "isobaricInhPa", "v", pressure_hpa, "v",
                     lat, lon360)
    return {
        "airTemperature": T,        # ch 0
        "dewPointTemperature": None,  # ch 1 — excluded
        "wind_u": u,                 # ch 2
        "wind_v": v,                 # ch 3
    }


def _load_gfs_surface_obs(
    path: str,
    lat: np.ndarray,
    lon360: np.ndarray,
) -> dict[str, np.ndarray | None]:
    """Load MSLP, 2m temperature, 10m winds from GFS surface analysis."""
    import cfgrib
    import xarray as xr

    # MSLP (Pa → hPa)
    try:
        for short in ("prmsl", "msl"):
            try:
                ds = cfgrib.open_dataset(path, filter_by_keys={
                    "typeOfLevel": "meanSea", "shortName": short}, indexpath="")
                var = list(ds.data_vars)[0]
                lat_da = xr.DataArray(lat.astype(np.float64), dims="points")
                lon_da = xr.DataArray(lon360.astype(np.float64), dims="points")
                mslp = ds[var].interp(latitude=lat_da, longitude=lon_da,
                                      method="linear").values.astype(np.float64) / 100.0
                getattr(ds, "close", lambda: None)()
                break
            except Exception:
                mslp = None
    except Exception:
        mslp = None

    T2m = _interp_grib(path, "heightAboveGround", "2t", 2, "t2m",
                       lat, lon360, scale=1.0, offset=-273.15)
    u10 = _interp_grib(path, "heightAboveGround", "10u", 10, "u10", lat, lon360)
    v10 = _interp_grib(path, "heightAboveGround", "10v", 10, "v10", lat, lon360)

    return {
        "pressureMeanSeaLevel_prepbufr": mslp,  # ch 0
        "airTemperature": T2m,                   # ch 1
        "dewPointTemperature": None,              # ch 2 — excluded
        "wind_u": u10,                            # ch 3
        "wind_v": v10,                            # ch 4
    }


def _time_interp_gfs(
    valid_dt: datetime,
    gfs_root: str,
    loader,
) -> dict[str, np.ndarray | None]:
    """Time-interpolate GFS analysis between the two bracketing 6-hourly cycles."""
    lo = _nearest_gfs_cycle(valid_dt)
    hi = lo + timedelta(hours=6)
    lo_path = _gfs_analysis_path(gfs_root, lo)
    hi_path = _gfs_analysis_path(gfs_root, hi)

    lo_ok = os.path.exists(lo_path)
    hi_ok = os.path.exists(hi_path)

    if lo_ok and hi_ok:
        w_hi = (valid_dt - lo).total_seconds() / 21600.0
        w_lo = 1.0 - w_hi
        lo_data = loader(lo_path)
        hi_data = loader(hi_path)
        result = {}
        for k in lo_data:
            if lo_data[k] is None or hi_data[k] is None:
                result[k] = None
            else:
                result[k] = lo_data[k] * w_lo + hi_data[k] * w_hi
        return result
    elif lo_ok:
        return loader(lo_path)
    elif hi_ok:
        return loader(hi_path)
    else:
        raise FileNotFoundError(
            f"No GFS analysis found near {valid_dt}: "
            f"tried {lo_path} and {hi_path}"
        )


def _normalize(values: np.ndarray, feature_name: str, stats: dict) -> np.ndarray:
    """Normalize physical values to the model's feature space: (x - mean) / std."""
    if feature_name not in stats:
        raise KeyError(f"Feature '{feature_name}' not in feature_stats; "
                       f"available: {list(stats.keys())}")
    mean, std = stats[feature_name]
    std = max(float(std), 1e-6)
    return (values - float(mean)) / std


def load_gfs_on_mesh(
    valid_dt: datetime,
    mesh_lats: np.ndarray,
    mesh_lons: np.ndarray,
    inst_name: str,
    feature_stats: dict,
    gfs_root: str,
    pressure_level_idx: int = 4,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Load GFS analysis at OCELOT mesh nodes, normalized to model feature space.

    Parameters
    ----------
    valid_dt      : valid datetime of the GFS analysis (UTC).
    mesh_lats     : [N_mesh] latitude array of mesh nodes (degrees).
    mesh_lons     : [N_mesh] longitude array of mesh nodes (degrees, any convention).
    inst_name     : 'radiosonde' or 'surface_obs'.
    feature_stats : {feature_name: [mean, std]} from load_weights_from_yaml().
    gfs_root      : Root directory of GFS GRIB2 files.
    pressure_level_idx: 0–15 index into STANDARD_PRESSURE_LEVELS_HPA (radiosonde only).
    device        : torch device for the output tensor.

    Returns
    -------
    tensor [N_mesh, C] in normalized feature space.
    Channels missing from GFS (e.g., dewPointTemperature) are filled with NaN
    so they are automatically excluded from the MSE.
    """
    inst_stats = feature_stats.get(inst_name, {})
    if not inst_stats:
        raise KeyError(f"No feature_stats for instrument '{inst_name}'")

    lon360 = _lon_to_360(mesh_lons)
    N = len(mesh_lats)

    if inst_name == "radiosonde":
        pressure_hpa = STANDARD_PRESSURE_LEVELS_HPA[pressure_level_idx]
        print(f"[GFS Loader] radiosonde at {pressure_hpa} hPa — {valid_dt.isoformat()}")

        def loader(path):
            return _load_gfs_radiosonde(path, pressure_hpa, mesh_lats, lon360)

        features_order = ["airTemperature", "dewPointTemperature", "wind_u", "wind_v"]

    elif inst_name == "surface_obs":
        print(f"[GFS Loader] surface_obs — {valid_dt.isoformat()}")

        def loader(path):
            return _load_gfs_surface_obs(path, mesh_lats, lon360)

        features_order = [
            "pressureMeanSeaLevel_prepbufr", "airTemperature",
            "dewPointTemperature", "wind_u", "wind_v"
        ]
    else:
        raise ValueError(f"Mesh FSOI GFS loader not implemented for '{inst_name}'. "
                         f"Supported: 'radiosonde', 'surface_obs'.")

    gfs_data = _time_interp_gfs(valid_dt, gfs_root, loader)

    # Build [N_mesh, C] tensor in normalized space
    C = len(features_order)
    out = np.full((N, C), np.nan, dtype=np.float32)
    for ch_idx, feat in enumerate(features_order):
        values = gfs_data.get(feat)
        if values is None:
            # NaN → excluded from MSE in compute_forecast_error_on_mesh
            continue
        out[:, ch_idx] = _normalize(values, feat, inst_stats).astype(np.float32)

    loaded_ch = [f for f in features_order
                 if gfs_data.get(f) is not None]
    print(f"[GFS Loader] Loaded {len(loaded_ch)}/{C} channels: {loaded_ch}")

    return torch.from_numpy(out).to(device)
