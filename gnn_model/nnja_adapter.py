import numpy as np
import pandas as pd


def _norm_qm(series: pd.Series) -> np.ndarray:
    """
    Normalize NNJA QM flags across schema variants:
      - numeric cast with NaN -> -1
      - map sentinel -9999 -> -1
    Returns int16 numpy array.
    """
    s = pd.to_numeric(series, errors="coerce").fillna(-1)
    s = s.mask(s == -9999, -1)
    return s.astype("int16").to_numpy()


def build_zlike_from_df(
    df: pd.DataFrame,
    var_map: dict,
    time_col: str = "OBS_TIMESTAMP",
    lat_col: str = "LAT",
    lon_col: str = "LON",
):
    """
    Build a 'z-like' dictionary from an NNJA Parquet DataFrame.

    Outputs:
      time      : int64 seconds since epoch (UTC)
      zar_time  : int64 nanoseconds since epoch (UTC)
      latitude  : float32
      longitude : float32
      features  : dict of feature_name -> np.ndarray
      metadata  : dict (e.g., height)
      Plus flat aliases for some features (airPressure, airTemperature, ...)
    """
    # ---- Time to UTC + seconds/ns since epoch (unit-safe) ----
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    valid_time = t.notna()
    if not valid_time.all():
        df = df.loc[valid_time].reset_index(drop=True)
        t = t.loc[valid_time]

    # Detect pandas datetime64 resolution and convert robustly
    unit = getattr(t.dtype, "unit", "ns")
    to_sec_div = {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1_000, "s": 1}.get(unit, 1_000_000_000)
    to_ns_mul = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}.get(unit, 1)

    t_i64 = t.astype("int64", copy=False)
    time_s = (t_i64 // to_sec_div).astype("int64")
    time_ns = (t_i64 * to_ns_mul).astype("int64")

    out = {
        "time": time_s.to_numpy(),
        "zar_time": time_ns.to_numpy(),
        "latitude": pd.to_numeric(df[lat_col], errors="coerce").astype("float32").to_numpy(),
        "longitude": pd.to_numeric(df[lon_col], errors="coerce").astype("float32").to_numpy(),
        "features": {},
        "metadata": {},
    }

    # ------------ Helpers ------------
    def _num(colname, dtype="float32"):
        return pd.to_numeric(df[colname], errors="coerce").astype(dtype).to_numpy()

    # ------------ Pressure (Pa → hPa) ------------
    if "airPressure" in var_map and var_map["airPressure"] in df:
        p_pa = pd.to_numeric(df[var_map["airPressure"]], errors="coerce")
        p_hpa = (p_pa / 100.0).astype("float32").to_numpy()
        out["features"]["airPressure"] = p_hpa
        out["airPressure"] = p_hpa

    # QM pressure
    if "qm_airPressure" in var_map:
        candidates = [
            var_map["qm_airPressure"],  # config-preferred
            "PRSSQ1.QMPR",  # NC000001
            "PRESDATA.PRESSQ03.QMPR",  # sometimes present
            "QMPR",  # fallback
        ]
        found = next((c for c in candidates if c in df.columns), None)
        out["qm_airPressure"] = _norm_qm(df[found]) if found else np.full(len(df), -1, dtype="int16")

    # ------------ Air temperature (K → °C) ------------
    if "airTemperature" in var_map and var_map["airTemperature"] in df:
        t_k = pd.to_numeric(df[var_map["airTemperature"]], errors="coerce")
        t_c = (t_k - 273.15).astype("float32").to_numpy()
        out["features"]["airTemperature"] = t_c
        out["airTemperature"] = t_c

    if "qm_airTemperature" in var_map:
        candidates = [var_map["qm_airTemperature"], "TEMHUMDA.QMAT", "QMAT"]
        found = next((c for c in candidates if c in df.columns), None)
        out["qm_airTemperature"] = _norm_qm(df[found]) if found else np.full(len(df), -1, dtype="int16")

    # ------------ Dew point (K → °C) ------------
    if "dewPointTemperature" in var_map and var_map["dewPointTemperature"] in df:
        dp_k = pd.to_numeric(df[var_map["dewPointTemperature"]], errors="coerce")
        dp_c = (dp_k - 273.15).astype("float32").to_numpy()
        out["features"]["dewPointTemperature"] = dp_c
        out["dewPointTemperature"] = dp_c

    if "qm_dewPointTemperature" in var_map:
        candidates = [var_map["qm_dewPointTemperature"], "QMDD"]
        found = next((c for c in candidates if c in df.columns), None)
        out["qm_dewPointTemperature"] = _norm_qm(df[found]) if found else np.full(len(df), -1, dtype="int16")

    # ------------ Relative Humidity (%) ------------
    if "relativeHumidity" in var_map and var_map["relativeHumidity"] in df:
        rh = pd.to_numeric(df[var_map["relativeHumidity"]], errors="coerce").clip(0, 100).astype("float32").to_numpy()
        out["features"]["relativeHumidity"] = rh
        out["relativeHumidity"] = rh

    # ------------ Wind: speed/dir → u, v (m/s) ------------
    if "windSpeed" in var_map and var_map["windSpeed"] in df and "windDirection" in var_map and var_map["windDirection"] in df:

        spd = pd.to_numeric(df[var_map["windSpeed"]], errors="coerce").astype("float64")
        wdir_deg = pd.to_numeric(df[var_map["windDirection"]], errors="coerce").astype("float64")

        # Negative speeds → NaN; wrap direction to [0, 360)
        spd = spd.where(spd >= 0, np.nan)
        wdir_deg = np.mod(wdir_deg, 360.0)

        theta = np.deg2rad(wdir_deg.to_numpy())
        u = (-spd.to_numpy() * np.sin(theta)).astype("float32")
        v = (-spd.to_numpy() * np.cos(theta)).astype("float32")

        # raw inputs (handy for QC/debug)
        out["windSpeed"] = spd.astype("float32").to_numpy()
        out["windDirection"] = wdir_deg.astype("float32").to_numpy()

        # derived features
        out["features"]["wind_u"] = u
        out["features"]["wind_v"] = v
        out["wind_u"] = u
        out["wind_v"] = v

    # QM wind
    if "qm_wind" in var_map:
        candidates = [var_map["qm_wind"], "QMWN", "BSYWND1.QMWN"]
        found = next((c for c in candidates if c in df.columns), None)
        out["qm_wind"] = _norm_qm(df[found]) if found else np.full(len(df), -1, dtype="int16")

    # ------------ Metadata ------------
    if "height" in var_map and var_map["height"] in df:
        h = pd.to_numeric(df[var_map["height"]], errors="coerce").astype("float32").to_numpy()
        out["metadata"]["height"] = h
        out["height"] = h

    return out
