import numpy as np
import pandas as pd


def _to_time_seconds(series: pd.Series) -> np.ndarray:
    # Always coerce to timezone-aware, handle any incoming resolution
    s = pd.to_datetime(series, utc=True, errors="coerce")
    # Determine resolution of the dtype (ns/us/ms/s)
    unit = getattr(s.dtype, "unit", "ns")
    denom = {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1_000, "s": 1}.get(unit, 1_000_000_000)
    # Convert to *seconds since epoch* int64
    return (s.astype("int64", copy=False) // denom).astype("int64")


def build_zlike_from_df(
    df,
    var_map,
    time_col="OBS_TIMESTAMP",
    lat_col="LAT",
    lon_col="LON",
):
    # Parse to UTC and keep both ns and seconds since epoch
    t = pd.to_datetime(df[time_col], utc=True, errors="coerce")
    mask = t.notna()
    if not mask.all():
        df = df.loc[mask].reset_index(drop=True)
        t = t.loc[mask]
    # Robust seconds-since-epoch regardless of dtype unit (ns/us/ms/s)
    unit = getattr(t.dtype, "unit", "ns")
    denom = {"ns": 1_000_000_000, "us": 1_000_000, "ms": 1_000, "s": 1}.get(unit, 1_000_000_000)
    t_i64 = t.astype("int64", copy=False)
    time_s = (t_i64 // denom).astype("int64")

    # Normalize to ns explicitly so it's correct for any unit:
    scale_to_ns = {"ns": 1, "us": 1_000, "ms": 1_000_000, "s": 1_000_000_000}.get(unit, 1)
    time_ns = (t_i64 * scale_to_ns).astype("int64")

    out = {
        "time": time_s.to_numpy(),  # seconds (what organize_bins_times/extract_features use)
        "zar_time": time_ns.to_numpy(),  # ns (used for sorting elsewhere)
        "latitude": df[lat_col].astype("float32").to_numpy(),
        "longitude": df[lon_col].astype("float32").to_numpy(),
        # Keep these for compatibility if other code references them
        "features": {},
        "metadata": {},
    }

    # Convert Pa → hPa to match feature_stats/QC ranges
    if "airPressure" in var_map:
        p_hpa = (df[var_map["airPressure"]].astype("float64") / 100.0).to_numpy().astype("float32")
        out["features"]["airPressure"] = p_hpa
        out["airPressure"] = p_hpa  # <-- flat alias (required by current pipeline)

    if "height" in var_map:
        h = df[var_map["height"]].astype("float32").to_numpy()
        out["metadata"]["height"] = h
        out["height"] = h  # <-- flat alias

    if "qm_airPressure" in var_map:
        name = var_map["qm_airPressure"]
        cand = [name, "PRSSQ1.QMPR", "PRESDATA.PRESSQ03.QMPR"]
        found = next((c for c in cand if c in df.columns), None)
        if found:
            flags = pd.to_numeric(df[found], errors="coerce").fillna(-1).astype("int16").to_numpy()
        else:
            print(f"[NNJA ADAPTER] QM flag column not found ({cand}); filling -1.")
            flags = np.full(len(df), -1, dtype="int16")
        out["qm_airPressure"] = flags

    return out
