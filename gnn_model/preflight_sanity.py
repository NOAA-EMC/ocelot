import importlib
import numpy as np
import pandas as pd
import my_loaders
from nnja_adapter import build_zlike_from_df

# ------------ config ------------
start, end = "2024-04-01", "2024-04-15"
dataset_ids = ("conv-adpsfc-NC000101", "conv-adpsfc-NC000001")

cols = [
    # coords/meta
    "OBS_TIMESTAMP",
    "LAT",
    "LON",
    "SELV",
    "RPID",
    # canonical vars + QM
    "PRESDATA.PRESSQ03.PRES",
    "QMPR",
    "TEMHUMDA.TMDB",
    "QMAT",
    "TEMHUMDA.TMDP",
    "QMDD",
    "TEMHUMDA.REHU",
    "BSYWND1.WSPD",
    "BSYWND1.WDIR",
    "QMWN",
]

var_map = {
    "airPressure": "PRESDATA.PRESSQ03.PRES",
    "qm_airPressure": "QMPR",
    "airTemperature": "TEMHUMDA.TMDB",
    "qm_airTemperature": "QMAT",
    "dewPointTemperature": "TEMHUMDA.TMDP",
    "qm_dewPointTemperature": "QMDD",
    "relativeHumidity": "TEMHUMDA.REHU",
    "windSpeed": "BSYWND1.WSPD",
    "windDirection": "BSYWND1.WDIR",
    "qm_wind": "QMWN",
    "height": "SELV",
}

# QC policy (matches your YAML)
rng = {
    "airPressure": (300, 1200),  # hPa (adapter converts Pa->hPa)
    "airTemperature": (-80, 60),  # °C
    "dewPointTemperature": (-100, 40),
    "relativeHumidity": (0, 100),  # %
    "wind_u": (-75, 75),
    "wind_v": (-75, 75),
}
keep_set = {-1, 0, 1, 2, 4}  # exclude 14

# ------------ load 101+001 together ------------
importlib.reload(my_loaders)
df = my_loaders.load_adpsfc(
    start,
    end,
    columns=cols,
    dataset_ids=dataset_ids,
    mirror="gcp_brightband",
    drop_dupes=True,  # prefer 101, drop rows with QM=14, prefer more-populated targets
)
print(f"shape: {df.shape}")
print("columns present? ", set(cols).issubset(df.columns))


# ------------ QM histograms ------------
def qm_hist(series):
    s = pd.to_numeric(series, errors="coerce").fillna(-1).astype(np.int64)
    v, c = np.unique(s, return_counts=True)
    return dict(zip(v, c))


qm_cols = ["QMPR", "QMAT", "QMDD", "QMWN"]
for q in qm_cols:
    if q in df.columns:
        print(f"{q} hist:", qm_hist(df[q]))

# ------------ adapter & QC ------------
z = build_zlike_from_df(df, var_map, time_col="OBS_TIMESTAMP", lat_col="LAT", lon_col="LON")

# Valid masks by variable
masks = {}
# range checks
for k, (lo, hi) in rng.items():
    if k in z:
        x = pd.to_numeric(pd.Series(z[k]), errors="coerce")
        masks[k] = x.between(lo, hi)
    else:
        masks[k] = pd.Series(False, index=np.arange(len(df)))


# QM keep (where available)
def keep_qm(arr):
    if arr is None:  # RH has no QM — treat as all keepable
        return pd.Series(True, index=np.arange(len(df)))
    return pd.Series(np.isin(arr, list(keep_set)))


qm_keep = {
    "airPressure": keep_qm(z.get("qm_airPressure")),
    "airTemperature": keep_qm(z.get("qm_airTemperature")),
    "dewPointTemperature": keep_qm(z.get("qm_dewPointTemperature")),
    "wind_u": keep_qm(z.get("qm_wind")),
    "wind_v": keep_qm(z.get("qm_wind")),
    "relativeHumidity": pd.Series(True, index=np.arange(len(df))),
}

kept = {}
for k in ["airPressure", "airTemperature", "dewPointTemperature", "relativeHumidity", "wind_u", "wind_v"]:
    rmask = masks[k] if k in masks else pd.Series(True, index=np.arange(len(df)))
    qmask = qm_keep[k]
    kept[k] = (rmask & qmask).sum()

print("\nQC kept counts (range + QM keep):")
for k in ["airPressure", "airTemperature", "dewPointTemperature", "relativeHumidity", "wind_u", "wind_v"]:
    print(f"  {k:20s}: {kept[k]}")

# ------------ physical sanity ------------
# Dew point <= temperature
if ("airTemperature" in z) and ("dewPointTemperature" in z):
    t = pd.Series(z["airTemperature"])
    td = pd.Series(z["dewPointTemperature"])
    bad = (td > t).sum()
    print(f"\nDewpoint > Temperature violations: {int(bad)} / {len(t)}")

# Wind checks (compare back to speed/dir if available)
if ("wind_u" in z) and ("wind_v" in z):
    u = pd.Series(z["wind_u"])
    v = pd.Series(z["wind_v"])
    spd_rec = np.hypot(u, v)
    spd_src = pd.to_numeric(pd.Series(z.get("windSpeed", np.full_like(u, np.nan))), errors="coerce")
    if np.isfinite(spd_src).any():
        se = np.abs(spd_rec - spd_src)
        print(f"\nWind speed |abs(rec - src)|: median={np.nanmedian(se):.2f} m/s, 95%={np.nanpercentile(se,95):.2f} m/s")

    # Direction from (u,v): met convention atan2(-u, -v)
    wdir_src = pd.to_numeric(pd.Series(z.get("windDirection", np.full_like(u, np.nan))), errors="coerce")
    tdir = (np.degrees(np.arctan2(-u, -v)) + 360.0) % 360.0

    def shortest_arc(a, b):
        return np.abs(((a - b + 180.0) % 360.0) - 180.0)

    if np.isfinite(wdir_src).any():
        dang = shortest_arc(tdir, wdir_src)
        print(f"Wind direction error: median={np.nanmedian(dang):.1f}°, 95%={np.nanpercentile(dang,95):.1f}°")

# ------------ coverage in 12h bins ------------
t = pd.to_datetime(df["OBS_TIMESTAMP"], utc=True, errors="coerce")
bins = t.dt.floor("12H")
counts = bins.value_counts().sort_index()
print(f"\n12h bins: {counts.size}, zero bins: {(counts==0).sum()}")
print(counts.head(5).to_string())


# ------------ feature stats after QC (to compare with config) ------------
def masked_stats(name, arr, mask):
    x = pd.to_numeric(pd.Series(arr), errors="coerce")
    x = x[mask]
    return float(np.nanmean(x)), float(np.nanstd(x)), int(x.notna().sum())


print("\nFeature stats AFTER QC (mean, std, N):")
for k in ["airPressure", "airTemperature", "dewPointTemperature", "relativeHumidity", "wind_u", "wind_v"]:
    if k in z:
        # combine range+QM masks
        rmask = masks.get(k, pd.Series(True, index=np.arange(len(df))))
        qmask = qm_keep[k]
        m = (rmask & qmask).to_numpy()
        mu, sd, n = masked_stats(k, z[k], m)
        unit = {
            "airPressure": "hPa",
            "airTemperature": "°C",
            "dewPointTemperature": "°C",
            "relativeHumidity": "pp",
            "wind_u": "m/s",
            "wind_v": "m/s",
        }[k]
        print(f"  {k:20s}: [{mu:.2f}, {sd:.2f}]  N={n}  ({unit})")
