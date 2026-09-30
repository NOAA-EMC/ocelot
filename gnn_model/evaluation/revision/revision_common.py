"""Shared helpers for the AIES revision evaluation scripts.

Reads OCELOT obs-space prediction CSVs (``pred_<inst>_target_init_<YYYYMMDDHH>.csv``)
and defines the stratification keys used by the observation-space climatology.

Climatology keys (coarsest fallback last):
    L1: instrument, variable, grid cell, month, 3-h UTC bin, pressure level, scan-angle bin
    L2: instrument, variable, grid cell, month,              pressure level, scan-angle bin
    L3: instrument, variable, 5-degree latitude band, month, pressure level, scan-angle bin
"""

from __future__ import annotations

import glob
import os
import re

import numpy as np
import pandas as pd

CONVENTIONAL = {"surface_obs", "radiosonde", "aircraft"}
FILE_RE = re.compile(r"pred_(?P<inst>.+)_target_init_(?P<init>\d{10})\.csv$")
LAT_BAND_DEG = 5.0
N_SCAN_BINS = 8


def cell_deg_for(inst: str) -> float:
    """Grid-cell size: 1 deg for conventional networks, 2 deg for satellites."""
    return 1.0 if inst in CONVENTIONAL else 2.0


def list_prediction_files(pred_dir: str, recursive: bool = False) -> list[tuple[str, str, str]]:
    """Return (path, instrument, init) for every obs-space prediction CSV."""
    pattern = os.path.join(pred_dir, "**", "pred_*_target_init_*.csv") if recursive else os.path.join(
        pred_dir, "pred_*_target_init_*.csv"
    )
    out = []
    for path in sorted(glob.glob(pattern, recursive=recursive)):
        m = FILE_RE.search(os.path.basename(path))
        if m:  # excludes *_vs_gfs.csv and other derived files
            out.append((path, m.group("inst"), m.group("init")))
    return out


def variables_in(df: pd.DataFrame) -> list[str]:
    return [c[len("true_"):] for c in df.columns if c.startswith("true_") and f"pred_{c[len('true_'):]}" in df.columns]


def valid_rows(df: pd.DataFrame, var: str) -> np.ndarray:
    ok = np.isfinite(df[f"pred_{var}"].to_numpy(float)) & np.isfinite(df[f"true_{var}"].to_numpy(float))
    mcol = f"mask_{var}"
    if mcol in df.columns:
        m = df[mcol]
        if m.dtype != bool:
            m = m.astype(str).str.strip().str.lower().isin({"true", "1", "1.0"})
        ok &= m.to_numpy(bool)
    return ok


def _time_unix(df: pd.DataFrame) -> np.ndarray:
    t = np.full(len(df), -1, dtype=np.int64)
    for col in ("obs_time_unix", "valid_time_unix"):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").fillna(-1).to_numpy(np.int64)
            t = np.where(t >= 0, t, v)
    return t


def scan_bin(df: pd.DataFrame, edges: np.ndarray | None) -> np.ndarray:
    if edges is None or "scan_angle_0" not in df.columns:
        return np.full(len(df), -1, dtype=np.int16)
    sa = np.abs(pd.to_numeric(df["scan_angle_0"], errors="coerce").to_numpy(float))
    b = np.clip(np.searchsorted(edges, sa, side="right") - 1, 0, len(edges) - 2)
    return np.where(np.isfinite(sa), b, -1).astype(np.int16)


def strat_keys(df: pd.DataFrame, inst: str, scan_edges: np.ndarray | None) -> pd.DataFrame:
    """Per-row climatology keys for one prediction CSV."""
    lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(float)
    lon = np.mod(pd.to_numeric(df["lon"], errors="coerce").to_numpy(float) + 180.0, 360.0)
    d = cell_deg_for(inst)
    nlon = int(round(360.0 / d))
    lat_i = np.clip(np.floor((lat + 90.0) / d), 0, int(round(180.0 / d)) - 1)
    lon_i = np.clip(np.floor(lon / d), 0, nlon - 1)
    t = pd.to_datetime(_time_unix(df), unit="s", utc=True)
    plev = (
        pd.to_numeric(df["pressure_level_idx"], errors="coerce").fillna(-1).to_numpy(np.int16)
        if "pressure_level_idx" in df.columns
        else np.full(len(df), -1, dtype=np.int16)
    )
    return pd.DataFrame(
        {
            "cell": (lat_i * nlon + lon_i).astype(np.int32),
            "band": np.clip(np.floor((lat + 90.0) / LAT_BAND_DEG), 0, 180 / LAT_BAND_DEG - 1).astype(np.int16),
            "month": t.month.to_numpy(np.int8),
            "hour3": (t.hour.to_numpy() // 3).astype(np.int8),
            "plev": plev,
            "sab": scan_bin(df, scan_edges),
        }
    )


LEVELS = {
    "L1": ["cell", "month", "hour3", "plev", "sab"],
    "L2": ["cell", "month", "plev", "sab"],
    "L3": ["band", "month", "plev", "sab"],
}


# --------------------------------------------------------------------------------------
# Verification-time quality control
# --------------------------------------------------------------------------------------
class VerifyQC:
    """Extra QC applied to the VERIFYING observations only (model inputs and forecasts are untouched).

    Rules come from a YAML file (see verify_qc.yaml):
      satellite: {value_range: [lo, hi], range_exclude: [inst, ...], outlier_k: k}
      conventional: {<inst>: {<variable>: {flag: <zarr flag column>, keep: [...] | reject: [...],
                                           min_pressure_hpa: p,
                                           le_variable: <other variable>, le_margin: m, max_spread: s}}}
    Flag columns are read from the prediction CSV as ``qm_<flag column>`` (written when the
    instrument's ``export_flag_cols`` is set in the observation config). A rule whose flag column is
    missing raises, so a filter can never be skipped silently.
    """

    def __init__(self, path: str | None, flags_from_dir: str | None = None):
        self.rules = {}
        self.flags_from_dir = flags_from_dir
        if path:
            import yaml
            with open(path) as f:
                self.rules = yaml.safe_load(f) or {}
        self.sat = self.rules.get("satellite") or {}
        self.conv = self.rules.get("conventional") or {}

    @property
    def active(self) -> bool:
        return bool(self.rules)

    def needed_flag_cols(self, inst: str) -> list[str]:
        return sorted({f"qm_{r['flag']}" for r in (self.conv.get(inst) or {}).values() if r.get("flag")})

    def attach_flags(self, df: pd.DataFrame, path: str, inst: str) -> pd.DataFrame:
        """Ensure the qm_* columns a rule needs are present; optionally take them, by row position,
        from the same-named file in ``flags_from_dir`` (identical targets, e.g. a denial run)."""
        need = [c for c in self.needed_flag_cols(inst) if c not in df.columns]
        if not need:
            return df
        if not self.flags_from_dir:
            raise SystemExit(
                f"{os.path.basename(path)} lacks {need}. Re-run the predictions with `export_flag_cols` set for "
                f"'{inst}' in the observation config, or pass --flags_from_dir pointing at such a run."
            )
        src = os.path.join(self.flags_from_dir, os.path.basename(path))
        if not os.path.exists(src):
            raise SystemExit(f"No flag source for {os.path.basename(path)} in {self.flags_from_dir}")
        ref = pd.read_csv(src, usecols=["lat", "lon"] + need)
        if len(ref) != len(df) or not np.allclose(ref["lat"].to_numpy(float), df["lat"].to_numpy(float), atol=1e-4,
                                                  equal_nan=True):
            raise SystemExit(f"Rows of {src} do not line up with {path}; cannot borrow QC flags.")
        for c in need:
            df[c] = ref[c].to_numpy()
        return df

    def apply(self, df: pd.DataFrame, inst: str, var: str, ok: np.ndarray, clim: np.ndarray | None = None):
        """Return (ok_after_qc, n_removed_by_rule_dict)."""
        removed = {"range": 0, "flag": 0, "pressure": 0, "relation": 0, "outlier": 0}
        if not self.active:
            return ok, removed
        ok = ok.copy()
        obs = df[f"true_{var}"].to_numpy(float)
        if inst not in CONVENTIONAL:
            rng = self.sat.get("value_range")
            if rng and inst not in set(self.sat.get("range_exclude") or []):
                bad = ok & ~((obs >= float(rng[0])) & (obs <= float(rng[1])))
                removed["range"] = int(bad.sum())
                ok &= ~bad
            k = self.sat.get("outlier_k")
            if k and ok.sum() > 50:
                # Robust gross-error check on the observation's departure from climatology
                # (or from the sample median when no climatology is available). Uses observations only.
                dep = obs - clim if clim is not None and np.isfinite(clim[ok]).mean() > 0.5 else obs.copy()
                use = ok & np.isfinite(dep)
                med = np.median(dep[use])
                sig = 1.4826 * np.median(np.abs(dep[use] - med))
                if sig > 0:
                    bad = use & (np.abs(dep - med) > float(k) * sig)
                    removed["outlier"] = int(bad.sum())
                    ok &= ~bad
        else:
            rule = (self.conv.get(inst) or {}).get(var)
            if rule:
                if rule.get("flag"):
                    flag = pd.to_numeric(df[f"qm_{rule['flag']}"], errors="coerce").to_numpy(float)
                    if ok.any() and not np.isfinite(flag[ok]).any():
                        raise SystemExit(
                            f"QC flag column '{rule['flag']}' is empty for {inst}/{var}: check that this column "
                            f"exists in the archive and is listed in export_flag_cols."
                        )
                    if "keep" in rule:
                        good = np.isin(flag, [float(v) for v in rule["keep"]])
                    else:
                        good = ~np.isin(flag, [float(v) for v in rule.get("reject", [])])
                    bad = ok & ~good
                    removed["flag"] = int(bad.sum())
                    ok &= ~bad
                if rule.get("min_pressure_hpa") is not None:
                    p = pd.to_numeric(df["pressure_hPa"], errors="coerce").to_numpy(float)
                    bad = ok & ~(p >= float(rule["min_pressure_hpa"]))
                    removed["pressure"] = int(bad.sum())
                    ok &= ~bad
                if rule.get("le_variable"):
                    # Cross-variable check, e.g. dew point <= air temperature (+ margin), and a cap on
                    # their spread; both read from the observed values of the same report.
                    other = pd.to_numeric(df[f"true_{rule['le_variable']}"], errors="coerce").to_numpy(float)
                    both = np.isfinite(other) & np.isfinite(obs)
                    bad = ok & both & (obs > other + float(rule.get("le_margin", 0.0)))
                    if rule.get("max_spread") is not None:
                        bad |= ok & both & ((other - obs) > float(rule["max_spread"]))
                    removed["relation"] = int(bad.sum())
                    ok &= ~bad
        return ok, removed
