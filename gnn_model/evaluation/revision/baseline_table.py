#!/usr/bin/env python
"""OCELOT vs reference forecasts on a strictly common set of target observations.

References are any ``*_vs_<suffix>.csv`` files produced by evaluation/scripts/compare_to_gfs.py
(GFS, GraphCastGFS staged in GFS layout, ...), plus same-location persistence and the
training-period observation-space climatology. For each (instrument, variable, lead) the
RMSE of every method is computed on the rows where ALL methods are available, with 95%
bootstrap CIs over initializations, and the paired RMSE difference OCELOT - reference.

With ``--base OCELOT,GFS,Persistence`` the methods named there define the verified sample
(reproducing a published comparison such as Fig. 6), and every other method is scored on
the part of that sample where it is available; its paired difference uses OCELOT on the
same rows. The ``n_obs_<method>`` columns give each method's sample size.

Usage:
    python evaluation/revision/baseline_table.py \
      --pred_dir predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space \
      --ref GFS=predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space:vs_gfs \
      --ref GraphCastGFS=predictions/ocelot_v1_2025_graphcastgfs/obs-space:vs_graphcastgfs \
      --clim_dir evaluation/revision/climatology \
      --out evaluation/revision/results/baselines.csv
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import VerifyQC, list_prediction_files, strat_keys, valid_rows  # noqa: E402
from revision_metrics import Climatology  # noqa: E402

# OCELOT variable -> column written by compare_to_gfs.py, per instrument.
REF_COLS = {
    "surface_obs": {"airTemperature": "gfs_t2m_C", "wind_u": "gfs_u10", "wind_v": "gfs_v10",
                    "pressureMeanSeaLevel_prepbufr": "gfs_mslp_hPa"},
    "radiosonde": {"airTemperature": "gfs_airTemperature_C", "wind_u": "gfs_u", "wind_v": "gfs_v"},
    "aircraft": {"airTemperature": "gfs_airTemperature_C", "windU": "gfs_u", "windV": "gfs_v"},
}
KEY_COLS = ["lat", "lon", "obs_time_unix", "lead_hours_nominal", "pressure_hPa"]


def _row_key(df: pd.DataFrame, true_cols: list[str]) -> pd.Series:
    cols = [c for c in KEY_COLS if c in df.columns] + true_cols
    return pd.util.hash_pandas_object(df[cols].round(5), index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--ref", action="append", default=[], help="NAME=DIR:SUFFIX (repeatable)")
    ap.add_argument("--clim_dir", default=None)
    ap.add_argument("--instruments", default="surface_obs,radiosonde,aircraft")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify_qc", default=None, help="YAML of verification-time QC rules (verify_qc.yaml)")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--no_dedup", action="store_true",
                    help="keep repeated prediction rows, as the original evaluation scripts did")
    ap.add_argument("--base", default=None,
                    help="comma list of methods defining the sample; others are scored where available on it")
    args = ap.parse_args()
    base = set(args.base.split(",")) if args.base else None

    refs = []
    for spec in args.ref:
        name, rest = spec.split("=", 1)
        d, suffix = rest.rsplit(":", 1)
        refs.append((name, d, suffix))
    clim = Climatology(args.clim_dir, 10) if args.clim_dir else None
    insts = set(args.instruments.split(","))
    qc = VerifyQC(args.verify_qc)

    recs = []  # one row per (init, inst, var, lead) with per-method SSE on the common subset
    for path, inst, init in list_prediction_files(args.pred_dir):
        if inst not in insts:
            continue
        df = pd.read_csv(path, low_memory=False)
        if qc.active:
            df = qc.attach_flags(df, path, inst)
        varmap = {v: c for v, c in REF_COLS[inst].items() if f"pred_{v}" in df.columns}
        true_cols = [f"true_{v}" for v in varmap]
        df["_key"] = _row_key(df, true_cols)
        if not args.no_dedup:
            df = df.drop_duplicates("_key")
        ok_refs = True
        for name, d, suffix in refs:
            rpath = os.path.join(d, f"pred_{inst}_target_init_{init}_{suffix}.csv")
            if not os.path.exists(rpath):
                if base is not None and name not in base:
                    continue  # scored as missing for this init
                ok_refs = False
                break
            r = pd.read_csv(rpath, low_memory=False)
            r["_key"] = _row_key(r, true_cols)
            keep = ["_key"] + [c for c in set(varmap.values()) if c in r.columns]
            r = r[keep].drop_duplicates("_key").rename(columns={c: f"{name}::{c}" for c in keep if c != "_key"})
            df = df.merge(r, on="_key", how="left")
        if not ok_refs:
            continue
        keys = strat_keys(df, inst, clim.edges.get(inst) if clim else None) if clim else None
        lead = df["lead_hours_nominal"].to_numpy(float)
        for var, rcol in varmap.items():
            o = df[f"true_{var}"].to_numpy(float)
            meth = {"OCELOT": df[f"pred_{var}"].to_numpy(float)}
            if f"persist_{var}" in df.columns:
                meth["Persistence"] = pd.to_numeric(df[f"persist_{var}"], errors="coerce").to_numpy(float)
            if clim is not None:
                meth["Climatology"] = clim.lookup(inst, var, keys)
            for name, _, _ in refs:
                col = f"{name}::{rcol}"
                meth[name] = df[col].to_numpy(float) if col in df.columns else np.full(len(df), np.nan)
            common = valid_rows(df, var)
            if qc.active:
                common, _ = qc.apply(df, inst, var, common, meth.get("Climatology"))
            if base is not None and not base <= set(meth):
                raise SystemExit(f"--base names unknown methods: {sorted(base - set(meth))}")
            for m, f in meth.items():
                if base is None or m in base:
                    common &= np.isfinite(f)
            po = (meth["OCELOT"] - o) ** 2
            for ld in np.unique(lead[common]):
                sel = common & (lead == ld)
                rec = dict(instrument=inst, variable=var, lead=ld, init=init, n=int(sel.sum()))
                for m, f in meth.items():
                    s = sel & np.isfinite(f)
                    rec[f"sse::{m}"] = float(((f[s] - o[s]) ** 2).sum())
                    rec[f"n::{m}"] = int(s.sum())
                    rec[f"sseO::{m}"] = float(po[s].sum())  # OCELOT on the same rows
                recs.append(rec)

    st = pd.DataFrame(recs)
    if st.empty:
        raise SystemExit("No common rows found; check --ref directories/suffixes.")
    st.to_csv(os.path.splitext(args.out)[0] + "_per_init.csv", index=False)
    methods = [c.split("::", 1)[1] for c in st.columns if c.startswith("sse::")]
    rng = np.random.default_rng(args.seed)
    rows = []
    for key, g in st.groupby(["instrument", "variable", "lead"]):
        n = g["n"].to_numpy(float)
        W = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=args.n_boot).astype(float)
        rec = dict(zip(["instrument", "variable", "lead"], key), n_inits=len(g), n_obs=int(n.sum()))
        for m in methods:
            nm = g[f"n::{m}"].to_numpy(float)
            rec[f"n_obs_{m}"] = int(nm.sum())
            if nm.sum() == 0:  # e.g. a 6-hourly reference at +3 h
                rec[f"rmse_{m}"] = rec[f"rmse_{m}_lo"] = rec[f"rmse_{m}_hi"] = np.nan
                continue
            s = g[f"sse::{m}"].to_numpy(float)
            nb = np.maximum(W @ nm, 1)
            rec[f"rmse_{m}"] = np.sqrt(s.sum() / nm.sum())
            rec[f"rmse_{m}_lo"], rec[f"rmse_{m}_hi"] = np.percentile(np.sqrt((W @ s) / nb), [2.5, 97.5])
            if m == "OCELOT":
                continue
            so = g[f"sseO::{m}"].to_numpy(float)
            d = np.sqrt((W @ so) / nb) - np.sqrt((W @ s) / nb)
            rec[f"dRMSE_OCELOT_minus_{m}"] = np.sqrt(so.sum() / nm.sum()) - rec[f"rmse_{m}"]
            rec[f"dRMSE_OCELOT_minus_{m}_lo"], rec[f"dRMSE_OCELOT_minus_{m}_hi"] = np.percentile(d, [2.5, 97.5])
        rows.append(rec)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False, float_format="%.4f")
    print(out[["instrument", "variable", "lead", "n_obs"] + [f"rmse_{m}" for m in methods]].round(3).to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
