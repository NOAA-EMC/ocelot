#!/usr/bin/env python
"""Systematic observation-space verification for the AIES revision.

For EVERY prognostic target (all instruments, all channels/variables, all leads, and per
pressure level for radiosonde/aircraft) this computes, from the obs-space prediction CSVs:

  * N, bias, RMSE, MAE                       (as in the original manuscript)
  * Pearson correlation r(f, o)
  * anomaly correlation ACC against a training-period (2015-2023) observation-space
    climatology: centered (mean anomaly removed) and uncentered
  * RMSE of the climatology forecast and MSE skill score  MSESS_clim = 1 - MSE / MSE_clim
  * persistence RMSE and MSESS_pers on the persistence-valid subset
  * spatial error-coherence index  C_LS = (S_LS - S_0) / (SSE - S_0), where
    S_LS = sum_i (mean error of obs i's 5-deg box)^2 and S_0 is its expectation if the same
    errors were randomly permuted among locations (spatially uncorrelated errors).
    C_LS ~ 0 for noise-like errors and -> 1 when errors are organized at >= 5-deg scales
  * 95% bootstrap confidence intervals obtained by resampling forecast initializations

Per-initialization sufficient statistics are written first, so the bootstrap and any
re-grouping are cheap.

Usage:
    python evaluation/revision/revision_metrics.py \
        --pred_dir predictions/ocelot_v1_2025_gfs_eval/pred_csv/obs-space \
        --clim_dir evaluation/revision/climatology \
        --out_dir  evaluation/revision/results/v1_2025
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from revision_common import LEVELS, VerifyQC, list_prediction_files, strat_keys, valid_rows, variables_in  # noqa: E402

LS_BOX_DEG = 5.0
STAT_COLS = [
    "n", "se", "sse", "sae", "sf", "so", "sff", "soo", "sfo",
    "nc", "sse_m_c", "sse_clim", "sfa", "soa", "sfaa", "soaa", "sfaoa",
    "np", "sse_m_p", "sse_pers", "sse_ls", "sse_ls0",
]


class Climatology:
    def __init__(self, clim_dir: str | None, min_count: int):
        self.dir = clim_dir
        self.min_count = min_count
        self.cache: dict[str, dict[str, pd.DataFrame] | None] = {}
        self.edges = {}
        if clim_dir and os.path.exists(os.path.join(clim_dir, "scan_edges.json")):
            with open(os.path.join(clim_dir, "scan_edges.json")) as f:
                self.edges = {k: np.asarray(v) for k, v in json.load(f).items()}

    def _load(self, inst: str):
        if inst not in self.cache:
            path = os.path.join(self.dir, f"clim_{inst}.csv.gz") if self.dir else ""
            if not path or not os.path.exists(path):
                self.cache[inst] = None
            else:
                t = pd.read_csv(path)
                t = t[t["count"] >= self.min_count]
                # {level: {variable: table}} so each lookup merges against one variable's rows only.
                self.cache[inst] = {
                    lvl: {var: g[cols + ["mean"]].astype({c: "int64" for c in cols})
                          for var, g in t[t["level"] == lvl].groupby("var")}
                    for lvl, cols in LEVELS.items()
                }
        return self.cache[inst]

    def lookup(self, inst: str, var: str, keys: pd.DataFrame) -> np.ndarray:
        tables = self._load(inst)
        out = np.full(len(keys), np.nan)
        if tables is None:
            return out
        k = keys.reset_index(drop=True).astype("int64")
        for lvl, cols in LEVELS.items():
            need = np.isnan(out)
            if not need.any():
                break
            t = tables[lvl].get(var)
            if t is None or t.empty:
                continue
            m = k.loc[need, cols].merge(t, on=cols, how="left")["mean"].to_numpy(float)
            out[need] = m
        return out


def _ls_sse(err: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> tuple[float, float]:
    """Large-scale (5-degree box-mean) squared error and its random-permutation expectation."""
    nlon = int(360 / LS_BOX_DEG)
    box = (np.floor((lat + 90) / LS_BOX_DEG) * nlon + np.floor(np.mod(lon + 180, 360) / LS_BOX_DEG)).astype(np.int64)
    s = pd.Series(err).groupby(box).agg(["sum", "count"])
    s_ls = float(((s["sum"] / s["count"]) ** 2 * s["count"]).sum())
    n = err.size
    # E[sum_b k_b * mean_b^2] under permutation = n_boxes*var + n*mean^2 (finite-population form)
    var = float(err.var()) * (n / (n - 1)) if n > 1 else 0.0
    fpc = (n - s["count"]) / max(n, 1)  # finite-population correction per box
    s0 = float((var * fpc).sum()) + n * float(err.mean()) ** 2
    return s_ls, min(s0, float((err * err).sum()))


def per_init_stats(files, clim: Climatology, min_level_n: int, qc: VerifyQC | None = None):
    """Return (per-init sufficient statistics, per-init QC removal counts)."""
    qc = qc or VerifyQC(None)
    rows, qc_rows = [], []
    for i, (path, inst, init) in enumerate(files):
        df = pd.read_csv(path, low_memory=False)
        if df.empty or "lead_hours_nominal" not in df.columns:
            continue
        if qc.active:
            df = qc.attach_flags(df, path, inst)
        keys = strat_keys(df, inst, clim.edges.get(inst))
        lead = pd.to_numeric(df["lead_hours_nominal"], errors="coerce").to_numpy(float)
        plev_lab = df["pressure_level_label"].astype(str).to_numpy() if "pressure_level_label" in df.columns else None
        lat = pd.to_numeric(df["lat"], errors="coerce").to_numpy(float)
        lon = pd.to_numeric(df["lon"], errors="coerce").to_numpy(float)
        for var in variables_in(df):
            ok = valid_rows(df, var)
            if not ok.any():
                continue
            f_all = df[f"pred_{var}"].to_numpy(float)
            o_all = df[f"true_{var}"].to_numpy(float)
            c_all = clim.lookup(inst, var, keys)
            if qc.active:
                n_before = int(ok.sum())
                ok, removed = qc.apply(df, inst, var, ok, c_all)
                qc_rows.append(dict(instrument=inst, variable=var, init=init, n_before=n_before, **removed))
                if not ok.any():
                    continue
            p_all = pd.to_numeric(df[f"persist_{var}"], errors="coerce").to_numpy(float) if f"persist_{var}" in df else np.full(len(df), np.nan)
            groups = [("all", np.ones(len(df), bool))]
            if plev_lab is not None:
                groups += [(lab, plev_lab == lab) for lab in np.unique(plev_lab) if lab not in ("unknown", "nan")]
            for plev, gmask in groups:
                for ld in np.unique(lead[np.isfinite(lead)]):
                    sel = ok & gmask & (lead == ld)
                    n = int(sel.sum())
                    if n < (1 if plev == "all" else min_level_n):
                        continue
                    f, o, c, p = f_all[sel], o_all[sel], c_all[sel], p_all[sel]
                    e = f - o
                    r = dict(instrument=inst, variable=var, plev=plev, lead=ld, init=init, n=n,
                             se=e.sum(), sse=(e * e).sum(), sae=np.abs(e).sum(),
                             sf=f.sum(), so=o.sum(), sff=(f * f).sum(), soo=(o * o).sum(), sfo=(f * o).sum(),
                             )
                    r["sse_ls"], r["sse_ls0"] = _ls_sse(e, lat[sel], lon[sel])
                    hc = np.isfinite(c)
                    fa, oa = f[hc] - c[hc], o[hc] - c[hc]
                    r.update(nc=int(hc.sum()), sse_m_c=(e[hc] ** 2).sum(), sse_clim=(oa ** 2).sum(),
                             sfa=fa.sum(), soa=oa.sum(), sfaa=(fa * fa).sum(), soaa=(oa * oa).sum(), sfaoa=(fa * oa).sum())
                    hp = np.isfinite(p)
                    r.update(np=int(hp.sum()), sse_m_p=(e[hp] ** 2).sum(), sse_pers=((p[hp] - o[hp]) ** 2).sum())
                    rows.append(r)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(files)} files", flush=True)
    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


def write_qc_summary(qc_counts: pd.DataFrame, out_dir: str) -> None:
    """Fraction of verifying observations removed by each verification-QC rule, per target."""
    if qc_counts is None or qc_counts.empty:
        return
    rules = [c for c in ("range", "flag", "pressure", "relation", "outlier", "level") if c in qc_counts.columns]
    g = qc_counts.groupby(["instrument", "variable"])[["n_before"] + rules].sum()
    for c in rules:
        g[f"pct_{c}"] = 100.0 * g[c] / g["n_before"]
    g["pct_removed_total"] = g[[f"pct_{c}" for c in rules]].sum(axis=1)
    g.to_csv(os.path.join(out_dir, "qc_removal_summary.csv"), float_format="%.4g")


def _acc_per_init(g: pd.DataFrame, centered: bool, min_n: int) -> np.ndarray:
    nc = g["nc"].to_numpy(float)
    if centered:
        cov = g["sfaoa"] / nc - (g["sfa"] / nc) * (g["soa"] / nc)
        vf = g["sfaa"] / nc - (g["sfa"] / nc) ** 2
        vo = g["soaa"] / nc - (g["soa"] / nc) ** 2
        acc = cov / np.sqrt(vf * vo)
    else:
        acc = g["sfaoa"] / np.sqrt(g["sfaa"] * g["soaa"])
    acc = acc.to_numpy(float, copy=True)
    acc[nc < min_n] = np.nan
    return acc


def summarize(stats: pd.DataFrame, n_boot: int, seed: int, min_acc_n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    out = []
    for key, g in stats.groupby(["instrument", "variable", "plev", "lead"], sort=True):
        g = g.reset_index(drop=True)
        S = g[STAT_COLS].sum()
        n = S["n"]
        mf, mo = S["sf"] / n, S["so"] / n
        corr = (S["sfo"] / n - mf * mo) / np.sqrt((S["sff"] / n - mf ** 2) * (S["soo"] / n - mo ** 2))
        acc_c = _acc_per_init(g, True, min_acc_n)
        acc_u = _acc_per_init(g, False, min_acc_n)
        rec = dict(zip(["instrument", "variable", "plev", "lead"], key))
        rec.update(
            n_inits=len(g), n_obs=int(n), bias=S["se"] / n, rmse=np.sqrt(S["sse"] / n), mae=S["sae"] / n, corr=corr,
            acc_centered_mean=np.nanmean(acc_c) if np.isfinite(acc_c).any() else np.nan,
            acc_uncentered_mean=np.nanmean(acc_u) if np.isfinite(acc_u).any() else np.nan,
            n_obs_clim=int(S["nc"]),
            rmse_clim=np.sqrt(S["sse_clim"] / S["nc"]) if S["nc"] else np.nan,
            msess_clim=1 - S["sse_m_c"] / S["sse_clim"] if S["sse_clim"] > 0 else np.nan,
            n_obs_pers=int(S["np"]),
            rmse_model_on_pers=np.sqrt(S["sse_m_p"] / S["np"]) if S["np"] else np.nan,
            rmse_pers=np.sqrt(S["sse_pers"] / S["np"]) if S["np"] else np.nan,
            msess_pers=1 - S["sse_m_p"] / S["sse_pers"] if S["sse_pers"] > 0 else np.nan,
            error_coherence=(S["sse_ls"] - S["sse_ls0"]) / (S["sse"] - S["sse_ls0"]) if S["sse"] > S["sse_ls0"] else np.nan,
        )
        if n_boot > 0 and len(g) > 1:
            W = rng.multinomial(len(g), np.full(len(g), 1 / len(g)), size=n_boot).astype(float)
            A = W @ g[STAT_COLS].to_numpy(float)
            ix = {c: j for j, c in enumerate(STAT_COLS)}

            def ci(x):
                x = x[np.isfinite(x)]
                return (np.percentile(x, 2.5), np.percentile(x, 97.5)) if x.size else (np.nan, np.nan)

            with np.errstate(divide="ignore", invalid="ignore"):
                rec["rmse_lo"], rec["rmse_hi"] = ci(np.sqrt(A[:, ix["sse"]] / A[:, ix["n"]]))
                rec["msess_clim_lo"], rec["msess_clim_hi"] = ci(1 - A[:, ix["sse_m_c"]] / A[:, ix["sse_clim"]])
                rec["msess_pers_lo"], rec["msess_pers_hi"] = ci(1 - A[:, ix["sse_m_p"]] / A[:, ix["sse_pers"]])
                a = np.nan_to_num(acc_c, nan=0.0)
                valid = np.isfinite(acc_c).astype(float)
                rec["acc_lo"], rec["acc_hi"] = ci((W @ a) / (W @ valid))
        out.append(rec)
    return pd.DataFrame(out)


def _run_fingerprint(args) -> str:
    """Identify the configuration a shard ran under, so a merge can refuse to mix two runs.

    Concurrent runs writing the same out_dir overwrite each other's shard files one index at a
    time, which silently produces a merged result built from two different configurations.
    """
    import hashlib
    parts = [os.path.abspath(args.pred_dir), str(args.clim_dir), str(args.min_clim_count),
             str(args.min_level_n), str(bool(args.recursive))]
    for path in (args.verify_qc, args.flags_from_dir):
        if path and os.path.exists(path) and os.path.isfile(path):
            parts.append(open(path, "rb").read().decode("utf-8", "replace"))
        else:
            parts.append(str(path))
    if args.clim_dir:
        for f in sorted(glob.glob(os.path.join(args.clim_dir, "*"))):
            parts.append(f"{os.path.basename(f)}:{os.path.getsize(f)}:{int(os.path.getmtime(f))}")
    return hashlib.blake2b("|".join(parts).encode(), digest_size=8).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred_dir", required=True)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--clim_dir", default=None, help="Output of build_obs_climatology.py (omit to skip ACC/climatology)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--init_start", default=None, help="YYYYMMDDHH (inclusive)")
    ap.add_argument("--init_end", default=None, help="YYYYMMDDHH (inclusive)")
    ap.add_argument("--min_clim_count", type=int, default=10)
    ap.add_argument("--min_level_n", type=int, default=20, help="Min obs per init for a pressure-level row")
    ap.add_argument("--min_acc_n", type=int, default=30, help="Min obs per init for a per-init ACC")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--reuse_stats", action="store_true", help="Reuse per_init_stats.csv if present")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="Process only every N-th file starting at I (0-based) and write "
                         "<out_dir>/shards/per_init_stats_III.csv; run as a Slurm array, then --merge_shards")
    ap.add_argument("--merge_shards", action="store_true",
                    help="Concatenate <out_dir>/shards/*.csv into per_init_stats.csv and write the summaries")
    ap.add_argument("--verify_qc", default=None,
                    help="YAML of verification-time QC rules applied to the verifying observations (verify_qc.yaml)")
    ap.add_argument("--flags_from_dir", default=None,
                    help="Directory of same-named prediction CSVs to borrow qm_* flag columns from (by row position)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    stats_path = os.path.join(args.out_dir, "per_init_stats.csv")
    shard_dir = os.path.join(args.out_dir, "shards")
    if args.merge_shards:
        parts = sorted(glob.glob(os.path.join(shard_dir, "per_init_stats_*.csv")))
        done = sorted(glob.glob(os.path.join(shard_dir, "per_init_stats_*.done")))
        if not parts:
            raise SystemExit(f"No shard files under {shard_dir}")
        stamps = [open(f).read().strip().split() for f in done]
        n_expected = int(stamps[0][0].split("/")[1]) if stamps else len(parts)
        if len(done) != n_expected:
            raise SystemExit(f"Only {len(done)} of {n_expected} shards finished; rerun the missing array tasks first.")
        marks = {t[1] for t in stamps if len(t) > 1}
        if len(marks) > 1:
            raise SystemExit(
                f"The shards in {shard_dir} come from {len(marks)} different configurations ({', '.join(sorted(marks))}). "
                f"Two runs wrote the same out_dir, so the merge would mix them. Delete {shard_dir} and run once."
            )
        stats = pd.concat([pd.read_csv(p, dtype={"init": str}) for p in parts], ignore_index=True)
        stats.to_csv(stats_path, index=False)
        qparts = sorted(glob.glob(os.path.join(shard_dir, "qc_counts_*.csv")))
        if qparts:
            write_qc_summary(pd.concat([pd.read_csv(p, dtype={"init": str}) for p in qparts], ignore_index=True), args.out_dir)
        print(f"Merged {len(parts)} shards: {len(stats)} rows")
    elif args.reuse_stats and os.path.exists(stats_path):
        stats = pd.read_csv(stats_path, dtype={"init": str})
    else:
        files = list_prediction_files(args.pred_dir, args.recursive)
        if args.init_start:
            files = [f for f in files if f[2] >= args.init_start]
        if args.init_end:
            files = [f for f in files if f[2] <= args.init_end]
        if not files:
            raise SystemExit(f"No prediction CSVs found under {args.pred_dir}")
        if args.shard:
            i, n = (int(x) for x in args.shard.split("/"))
            files = files[i::n]
            os.makedirs(shard_dir, exist_ok=True)
            print(f"Shard {i}/{n}: {len(files)} files", flush=True)
            qc = VerifyQC(args.verify_qc, args.flags_from_dir)
            stats, qc_counts = per_init_stats(files, Climatology(args.clim_dir, args.min_clim_count), args.min_level_n, qc)
            stats.to_csv(os.path.join(shard_dir, f"per_init_stats_{i:03d}.csv"), index=False)
            if not qc_counts.empty:
                qc_counts.to_csv(os.path.join(shard_dir, f"qc_counts_{i:03d}.csv"), index=False)
            with open(os.path.join(shard_dir, f"per_init_stats_{i:03d}.done"), "w") as f:
                f.write(f"{i}/{n} {_run_fingerprint(args)}")
            print(f"Shard {i}/{n} done: {len(stats)} rows")
            return 0
        print(f"Computing per-init statistics for {len(files)} files ...")
        qc = VerifyQC(args.verify_qc, args.flags_from_dir)
        stats, qc_counts = per_init_stats(files, Climatology(args.clim_dir, args.min_clim_count), args.min_level_n, qc)
        stats.to_csv(stats_path, index=False)
        write_qc_summary(qc_counts, args.out_dir)

    summary = summarize(stats, args.n_boot, args.seed, args.min_acc_n)
    summary.to_csv(os.path.join(args.out_dir, "metrics_summary.csv"), index=False)
    # Compact table (all-level rows only) for the manuscript/supplement.
    cols = ["instrument", "variable", "lead", "n_inits", "n_obs", "bias", "rmse", "corr", "acc_centered_mean",
            "rmse_clim", "msess_clim", "rmse_pers", "msess_pers", "error_coherence"]
    compact = summary[summary["plev"] == "all"].copy()
    # Same-location persistence is only meaningful where a prior observation at the same location
    # usually exists (fixed networks). Blank it where < 20% of observations have a match (satellites).
    sparse = compact["n_obs_pers"] < 0.2 * compact["n_obs"]
    compact.loc[sparse, ["rmse_pers", "msess_pers"]] = np.nan
    compact = compact[[c for c in cols if c in compact.columns]]
    compact.to_csv(os.path.join(args.out_dir, "metrics_compact_all_targets.csv"), index=False, float_format="%.4g")
    print(f"Wrote {len(summary)} rows to {args.out_dir}/metrics_summary.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
