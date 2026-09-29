#!/usr/bin/env python
"""Figures for the AIES revision.

  scorecard  : MSESS vs climatology (or ACC) for every target channel/variable x lead   (new Fig. 8)
  baselines  : RMSE vs lead for OCELOT, GFS, GraphCastGFS, persistence, climatology      (revised Fig. 6)
  denial     : % MSE change per denial group x target instrument                       (new Fig. 9)
  rollout    : RMSE / MSESS_clim vs lead out to 48 h for selected targets               (new Fig. 10)

Usage examples:
  python evaluation/revision/revision_plots.py scorecard --summary results/v1_2025/metrics_summary.csv --out fig8.png
  python evaluation/revision/revision_plots.py baselines --table results/baselines.csv --out fig6.png
  python evaluation/revision/revision_plots.py denial --summary results/denial/denial_summary.csv --out fig9.png
  python evaluation/revision/revision_plots.py rollout --summary results/rollout_48h/metrics_summary.csv --out fig10.png
"""

from __future__ import annotations

import argparse
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["atms", "amsua", "ssmis", "avhrr", "seviri_asr", "seviri_csr", "ascat", "aircraft", "radiosonde", "surface_obs"]
LABEL = {"atms": "ATMS", "amsua": "AMSU-A", "ssmis": "SSMIS", "avhrr": "AVHRR", "seviri_asr": "SEVIRI ASR",
         "seviri_csr": "SEVIRI CSR", "ascat": "ASCAT", "aircraft": "Aircraft", "radiosonde": "Radiosonde",
         "surface_obs": "Surface"}


def _ord(inst):
    return ORDER.index(inst) if inst in ORDER else len(ORDER)


def scorecard(a):
    s = pd.read_csv(a.summary)
    s = s[s["plev"] == "all"]
    s["row"] = s["instrument"].map(lambda x: LABEL.get(x, x)) + " " + s["variable"].astype(str)
    s["o"] = s["instrument"].map(_ord)
    piv = s.pivot_table(index=["o", "row"], columns="lead", values=a.metric).sort_index()
    fig, ax = plt.subplots(figsize=(2.2 + 0.9 * piv.shape[1], 0.18 * len(piv) + 1.2))
    vmax = 1.0
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="RdBu", vmin=-vmax if a.metric.startswith("msess") else 0, vmax=vmax)
    ax.set_yticks(range(len(piv)), [r for _, r in piv.index], fontsize=6)
    ax.set_xticks(range(piv.shape[1]), [f"+{int(c)} h" for c in piv.columns])
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5)
    fig.colorbar(im, ax=ax, label={"msess_clim": "MSESS vs climatology", "acc_centered_mean": "ACC",
                                   "msess_pers": "MSESS vs persistence"}.get(a.metric, a.metric))
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)


def baselines(a):
    t = pd.read_csv(a.table)
    methods = [c[5:] for c in t.columns if c.startswith("rmse_") and not c.endswith(("_lo", "_hi"))]
    groups = list(t.groupby(["instrument", "variable"]))
    fig, axes = plt.subplots(1, len(groups), figsize=(4 * len(groups), 3.4), squeeze=False)
    for ax, ((inst, var), g) in zip(axes[0], groups):
        g = g.sort_values("lead")
        for m in methods:
            ax.errorbar(g["lead"], g[f"rmse_{m}"], yerr=[g[f"rmse_{m}"] - g[f"rmse_{m}_lo"], g[f"rmse_{m}_hi"] - g[f"rmse_{m}"]],
                        marker="o", ms=3, capsize=2, label=m)
        ax.set_title(f"{LABEL.get(inst, inst)} {var}")
        ax.set_xlabel("Lead time (h)")
        ax.set_ylabel("RMSE")
        ax.set_xticks(sorted(g["lead"].unique()))
    axes[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)


def denial(a):
    d = pd.read_csv(a.summary)
    piv = d.pivot_table(index="experiment", columns="target_instrument", values="mse_change_pct")
    piv = piv[[c for c in ORDER if c in piv.columns]]
    lim = np.nanmax(np.abs(piv.to_numpy()))
    fig, ax = plt.subplots(figsize=(3.0 + 0.75 * piv.shape[1], 0.45 * len(piv) + 2.0))
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [LABEL.get(c, c) for c in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv)), [f"withhold {e.replace('_', ' ')}" for e in piv.index])
    ns = d.assign(ns=(d["ci_lo"] <= 0) & (d["ci_hi"] >= 0)).pivot_table(
        index="experiment", columns="target_instrument", values="ns", aggfunc="max").reindex_like(piv)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=7)
                if bool(ns.iat[i, j]):  # 95% CI includes zero
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", lw=0, alpha=0.4))
    fig.colorbar(im, ax=ax, label="Change in target MSE (%)")
    ax.set_xlabel("Verified target")
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)


def rollout(a):
    s = pd.read_csv(a.summary)
    s = s[s["plev"] == "all"]
    targets = [t.split(":") for t in a.targets.split(",")]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for inst, var in targets:
        g = s[(s["instrument"] == inst) & (s["variable"] == var)].sort_values("lead")
        if g.empty:
            continue
        lab = f"{LABEL.get(inst, inst)} {var}"
        line, = axes[0].plot(g["lead"], g["rmse"] / g["rmse"].iloc[0], marker="o", ms=3, label=lab)
        if "rmse_clim" in g:
            axes[0].plot(g["lead"], g["rmse_clim"] / g["rmse"].iloc[0], ls=":", color=line.get_color())
        axes[1].plot(g["lead"], g["msess_clim"], marker="o", ms=3, color=line.get_color(), label=lab)
    for ax in axes:
        ax.axvline(12, color="k", lw=0.8, ls="--")
        ax.set_xlabel("Lead time (h)")
    axes[0].set_ylabel("RMSE / RMSE(+3 h)  (dotted: climatology)")
    axes[1].set_ylabel("MSESS vs climatology")
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scorecard"); p.add_argument("--summary", required=True); p.add_argument("--metric", default="msess_clim"); p.add_argument("--out", required=True)
    p = sub.add_parser("baselines"); p.add_argument("--table", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("denial"); p.add_argument("--summary", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("rollout"); p.add_argument("--summary", required=True); p.add_argument("--out", required=True)
    p.add_argument("--targets", default="surface_obs:airTemperature,surface_obs:wind_u,radiosonde:airTemperature,amsua:bt_channel_7,atms:bt_channel_7")
    a = ap.parse_args()
    {"scorecard": scorecard, "baselines": baselines, "denial": denial, "rollout": rollout}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
