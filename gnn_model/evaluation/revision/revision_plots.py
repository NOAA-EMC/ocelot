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
import re
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


def _chan_num(v):
    """Trailing channel number, so bt_channel_2 sorts before bt_channel_10."""
    m = re.search(r"(\d+)$", str(v))
    return int(m.group(1)) if m else -1


def scorecard(a):
    s = pd.read_csv(a.summary)
    s = s[s["plev"] == "all"].copy()
    s["row"] = s["instrument"].map(lambda x: LABEL.get(x, x)) + " " + s["variable"].astype(str)
    s["o"] = s["instrument"].map(_ord)
    s["c"] = s["variable"].map(_chan_num)
    piv = s.pivot_table(index=["o", "c", "row"], columns="lead", values=a.metric).sort_index()
    label = {"msess_clim": "MSESS vs climatology", "acc_centered_mean": "ACC",
             "msess_pers": "MSESS vs persistence"}.get(a.metric, a.metric)
    diverging = a.metric.startswith("msess")

    # With ~90 targets a single column of rows makes a figure too tall for a journal page, so the
    # rows are dealt across `columns` panels that read top-to-bottom, left-to-right.
    ncol = max(1, int(a.columns))
    per = int(np.ceil(len(piv) / ncol))
    fig, axes = plt.subplots(1, ncol, figsize=(ncol * (2.05 + 0.26 * piv.shape[1]) + 0.7, 0.155 * per + 0.9),
                             squeeze=False, constrained_layout=True)
    im = None
    for k, ax in enumerate(axes[0]):
        block = piv.iloc[k * per:(k + 1) * per]
        if block.empty:
            ax.axis("off")
            continue
        im = ax.imshow(block.to_numpy(), aspect="auto",
                       cmap="RdBu" if diverging else "YlGnBu",
                       vmin=-1.0 if diverging else 0.0, vmax=1.0)
        ax.set_yticks(range(len(block)), [r for *_, r in block.index], fontsize=5.5)
        ax.set_xticks(range(block.shape[1]), [f"+{int(c)}" for c in block.columns], fontsize=7)
        ax.tick_params(length=0)
        for i in range(block.shape[0]):
            for j in range(block.shape[1]):
                v = block.iat[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=4.5)
    fig.colorbar(im, ax=axes[0].tolist(), label=label, fraction=0.03, pad=0.01)
    fig.suptitle(f"{label}   (lead time, hours)", fontsize=9)
    fig.savefig(a.out, dpi=300)
    print(f"{a.out}: {len(piv)} targets x {piv.shape[1]} lead times in {ncol} column(s), "
          f"{fig.get_size_inches()[0]:.1f} x {fig.get_size_inches()[1]:.1f} inches")


COLORS = {"OCELOT": "#1f77b4", "GFS": "#d62728", "GraphCastGFS": "#9467bd",
          "Climatology": "#2ca02c", "Persistence": "#ff7f0e"}


def _methods(t):
    return [c[5:] for c in t.columns if c.startswith("rmse_") and not c.endswith(("_lo", "_hi"))]


def _series(ax, g, m):
    lo, hi = g[f"rmse_{m}"] - g[f"rmse_{m}_lo"], g[f"rmse_{m}_hi"] - g[f"rmse_{m}"]
    ax.errorbar(g["lead"], g[f"rmse_{m}"], yerr=[lo, hi], marker="o", ms=3.5, lw=1.4, capsize=2,
                color=COLORS.get(m), label=m)


def baselines(a):
    t = pd.read_csv(a.table)
    # A reference that is only available at some lead times (GraphCastGFS is 6-hourly) would
    # otherwise force every method onto those leads, hiding the +3 h comparison entirely.
    ov = pd.read_csv(a.overlay) if a.overlay else None
    methods = _methods(t)
    extra = [m for m in _methods(ov)] if ov is not None else []
    extra = [m for m in extra if m not in methods]

    groups = list(t.groupby(["instrument", "variable"]))
    # Four panels in a single row are too wide for a journal page, so wrap to two rows.
    nrow = 1 if len(groups) <= 3 else 2
    ncol = int(np.ceil(len(groups) / nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=(2.7 * ncol, 2.7 * nrow + 0.5), squeeze=False)
    flat = axes.ravel()
    for ax in flat[len(groups):]:
        ax.axis("off")
    for ax, ((inst, var), g) in zip(flat, groups):
        g = g.sort_values("lead")
        for m in methods:
            _series(ax, g, m)
        if extra:
            og = ov[(ov.instrument == inst) & (ov.variable == var)].sort_values("lead")
            for m in extra:
                if not og.empty:
                    _series(ax, og, m)
        ax.set_title(f"{LABEL.get(inst, inst)} {var}", fontsize=9)
        ax.set_xlabel("Lead time (h)")
        ax.set_xticks(sorted(g["lead"].unique()))
        ax.grid(alpha=0.25, lw=0.5)
    for r in range(nrow):
        axes[r][0].set_ylabel("RMSE")
    h, l = flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(a.out, dpi=300, bbox_inches="tight")
    print(f"{a.out}: {len(groups)} panels in {nrow}x{ncol}, "
          f"{fig.get_size_inches()[0]:.1f} x {fig.get_size_inches()[1]:.1f} inches, methods {methods}"
          + (f" + {extra} on {sorted(ov.lead.unique())} h" if extra else ""))


def denial(a):
    d = pd.read_csv(a.summary)
    piv = d.pivot_table(index="experiment", columns="target_instrument", values="mse_change_pct")
    piv = piv[[c for c in ORDER if c in piv.columns]]
    # Rows in the same order as the target columns, so each group's own targets lie on a block
    # diagonal; the two combined groups go last, below a separator.
    rows = [r for r in DENIAL_ORDER if r in piv.index] + [r for r in piv.index if r not in DENIAL_ORDER]
    piv = piv.loc[rows]
    lim = np.nanmax(np.abs(piv.to_numpy()))
    fig, ax = plt.subplots(figsize=(3.0 + 0.75 * piv.shape[1], 0.45 * len(piv) + 2.0))
    # Impacts span four orders of magnitude (about 1% to over 1000%), so a linear scale would leave
    # every cross-system cell white. A symmetric-log scale keeps 1, 10, 100 and 1000% distinguishable.
    from matplotlib.colors import SymLogNorm
    norm = SymLogNorm(linthresh=1.0, linscale=0.5, vmin=-lim, vmax=lim, base=10)
    im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", norm=norm, aspect="auto")
    ax.set_xticks(range(piv.shape[1]), [LABEL.get(c, c) for c in piv.columns], rotation=45, ha="right")
    ax.set_yticks(range(len(piv)), [DENIAL_LABEL.get(e, e.replace("_", " ")) for e in piv.index])
    n_single = sum(1 for e in piv.index if not e.startswith("all_"))
    if 0 < n_single < len(piv):
        ax.axhline(n_single - 0.5, color="k", lw=1.2)
    ns = d.assign(ns=(d["ci_lo"] <= 0) & (d["ci_hi"] >= 0)).pivot_table(
        index="experiment", columns="target_instrument", values="ns", aggfunc="max").reindex_like(piv)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:+.0f}" if abs(v) >= 100 else f"{v:+.1f}", ha="center", va="center",
                        fontsize=7, color="white" if abs(v) >= 0.5 * lim else "black")
                if bool(ns.iat[i, j]):  # 95% CI includes zero
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", lw=0, alpha=0.4))
    fig.colorbar(im, ax=ax, label="Change in target MSE (%)")
    ax.set_xlabel("Verified target")
    ax.set_ylabel("Withheld inputs")
    fig.tight_layout()
    fig.savefig(a.out, dpi=300)


DENIAL_ORDER = ["mw_sounders", "mw_imager", "ir_imagers", "scatterometer", "aircraft", "radiosonde", "surface",
                "all_satellite", "all_conventional"]
DENIAL_LABEL = {"mw_sounders": "MW sounders", "mw_imager": "MW imager", "ir_imagers": "IR imagers",
                "scatterometer": "Scatterometer", "aircraft": "Aircraft", "radiosonde": "Radiosondes",
                "surface": "Surface stations", "all_satellite": "All satellite", "all_conventional": "All conventional"}


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
    p.add_argument("--columns", type=int, default=2, help="deal the target rows across this many panels")
    p = sub.add_parser("baselines"); p.add_argument("--table", required=True); p.add_argument("--out", required=True)
    p.add_argument("--overlay", default=None, help="second table whose extra methods cover only some lead times")
    p = sub.add_parser("denial"); p.add_argument("--summary", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("rollout"); p.add_argument("--summary", required=True); p.add_argument("--out", required=True)
    p.add_argument("--targets", default="surface_obs:airTemperature,surface_obs:wind_u,radiosonde:airTemperature,amsua:bt_channel_7,atms:bt_channel_7")
    a = ap.parse_args()
    {"scorecard": scorecard, "baselines": baselines, "denial": denial, "rollout": rollout}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
