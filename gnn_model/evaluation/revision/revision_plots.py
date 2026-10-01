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
import matplotlib.text  # noqa: F401  (print_check walks Text artists)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ORDER = ["atms", "amsua", "ssmis", "avhrr", "seviri_asr", "seviri_csr", "ascat", "aircraft", "radiosonde", "surface_obs"]
LABEL = {"atms": "ATMS", "amsua": "AMSU-A", "ssmis": "SSMIS", "avhrr": "AVHRR", "seviri_asr": "SEVIRI ASR",
         "seviri_csr": "SEVIRI CSR", "ascat": "ASCAT", "aircraft": "Aircraft", "radiosonde": "Radiosonde",
         "surface_obs": "Surface"}


# New main-text figures are drawn at their printed size, so a font size in the file is the size on
# the page: AMS full-page width is 39 pc (6.5 in) and the maximum depth 54 pc (9 in).
PRINT_WIDTH_IN = 6.5
MAX_HEIGHT_IN = 9.0
MIN_FONT_PT = 10.5
MINUS = "−"

SHORT_VAR = {"airTemperature": "T", "dewPointTemperature": "Td", "windU": "u", "wind_u": "u",
             "windV": "v", "wind_v": "v", "pressureMeanSeaLevel_prepbufr": "MSLP"}
SURFACE_VAR = {"airTemperature": "T2m", "dewPointTemperature": "Td2m", "wind_u": "u10", "wind_v": "v10",
               "pressureMeanSeaLevel_prepbufr": "MSLP"}
CONVENTIONAL_INST = ("aircraft", "radiosonde", "surface_obs")


def _short_var(inst, var):
    """Row label inside an instrument block: the channel number, or a short variable name."""
    if inst == "surface_obs" and var in SURFACE_VAR:
        return SURFACE_VAR[var]
    if var in SHORT_VAR:
        return SHORT_VAR[var]
    n = _chan_num(var)
    return str(n) if n >= 0 else str(var)[:6]


def _group_title(inst):
    lab = LABEL.get(inst, inst)
    if inst in CONVENTIONAL_INST:
        return lab
    return f"{lab} beam" if inst == "ascat" else f"{lab} channel"


def _print_style():
    """Fixed text sizes for the print figures, independent of any site matplotlibrc or style.

    Setting only font.size is not enough: tick labels, axis labels, titles and legends use relative
    sizes ('small', 'x-small', ...) that a site configuration can shrink below it. Start from the
    matplotlib defaults and set every text size to MIN_FONT_PT explicitly.
    """
    plt.rcdefaults()
    plt.rcParams.update({k: MIN_FONT_PT for k in (
        "font.size", "axes.titlesize", "axes.labelsize", "xtick.labelsize", "ytick.labelsize",
        "legend.fontsize", "legend.title_fontsize", "figure.titlesize", "figure.labelsize")})


def print_check(fig, cell_texts=(), header_texts=()):
    """Problems that would show in print: text below MIN_FONT_PT, figure larger than the page,
    a cell value wider than its cell, or a header running past its panel."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    w, h = fig.get_size_inches()
    out = []
    small = sorted({round(t.get_fontsize(), 2) for t in fig.findobj(matplotlib.text.Text)
                    if t.get_visible() and t.get_text().strip() and t.get_fontsize() < MIN_FONT_PT - 1e-6})
    if small:
        out.append(f"text below {MIN_FONT_PT} pt: {small}")
    if w > PRINT_WIDTH_IN + 1e-6 or h > MAX_HEIGHT_IN + 1e-6:
        out.append(f"figure is {w:.2f} x {h:.2f} in, larger than {PRINT_WIDTH_IN} x {MAX_HEIGHT_IN} in")
    over = 0
    for t in cell_texts:
        ax = t.axes
        cw = ax.transData.transform((1, 0))[0] - ax.transData.transform((0, 0))[0]
        if t.get_window_extent(r).width > 0.97 * cw:
            over += 1
    if over:
        out.append(f"{over} cell values wider than their cells")
    for t in header_texts:
        if t.get_window_extent(r).x1 > t.axes.get_window_extent(r).x1 + 2:
            out.append(f"header '{t.get_text()}' runs past its panel")
    return out


def _report(out_path, fig, problems):
    w, h = fig.get_size_inches()
    if problems:
        print(f"{out_path}: PRINT CHECK FAILED ({w:.2f} x {h:.2f} in): " + "; ".join(problems))
        raise SystemExit(1)
    print(f"{out_path}: print check OK, {w:.2f} x {h:.2f} in, all text >= {MIN_FONT_PT} pt")


def _ord(inst):
    return ORDER.index(inst) if inst in ORDER else len(ORDER)


def _chan_num(v):
    """Trailing channel number, so bt_channel_2 sorts before bt_channel_10."""
    m = re.search(r"(\d+)$", str(v))
    return int(m.group(1)) if m else -1


def scorecard(a):
    """Every target x lead time, drawn at 6.5 in wide with all text >= 10.5 pt.

    Rows are grouped under bold instrument headers and labelled only by channel number or a short
    variable name, so the labels stay narrow; the blocks are dealt across `columns` panels.
    """
    _print_style()
    s = pd.read_csv(a.summary)
    s = s[s["plev"] == "all"].copy()
    label = {"msess_clim": "MSESS vs climatology", "acc_centered_mean": "ACC",
             "msess_pers": "MSESS vs persistence"}.get(a.metric, a.metric)
    diverging = a.metric.startswith("msess")
    s["o"] = s["instrument"].map(_ord)
    s["c"] = s["variable"].map(_chan_num)
    piv = s.pivot_table(index=["o", "instrument", "c", "variable"], columns="lead", values=a.metric).sort_index()
    leads = list(piv.columns)

    # optional relabelling of internal channel numbers, e.g. avhrr:50=3B,51=4,52=5
    relabel = {}
    for item in (a.channel_labels or "").split(";"):
        if ":" in item:
            inst, pairs = item.split(":", 1)
            relabel[inst.strip()] = dict(p.split("=") for p in pairs.split(",") if "=" in p)
    items = []  # ("H", title, None) for an instrument header, ("R", label, values) for a target
    for inst, g in piv.groupby(level="instrument", sort=False):
        items.append(("H", _group_title(inst), None))
        for idx, row in g.iterrows():
            lab = _short_var(inst, idx[3])
            items.append(("R", relabel.get(inst, {}).get(lab, lab), row.to_numpy(float)))

    ncol = max(1, int(a.columns))
    per = int(np.ceil(len(items) / ncol))
    # rows of the current instrument still to come (including this one), so a column can run a
    # little long instead of stranding the last one or two rows of a group in the next column
    left, n = [0] * len(items), 0
    for k in range(len(items) - 1, -1, -1):
        n = 0 if items[k][0] == "H" else n + 1
        left[k] = n
    panels, cur, inst_label = [], [], ""
    for k, (kind, text, vals) in enumerate(items):
        if kind == "H":
            inst_label = text.replace(" channel", "").replace(" beam", "")
        if cur and len(panels) < ncol - 1:
            finish_group = kind == "R" and left[k] <= 2 and len(cur) < per + 2
            if (len(cur) >= per and not finish_group) or (kind == "H" and len(cur) >= per - 1):
                panels.append(cur)
                cur = [] if kind == "H" else [("H", f"{inst_label} (cont.)", None)]
        cur.append((kind, text, vals))
    panels.append(cur)
    nrows = max(len(p) for p in panels)

    cmap = plt.get_cmap("RdBu" if diverging else "YlGnBu").copy()
    cmap.set_bad("white")
    vmin, vmax = (-1.0, 1.0) if diverging else (0.0, 1.0)
    fig, axes = plt.subplots(1, len(panels), figsize=(PRINT_WIDTH_IN, 0.205 * nrows + 1.45),
                             squeeze=False, constrained_layout=True)
    cells, headers, im = [], [], None
    for ax, panel in zip(axes[0], panels):
        M = np.full((nrows, len(leads)), np.nan)
        ticks = [""] * nrows
        for i, (kind, text, vals) in enumerate(panel):
            if kind == "R":
                M[i], ticks[i] = vals, text
        im = ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_yticks(range(nrows), ticks)
        ax.xaxis.tick_top()
        ax.set_xticks(range(len(leads)), [f"+{int(l)}" for l in leads])
        ax.tick_params(length=0, pad=2)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for i, (kind, text, vals) in enumerate(panel):
            if kind == "H":
                headers.append(ax.text(-0.45, i, text, ha="left", va="center", fontweight="bold"))
                continue
            for j, v in enumerate(vals):
                if np.isfinite(v):
                    dark = abs(v) > 0.6 if diverging else v > 0.65
                    cells.append(ax.text(j, i, f"{v:.2f}".replace("-", MINUS), ha="center", va="center",
                                         color="white" if dark else "black"))
    cb = fig.colorbar(im, ax=axes[0].tolist(), orientation="horizontal", shrink=0.55, aspect=35, pad=0.015)
    cb.set_label(f"{label}  (columns: lead time, h)")

    problems = print_check(fig, cells, headers)
    if any("wider than their cells" in p for p in problems):
        # drop the leading zero (.85, −.04) before giving up
        for t in cells:
            t.set_text(t.get_text().replace("0.", ".", 1))
        problems = print_check(fig, cells, headers)
    fig.savefig(a.out, dpi=300)
    print(f"{a.out}: {sum(k == 'R' for k, *_ in items)} targets x {len(leads)} lead times in {len(panels)} columns")
    _report(a.out, fig, problems)


COLORS = {"OCELOT": "#1f77b4", "GFS": "#d62728", "GraphCastGFS": "#9467bd",
          "Climatology": "#2ca02c", "Persistence": "#ff7f0e"}


def _methods(t):
    return [c[5:] for c in t.columns if c.startswith("rmse_") and not c.endswith(("_lo", "_hi"))]


def _series(ax, g, m):
    lo, hi = g[f"rmse_{m}"] - g[f"rmse_{m}_lo"], g[f"rmse_{m}_hi"] - g[f"rmse_{m}"]
    ax.errorbar(g["lead"], g[f"rmse_{m}"], yerr=[lo, hi], marker="o", ms=3.5, lw=1.4, capsize=2,
                color=COLORS.get(m), label=m)


PANEL_TITLE = {("surface_obs", "airTemperature"): "T2m (K)", ("surface_obs", "dewPointTemperature"): "Td2m (K)",
               ("surface_obs", "wind_u"): "u10 (m s$^{-1}$)", ("surface_obs", "wind_v"): "v10 (m s$^{-1}$)",
               ("surface_obs", "pressureMeanSeaLevel_prepbufr"): "MSLP (hPa)"}


def baselines(a):
    _print_style()
    t = pd.read_csv(a.table)
    # A reference that is only available at some lead times (GraphCastGFS is 6-hourly) would
    # otherwise force every method onto those leads, hiding the +3 h comparison entirely.
    ov = pd.read_csv(a.overlay) if a.overlay else None
    methods = _methods(t)
    extra = [m for m in _methods(ov)] if ov is not None else []
    extra = [m for m in extra if m not in methods]

    # Only the requested variables, in the requested order (default: the variables of Fig. 6).
    want = [tuple(x.split(":")) for x in a.variables.split(",")]
    byvar = dict(list(t.groupby(["instrument", "variable"])))
    groups = [(k, byvar[k]) for k in want if k in byvar]
    missing = [k for k in want if k not in byvar]
    if missing:
        raise SystemExit(f"not in {a.table}: {missing}")
    nrow = 1 if len(groups) <= 3 else 2
    ncol = int(np.ceil(len(groups) / nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=(PRINT_WIDTH_IN, 2.55 * nrow + 0.9), squeeze=False,
                             constrained_layout=True)
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
        ax.set_title(PANEL_TITLE.get((inst, var), f"{LABEL.get(inst, inst)} {var}"))
        ax.set_xlabel("Lead time (h)")
        ax.set_xticks(sorted(g["lead"].unique()))
        ax.grid(alpha=0.25, lw=0.5)
    for r in range(nrow):
        axes[r][0].set_ylabel("RMSE")
    h, l = flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="outside lower center", ncol=3, frameon=False)
    problems = print_check(fig)
    fig.canvas.draw()
    leg = fig.legends[0].get_window_extent(fig.canvas.get_renderer())
    if leg.x0 < 0 or leg.x1 > fig.bbox.x1:
        problems.append("legend wider than the figure")
    fig.savefig(a.out, dpi=300)
    print(f"{a.out}: {len(groups)} panels in {nrow}x{ncol}, methods {methods}"
          + (f" + {extra} on {sorted(ov.lead.unique())} h" if extra else ""))
    _report(a.out, fig, problems)


def denial(a):
    _print_style()
    d = pd.read_csv(a.summary)
    piv = d.pivot_table(index="experiment", columns="target_instrument", values="mse_change_pct")
    piv = piv[[c for c in ORDER if c in piv.columns]]
    # Rows in the same order as the target columns, so each group's own targets lie on a block
    # diagonal; the two combined groups go last, below a separator.
    rows = [r for r in DENIAL_ORDER if r in piv.index] + [r for r in piv.index if r not in DENIAL_ORDER]
    piv = piv.loc[rows]
    lim = np.nanmax(np.abs(piv.to_numpy()))
    fig, ax = plt.subplots(figsize=(PRINT_WIDTH_IN, 0.40 * len(piv) + 2.25), constrained_layout=True)
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
    cells = []
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                # no "+" on positive values: at 10.5 pt "+1149" does not fit a cell at print width,
                # and the colour scale and caption already give the sign; negatives keep their minus
                txt = (f"{v:.0f}" if abs(v) >= 10 else f"{v:.1f}").replace("-", MINUS)
                cells.append(ax.text(j, i, txt, ha="center", va="center",
                                     color="white" if abs(v) >= 0.5 * lim else "black"))
                if bool(ns.iat[i, j]):  # 95% CI includes zero
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="///", lw=0, alpha=0.4))
    fig.colorbar(im, ax=ax, label="Change in target MSE (%)", pad=0.015)
    ax.set_xlabel("Verified target")
    ax.set_ylabel("Withheld inputs")
    problems = print_check(fig, cells)
    fig.savefig(a.out, dpi=300)
    _report(a.out, fig, problems)


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
    p.add_argument("--columns", type=int, default=3, help="deal the target rows across this many panels")
    p.add_argument("--channel_labels", default=None,
                   help="relabel internal channel numbers, e.g. 'avhrr:50=3B,51=4,52=5' (';' between instruments)")
    p = sub.add_parser("baselines"); p.add_argument("--table", required=True); p.add_argument("--out", required=True)
    p.add_argument("--overlay", default=None, help="second table whose extra methods cover only some lead times")
    p.add_argument("--variables", default="surface_obs:airTemperature,surface_obs:wind_u,surface_obs:wind_v",
                   help="instrument:variable panels, in order (default: T2m, u10, v10, the variables of Fig. 6)")
    p = sub.add_parser("denial"); p.add_argument("--summary", required=True); p.add_argument("--out", required=True)
    p = sub.add_parser("rollout"); p.add_argument("--summary", required=True); p.add_argument("--out", required=True)
    p.add_argument("--targets", default="surface_obs:airTemperature,surface_obs:wind_u,radiosonde:airTemperature,amsua:bt_channel_7,atms:bt_channel_7")
    a = ap.parse_args()
    {"scorecard": scorecard, "baselines": baselines, "denial": denial, "rollout": rollout}[a.cmd](a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
