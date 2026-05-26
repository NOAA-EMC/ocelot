#!/usr/bin/env python
"""
Compare obs-space vs mesh-space FSOI rankings across 4 seasons.

Reads:
  Obs-space  : FSOI/fsoi_outputs/seasonal_fixed/*/evaluation/fsoi_evaluation_summary.csv
               FSOI/fsoi_outputs/full_eval_20250701_20250731/radiosonde/evaluation/fsoi_evaluation_summary.csv
  Mesh-space : FSOI/fsoi_outputs/mesh_seasonal/radiosonde_{250,500,850}hPa_*/evaluation/fsoi_evaluation_summary.csv

Produces figures:
  comparison/obs_vs_mesh_ranking_scatter.png      — instrument impact: obs vs mesh (each season/level)
  comparison/ranking_flip_table.png               — which instruments change sign?
  comparison/seasonal_stability.png               — do rankings hold across seasons?
  comparison/vertical_profile_impact.png          — instrument impact vs pressure level (mesh)
  comparison/obs_vs_mesh_side_by_side.png         — bar chart at 500 hPa, obs vs mesh
  comparison/summary_table.csv                    — full numeric comparison

Usage:
    python FSOI/compare_obs_vs_mesh_fsoi.py [--output FSOI/fsoi_outputs/comparison]
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd


# ── Color palette consistent across all plots ─────────────────────────────────

INST_COLORS = {
    "atms":       "#1f77b4",   # blue
    "amsua":      "#aec7e8",   # light blue
    "ssmis":      "#17becf",   # teal
    "avhrr":      "#9edae5",   # light teal
    "seviri_asr": "#ff7f0e",   # orange
    "ascat":      "#ffbb78",   # light orange
    "radiosonde": "#2ca02c",   # green
    "aircraft":   "#98df8a",   # light green
    "surface_obs":"#d62728",   # red
}

PRESSURE_COLORS = {850: "#e377c2", 500: "#7f7f7f", 250: "#bcbd22"}
SEASON_MARKERS  = {"jan2025": "o", "apr2025": "s", "jul2025": "^", "oct2025": "D"}
SEASON_LABELS   = {"jan2025": "Winter (Jan)", "apr2025": "Spring (Apr)",
                   "jul2025": "Summer (Jul)", "oct2025": "Fall (Oct)"}


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_summary(path: Path, **meta) -> pd.DataFrame:
    df = pd.read_csv(path)
    for k, v in meta.items():
        df[k] = v
    return df


def load_obs_space(base: Path) -> pd.DataFrame:
    """Load all obs-space FSOI evaluation summaries."""
    frames = []

    # Fixed seasonal re-runs
    for p in sorted(base.glob("seasonal_fixed/radiosonde_*/evaluation/fsoi_evaluation_summary.csv")):
        season_tag = p.parts[-3]   # e.g. radiosonde_jan2025
        season = season_tag.replace("radiosonde_", "")
        frames.append(_load_summary(p, verification="obs", season=season, pressure_hpa=None))

    # Jul 2025 full-eval (completed separately)
    jul = base / "full_eval_20250701_20250731/radiosonde/evaluation/fsoi_evaluation_summary.csv"
    if jul.is_file():
        frames.append(_load_summary(jul, verification="obs", season="jul2025", pressure_hpa=None))

    if not frames:
        print(f"[WARN] No obs-space evaluation summaries found under {base}")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    print(f"Obs-space  : {len(df)} rows from {len(frames)} runs "
          f"({sorted(df['season'].unique())})")
    return df


def load_mesh_space(base: Path) -> pd.DataFrame:
    """Load all mesh-space FSOI evaluation summaries."""
    frames = []

    pattern = "mesh_seasonal/radiosonde_*hPa_*/evaluation/fsoi_evaluation_summary.csv"
    for p in sorted(base.glob(pattern)):
        parts = p.parts
        dir_name = parts[-3]   # e.g. radiosonde_500hPa_jan2025
        tokens = dir_name.split("_")
        # tokens: ['radiosonde', '500hPa', 'jan2025'] (possibly)
        try:
            pressure_hpa = int(tokens[1].replace("hPa", ""))
            season = "_".join(tokens[2:])
        except (IndexError, ValueError):
            pressure_hpa, season = None, dir_name
        frames.append(_load_summary(p, verification="mesh",
                                     season=season, pressure_hpa=pressure_hpa))

    if not frames:
        print(f"[WARN] No mesh-space evaluation summaries found under {base}")
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    levels = sorted(df["pressure_hpa"].dropna().unique())
    print(f"Mesh-space : {len(df)} rows from {len(frames)} runs "
          f"(levels={levels} hPa, seasons={sorted(df['season'].unique())})")
    return df


def seasonal_mean(df: pd.DataFrame) -> pd.DataFrame:
    """Average impact_mean_per_pair across seasons per instrument (and level for mesh)."""
    group_cols = ["instrument", "verification"]
    if "pressure_hpa" in df.columns and df["pressure_hpa"].notna().any():
        group_cols.append("pressure_hpa")
    return (df.groupby(group_cols)["impact_mean_per_pair"]
              .agg(mean="mean", std="std", n="count")
              .reset_index())


# ── Plot 1: Scatter obs vs mesh at 500 hPa ────────────────────────────────────

def plot_obs_vs_mesh_scatter(obs: pd.DataFrame, mesh: pd.DataFrame,
                              out_dir: Path, pressure_hpa: int = 500) -> None:
    """Scatter: obs-space mean impact (x) vs mesh-space mean impact (y) per instrument.

    One point per (instrument, season). The diagonal y=0 and x=0 lines
    divide the plane into quadrants:
      Q3 (x<0, y<0): beneficial in BOTH → confident beneficial
      Q1 (x>0, y>0): detrimental in BOTH → confident detrimental
      Q2 (x<0, y>0): beneficial obs-space, detrimental mesh → obs bias suspected
      Q4 (x>0, y<0): detrimental obs-space, beneficial mesh → background bias suspected
    """
    mesh_p = mesh[mesh["pressure_hpa"] == pressure_hpa]
    if mesh_p.empty:
        print(f"  Skipping scatter: no mesh data at {pressure_hpa} hPa")
        return

    # Merge on (instrument, season)
    merged = obs[["instrument", "season", "impact_mean_per_pair"]].merge(
        mesh_p[["instrument", "season", "impact_mean_per_pair"]],
        on=["instrument", "season"], suffixes=("_obs", "_mesh"),
    ).dropna(subset=["impact_mean_per_pair_obs", "impact_mean_per_pair_mesh"])

    if merged.empty:
        print("  Skipping scatter: no matched (instrument, season) pairs")
        return

    instruments = sorted(merged["instrument"].unique())
    seasons = sorted(merged["season"].unique())

    fig, ax = plt.subplots(figsize=(9, 9))

    ax.axhline(0, color="gray", lw=0.8, linestyle="--")
    ax.axvline(0, color="gray", lw=0.8, linestyle="--")

    # Quadrant labels
    xlim = merged[["impact_mean_per_pair_obs", "impact_mean_per_pair_mesh"]].abs().max().max() * 1.15
    xlim = max(xlim, 0.1)
    for text, xy in [("Beneficial\nboth", (-xlim*0.6, -xlim*0.75)),
                     ("Detrimental\nboth", (xlim*0.4, xlim*0.65)),
                     ("Obs bias?", (-xlim*0.65, xlim*0.65)),
                     ("Background\nbias?", (xlim*0.35, -xlim*0.75))]:
        ax.text(*xy, text, fontsize=8, color="gray", ha="center", va="center",
                style="italic", alpha=0.6)

    for inst in instruments:
        sub = merged[merged["instrument"] == inst]
        color = INST_COLORS.get(inst, "black")
        for _, row in sub.iterrows():
            mk = SEASON_MARKERS.get(row["season"], "o")
            ax.scatter(row["impact_mean_per_pair_obs"], row["impact_mean_per_pair_mesh"],
                       marker=mk, color=color, s=90, edgecolors="k", lw=0.5, zorder=5)
        # Label instrument at its mean position
        ax.annotate(inst,
                    xy=(sub["impact_mean_per_pair_obs"].mean(),
                        sub["impact_mean_per_pair_mesh"].mean()),
                    xytext=(5, 5), textcoords="offset points",
                    fontsize=8, color=color)

    # Legend: instruments
    inst_patches = [mpatches.Patch(color=INST_COLORS.get(i, "black"), label=i)
                    for i in instruments]
    # Legend: seasons
    season_handles = [plt.scatter([], [], marker=SEASON_MARKERS.get(s, "o"),
                                  color="gray", s=70, label=SEASON_LABELS.get(s, s))
                      for s in seasons]
    l1 = ax.legend(handles=inst_patches, loc="upper left", fontsize=7,
                   title="Instrument", title_fontsize=8, ncol=2)
    ax.legend(handles=season_handles, loc="lower right", fontsize=7,
              title="Season", title_fontsize=8)
    ax.add_artist(l1)

    ax.set_xlim(-xlim, xlim)
    ax.set_ylim(-xlim, xlim)
    ax.set_xlabel("Obs-space mean impact  (radiosonde targets, area-weighted)")
    ax.set_ylabel(f"Mesh-space mean impact  (GFS reference, {pressure_hpa} hPa)")
    ax.set_title(f"Obs-space vs Mesh-space FSOI rankings\n"
                 f"4 seasons · 9 instruments · mesh {pressure_hpa} hPa\n"
                 f"Background-bias candidates: Q4 (obs detrimental, mesh beneficial)")
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    out = out_dir / f"obs_vs_mesh_scatter_{pressure_hpa}hPa.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 2: Side-by-side bar chart at 500 hPa ────────────────────────────────

def plot_side_by_side_bars(obs: pd.DataFrame, mesh: pd.DataFrame,
                            out_dir: Path, pressure_hpa: int = 500) -> None:
    """Bar chart: obs-space vs mesh-space seasonal-mean impact per instrument."""
    obs_mean = (obs.groupby("instrument")["impact_mean_per_pair"]
                   .mean().sort_values())
    mesh_p = mesh[mesh["pressure_hpa"] == pressure_hpa]
    mesh_mean = mesh_p.groupby("instrument")["impact_mean_per_pair"].mean()
    mesh_std  = mesh_p.groupby("instrument")["impact_mean_per_pair"].std()
    obs_std   = obs.groupby("instrument")["impact_mean_per_pair"].std()

    instruments = obs_mean.index.tolist()
    x = np.arange(len(instruments))
    w = 0.35

    fig, ax = plt.subplots(figsize=(12, 6))

    bars_obs = ax.bar(x - w/2,
                      [obs_mean.get(i, np.nan) for i in instruments],
                      w, label="Obs-space (radiosonde targets)",
                      yerr=[obs_std.get(i, 0) for i in instruments],
                      color=[INST_COLORS.get(i, "gray") for i in instruments],
                      alpha=0.9, capsize=3, error_kw={"lw": 1})
    bars_mesh = ax.bar(x + w/2,
                       [mesh_mean.get(i, np.nan) for i in instruments],
                       w, label=f"Mesh-space ({pressure_hpa} hPa vs GFS)",
                       yerr=[mesh_std.get(i, 0) for i in instruments],
                       color=[INST_COLORS.get(i, "gray") for i in instruments],
                       alpha=0.45, capsize=3, error_kw={"lw": 1},
                       edgecolor="black", lw=0.8, linestyle="--")

    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(instruments, rotation=25, ha="right")
    ax.set_ylabel("Mean impact per pair  (negative = beneficial)")
    ax.set_title(f"Observation Impact: Obs-space vs Mesh-space ({pressure_hpa} hPa)\n"
                 f"Solid = obs-space (radiosonde target, area-weighted) | "
                 f"Striped = mesh-space (GFS reference global)\n"
                 f"Error bars = ±1σ across 4 seasons")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    out = out_dir / f"obs_vs_mesh_side_by_side_{pressure_hpa}hPa.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 3: Sign-flip table ───────────────────────────────────────────────────

def plot_sign_flip_table(obs: pd.DataFrame, mesh: pd.DataFrame,
                          out_dir: Path) -> None:
    """Table showing which instruments flip sign between obs-space and mesh-space."""
    obs_mean  = obs.groupby("instrument")["impact_mean_per_pair"].mean()
    levels = sorted(mesh["pressure_hpa"].dropna().unique())

    records = []
    for inst in sorted(obs_mean.index):
        row = {"instrument": inst,
               "obs_space": float(obs_mean.get(inst, np.nan)),
               "obs_sign": "(-) helpful" if obs_mean.get(inst, 0) < 0 else "(+) detrimental"}
        for lev in levels:
            mesh_l = mesh[mesh["pressure_hpa"] == lev]
            m = mesh_l.groupby("instrument")["impact_mean_per_pair"].mean()
            val = float(m.get(inst, np.nan))
            row[f"mesh_{int(lev)}hPa"] = val
            row[f"sign_{int(lev)}hPa"] = "(-) helpful" if val < 0 else "(+) detrimental"
        records.append(row)

    df_table = pd.DataFrame(records)
    df_table.to_csv(out_dir / "sign_flip_table.csv", index=False)

    # Identify flippers
    val_cols = [c for c in df_table.columns if c.startswith("mesh_")]
    flippers = df_table[df_table.apply(
        lambda r: any((r["obs_space"] < 0) != (r[c] < 0)
                      for c in val_cols if pd.notna(r[c])),
        axis=1
    )]["instrument"].tolist()

    # Matplotlib table figure
    display_cols = ["instrument", "obs_space"] + val_cols
    col_labels = ["Instrument", "Obs-space"] + [c.replace("mesh_", "Mesh\n") for c in val_cols]
    cell_data = []
    cell_colors = []
    for _, row in df_table.iterrows():
        cells = [row["instrument"], f"{row['obs_space']:+.3e}"]
        colors = ["white", "#ffd966" if row["instrument"] in flippers else "white"]
        for c in val_cols:
            v = row[c]
            cells.append(f"{v:+.3e}" if pd.notna(v) else "N/A")
            if pd.notna(v):
                obs_v = row["obs_space"]
                flip = pd.notna(obs_v) and (obs_v < 0) != (v < 0)
                colors.append("#ff9999" if flip else ("#d6e8d6" if v < 0 else "#ffe0e0"))
            else:
                colors.append("lightgray")
        cell_data.append(cells)
        cell_colors.append(colors)

    fig, ax = plt.subplots(figsize=(max(8, 3 * len(val_cols)), 0.6 * len(records) + 2))
    ax.axis("off")
    tbl = ax.table(cellText=cell_data, colLabels=col_labels,
                   cellLoc="center", loc="center", cellColours=cell_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.8)
    ax.set_title("Obs-space vs Mesh-space Sign Comparison\n"
                 "Yellow row = instrument flips sign (background bias suspected)\n"
                 "Red cell = sign disagreement at that level", pad=20, fontsize=10)
    plt.tight_layout()
    out = out_dir / "sign_flip_table.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    if flippers:
        print(f"  Sign-flip instruments: {flippers}")
    else:
        print("  No sign flips detected between obs-space and mesh-space")


# ── Plot 4: Vertical profile of instrument impact ─────────────────────────────

def plot_vertical_profile(mesh: pd.DataFrame, out_dir: Path) -> None:
    """Instrument impact vs pressure level — shows where each obs type matters most."""
    levels = sorted(mesh["pressure_hpa"].dropna().unique())
    if len(levels) < 2:
        print("  Skipping vertical profile: need >= 2 pressure levels")
        return

    instruments = sorted(mesh["instrument"].unique())
    fig, ax = plt.subplots(figsize=(8, 7))

    for inst in instruments:
        sub = mesh[mesh["instrument"] == inst]
        means = sub.groupby("pressure_hpa")["impact_mean_per_pair"].mean()
        stds  = sub.groupby("pressure_hpa")["impact_mean_per_pair"].std()
        lvls  = sorted(means.index)
        vals  = [float(means.get(l, np.nan)) for l in lvls]
        errs  = [float(stds.get(l, 0)) for l in lvls]

        color = INST_COLORS.get(inst, "gray")
        ax.plot(vals, lvls, "o-", color=color, lw=2, ms=7, label=inst)
        ax.fill_betweenx(lvls,
                         [v - e for v, e in zip(vals, errs)],
                         [v + e for v, e in zip(vals, errs)],
                         color=color, alpha=0.12)

    ax.axvline(0, color="black", lw=1, linestyle="--")
    ax.set_yscale("log")
    ax.invert_yaxis()
    ax.set_yticks(levels)
    ax.set_yticklabels([f"{int(l)} hPa" for l in levels])
    ax.set_xlabel("Mean FSOI impact per pair  (negative = beneficial)")
    ax.set_ylabel("Pressure level (hPa)")
    ax.set_title("Vertical Profile of Observation Impact\n"
                 "(mesh-space FSOI vs GFS analysis, 4-season mean ± 1σ)")
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    out = out_dir / "vertical_profile_impact.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Plot 5: Seasonal stability ────────────────────────────────────────────────

def plot_seasonal_stability(obs: pd.DataFrame, mesh: pd.DataFrame,
                             out_dir: Path, pressure_hpa: int = 500) -> None:
    """Does instrument ranking stay consistent across seasons?"""
    seasons = sorted(set(obs["season"].unique()) | set(mesh[mesh["pressure_hpa"] == pressure_hpa]["season"].unique()))
    instruments = sorted(obs["instrument"].unique())
    n_s = len(seasons)
    n_i = len(instruments)

    fig, axes = plt.subplots(1, 2, figsize=(16, max(5, n_i * 0.5)),
                              sharey=True)

    for ax, (df, label, extra_filter) in zip(axes, [
        (obs, "Obs-space (radiosonde targets)", {}),
        (mesh, f"Mesh-space ({pressure_hpa} hPa vs GFS)", {"pressure_hpa": pressure_hpa}),
    ]):
        dff = df.copy()
        for k, v in extra_filter.items():
            dff = dff[dff[k] == v]

        x = np.arange(n_i)
        w = 0.7 / max(n_s, 1)
        offsets = np.linspace(-0.3, 0.3, n_s)

        for j, season in enumerate(seasons):
            sub = dff[dff["season"] == season].set_index("instrument")
            vals = [float(sub["impact_mean_per_pair"].get(i, np.nan)) for i in instruments]
            mk = SEASON_MARKERS.get(season, "o")
            ax.barh(x + offsets[j], vals, w,
                    label=SEASON_LABELS.get(season, season),
                    color=[INST_COLORS.get(i, "gray") for i in instruments],
                    alpha=0.55)

        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(x)
        ax.set_yticklabels(instruments)
        ax.set_xlabel("Mean impact per pair")
        ax.set_title(label)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, alpha=0.15, axis="x")

    fig.suptitle("Seasonal Stability of FSOI Rankings  (negative = beneficial)",
                 fontsize=11, y=1.01)
    plt.tight_layout()
    out = out_dir / f"seasonal_stability_{pressure_hpa}hPa.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")


# ── Summary table ─────────────────────────────────────────────────────────────

def write_summary_table(obs: pd.DataFrame, mesh: pd.DataFrame,
                         out_dir: Path) -> None:
    """CSV with obs-space and mesh-space mean ± std per instrument per level."""
    obs_g = obs.groupby("instrument")["impact_mean_per_pair"].agg(
        obs_mean="mean", obs_std="std", obs_n_seasons="count")

    levels = sorted(mesh["pressure_hpa"].dropna().unique())
    parts = [obs_g]
    for lev in levels:
        m = (mesh[mesh["pressure_hpa"] == lev]
             .groupby("instrument")["impact_mean_per_pair"]
             .agg(**{f"mesh_{int(lev)}hPa_mean": "mean",
                     f"mesh_{int(lev)}hPa_std": "std",
                     f"mesh_{int(lev)}hPa_n": "count"}))
        parts.append(m)

    summary = pd.concat(parts, axis=1).reset_index()
    summary = summary.sort_values("obs_mean")

    # Add sign-flip flag
    for lev in levels:
        col = f"mesh_{int(lev)}hPa_mean"
        if col in summary.columns:
            summary[f"sign_flip_{int(lev)}hPa"] = (
                (summary["obs_mean"] < 0) != (summary[col] < 0)
            )

    out = out_dir / "obs_vs_mesh_summary.csv"
    summary.to_csv(out, index=False)
    print(f"  Saved: {out}")
    print("\nSummary table:")
    print(summary.to_string(index=False))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare obs-space vs mesh-space FSOI rankings")
    parser.add_argument("--base", type=Path,
                        default=Path("FSOI/fsoi_outputs"),
                        help="Base directory containing all FSOI output dirs")
    parser.add_argument("--output", type=Path,
                        default=None,
                        help="Output directory for comparison plots")
    parser.add_argument("--primary_level", type=int, default=500,
                        help="Primary pressure level for side-by-side plots (default: 500)")
    args = parser.parse_args()

    out_dir = args.output or (args.base / "comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base dir  : {args.base}")
    print(f"Output    : {out_dir}")
    print()

    # Load data
    obs  = load_obs_space(args.base)
    mesh = load_mesh_space(args.base)

    if obs.empty and mesh.empty:
        print("ERROR: No data found. Check --base path.")
        sys.exit(1)

    levels = sorted(mesh["pressure_hpa"].dropna().unique()) if not mesh.empty else []
    print(f"\nAvailable pressure levels: {levels} hPa")
    print(f"Obs-space seasons        : {sorted(obs['season'].unique()) if not obs.empty else []}")
    print()

    # Generate all figures
    print("Generating comparison figures...")

    if not obs.empty and not mesh.empty:
        print("[1/6] Scatter: obs vs mesh")
        for lev in (levels if levels else [args.primary_level]):
            plot_obs_vs_mesh_scatter(obs, mesh, out_dir, pressure_hpa=int(lev))

        print("[2/6] Side-by-side bars")
        for lev in (levels if levels else [args.primary_level]):
            plot_side_by_side_bars(obs, mesh, out_dir, pressure_hpa=int(lev))

        print("[3/6] Sign-flip table")
        plot_sign_flip_table(obs, mesh, out_dir)

        print("[4/6] Seasonal stability")
        plot_seasonal_stability(obs, mesh, out_dir, pressure_hpa=args.primary_level)

        print("[5/6] Summary table")
        write_summary_table(obs, mesh, out_dir)

    if not mesh.empty and len(levels) >= 2:
        print("[6/6] Vertical profile")
        plot_vertical_profile(mesh, out_dir)
    else:
        print("[6/6] Vertical profile — skipped (need >=2 pressure levels)")

    print(f"\nDone. All outputs in: {out_dir}")
    print("\nKey files:")
    print(f"  obs_vs_mesh_summary.csv            — full numeric table")
    print(f"  sign_flip_table.csv                — which instruments change sign?")
    print(f"  obs_vs_mesh_scatter_500hPa.png     — the main comparison scatter")
    print(f"  obs_vs_mesh_side_by_side_500hPa.png")
    print(f"  vertical_profile_impact.png        — where each instrument matters")
    print(f"  seasonal_stability_500hPa.png")


if __name__ == "__main__":
    main()
