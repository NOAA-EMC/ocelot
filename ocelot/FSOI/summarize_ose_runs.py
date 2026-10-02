#!/usr/bin/env python
"""Completion and closure summary for a directory of OSE runs.

Scans ``<root>/*/evaluation/ose_results.csv`` and reports, per run, whether the
run finished, which metric it used, and how well the trapezoidal FSOI estimate
closes against the finite error change.

Closure is only meaningful where ``matched_signal_valid`` is true: both sides
must exceed the repeated-control reproducibility threshold. ``drop_nodes`` runs
have no denied-endpoint input, so they carry no ``matched_fsoi`` and are
summarized on their own.

Usage:
    python FSOI/summarize_ose_runs.py --root FSOI/fsoi_outputs/ose_frozen_metric
    python FSOI/summarize_ose_runs.py --root ... --expect-cycles 62 --csv ose_summary.csv
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

STRUCTURAL = "drop_nodes"


def _config_digest(frame: pd.DataFrame) -> str:
    """Short hash of the frozen metric definition.

    ``target_metric_ids`` hashes the plan realized in a single cycle, so it
    legitimately varies cycle to cycle. ``target_metric_config`` is the frozen
    definition and must be identical everywhere J is compared.
    """
    if "target_metric_config" not in frame:
        return ""
    values = sorted(set(frame.target_metric_config.dropna().astype(str)))
    if not values:
        return ""
    digests = sorted(hashlib.sha256(v.encode()).hexdigest()[:10] for v in values)
    return ";".join(digests)


def _truthy(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def load_runs(root: Path) -> pd.DataFrame:
    frames = []
    for csv in sorted(root.glob("*/evaluation/ose_results.csv")):
        frame = pd.read_csv(csv)
        frame["run"] = csv.parent.parent.name
        frames.append(frame)
    if not frames:
        raise SystemExit(f"no */evaluation/ose_results.csv under {root}")
    return pd.concat(frames, ignore_index=True, sort=False)


def summarize(raw: pd.DataFrame, expect_cycles: int | None) -> pd.DataFrame:
    rows = []
    for (run, inst, mode), d in raw.groupby(
        ["run", "denied_instruments", "ose_denial_mode"], dropna=False
    ):
        valid = d[_truthy(d.matched_signal_valid)] if "matched_signal_valid" in d else d.iloc[:0]
        # pd.to_numeric on a missing column returns a scalar NaN, not a Series,
        # so the column has to be checked rather than fetched with .get().
        ratio = (pd.to_numeric(valid["matched_closure_ratio"], errors="coerce").dropna()
                 if "matched_closure_ratio" in valid else pd.Series(dtype=float))
        path = d[_truthy(d.path_integration_enabled)] if "path_integration_enabled" in d else d.iloc[:0]
        record = {
            "run": run,
            "instrument": inst,
            "mode": mode,
            "cycles": len(d),
            "missing_cycles": "" if expect_cycles is None else expect_cycles - len(d),
            "signal_valid": len(valid),
            "closure_median": round(float(ratio.median()), 4) if len(ratio) else np.nan,
            "closure_p10": round(float(ratio.quantile(0.10)), 4) if len(ratio) else np.nan,
            "closure_p90": round(float(ratio.quantile(0.90)), 4) if len(ratio) else np.nan,
            "sign_agree": (
                round(float(_truthy(valid.matched_sign_agree).mean()), 4)
                if len(valid) and "matched_sign_agree" in valid else np.nan
            ),
            "mean_delta_j": float(pd.to_numeric(d["delta_j_actual"], errors="coerce").mean())
            if "delta_j_actual" in d else np.nan,
            "path_cycles": len(path),
            # Path integration runs on one cycle, so its closure must be read
            # against the two-endpoint closure of that same cycle, never against
            # the median over every cycle in the month.
            "matched_closure_on_path_cycle": (
                round(float(pd.to_numeric(
                    path.matched_closure_ratio, errors="coerce").median()), 4)
                if len(path) and "matched_closure_ratio" in path else np.nan
            ),
            "path_closure_median": (
                round(float(pd.to_numeric(path.path_closure_ratio, errors="coerce").median()), 4)
                if len(path) and "path_closure_ratio" in path else np.nan
            ),
            "path_gain": (
                round(float(pd.to_numeric(
                    path.path_relative_error_reduction, errors="coerce").median()), 4)
                if len(path) and "path_relative_error_reduction" in path else np.nan
            ),
            "max_all_missing_frac": float(pd.to_numeric(
                d.get("ose_max_all_missing_row_fraction"), errors="coerce").max())
            if "ose_max_all_missing_row_fraction" in d else np.nan,
            "dropped_rows": (
                str(d.ose_dropped_rows.dropna().iloc[0]) if "ose_dropped_rows" in d
                and d.ose_dropped_rows.notna().any() else ""
            ),
            "metric_config": _config_digest(d),
        }
        rows.append(record)
    return pd.DataFrame(rows).sort_values(["mode", "instrument"]).reset_index(drop=True)


def flag(summary: pd.DataFrame, expect_cycles: int | None) -> list[str]:
    """Problems worth fixing before the numbers go anywhere near the manuscript."""
    notes = []
    configs = set(summary.metric_config) - {""}
    if len(configs) > 1:
        notes.append(
            f"the frozen metric config differs across runs ({len(configs)} distinct): "
            "J is not the same quantity and closure ratios are not comparable"
        )
    for _, r in summary.iterrows():
        tag = r["run"]
        if expect_cycles is not None and r["missing_cycles"]:
            notes.append(f"{tag}: {r['cycles']} cycles, expected {expect_cycles}")
        if r["mode"] != STRUCTURAL:
            if r["signal_valid"] == 0:
                notes.append(f"{tag}: no cycle clears the reproducibility threshold")
            elif np.isfinite(r["closure_median"]) and not 0.8 <= r["closure_median"] <= 1.2:
                notes.append(f"{tag}: closure median {r['closure_median']} is outside 0.8-1.2")
            # A per-cycle change barely above the reproducibility floor gives a
            # ratio with an almost-zero denominator, so the spread blows up even
            # though the cycle-aggregated impact is sound.
            if np.isfinite(r["closure_p10"]) and r["closure_p10"] <= 0:
                notes.append(
                    f"{tag}: closure spans zero (p10 {r['closure_p10']}, p90 {r['closure_p90']}); "
                    "per-cycle signal is marginal, quote the cycle-aggregated impact only"
                )
            if r["path_cycles"] == 0:
                notes.append(f"{tag}: no path-integration cycle was written")
        elif r["signal_valid"]:
            notes.append(f"{tag}: drop_nodes should not report matched closure")
    return notes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--expect-cycles", type=int, default=None,
                        help="cycles per run, to detect truncated jobs")
    parser.add_argument("--csv", type=Path, default=None)
    args = parser.parse_args()

    raw = load_runs(args.root)
    summary = summarize(raw, args.expect_cycles)
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(summary.to_string(index=False))
    notes = flag(summary, args.expect_cycles)
    print("\n" + ("\n".join(f"[CHECK] {n}" for n in notes) if notes else "[OK] no problems flagged"))
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.csv, index=False)
        print("wrote", args.csv)


if __name__ == "__main__":
    main()
