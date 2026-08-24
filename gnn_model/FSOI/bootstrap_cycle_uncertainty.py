#!/usr/bin/env python
"""Cycle-block bootstrap uncertainty for FSOI and matched OSE outputs.

This is post-processing only. It resamples consecutive 12-hour cycles from
existing CSV outputs and writes confidence intervals for the manuscript-level
tables:

  - FSOI impact rankings from csv/fsoi_by_instrument.csv
  - Innovation amplitude diagnostics used in Table 6
  - Robust innovation diagnostics used in Table 7
  - Background departure diagnostics used in Table 8
  - Closure diagnostics from evaluation/fsoi_closure_diagnostics.csv
  - Matched OSE diagnostics from evaluation/ose_vs_fsoi_comparison.csv

The bootstrap unit is the cycle/pair, not an individual observation.
"""

from __future__ import annotations

import argparse
import fnmatch
import math
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd


CI_LO = 2.5
CI_HI = 97.5
EPS = 1e-12


@dataclass(frozen=True)
class ResultRun:
    """One FSOI/OSE output directory."""

    name: str
    path: Path
    csv_dir: Path
    evaluation_dir: Path


def _stable_seed(seed: int, *parts: object) -> int:
    key = "::".join(str(p) for p in parts).encode("utf-8")
    return (int(seed) + zlib.crc32(key)) % (2**32 - 1)


def _is_result_dir(path: Path) -> bool:
    return (
        (path / "csv" / "fsoi_by_instrument.csv").is_file()
        or (path / "evaluation" / "innovation_diagnostics.csv").is_file()
        or (path / "evaluation" / "fsoi_closure_diagnostics.csv").is_file()
        or (path / "evaluation" / "ose_vs_fsoi_comparison.csv").is_file()
    )


def _make_run(path: Path) -> ResultRun:
    return ResultRun(
        name=path.name,
        path=path,
        csv_dir=path / "csv",
        evaluation_dir=path / "evaluation",
    )


def _matches_any(name: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(name, pattern) for pattern in patterns)


def _keep_run_name(name: str, include: list[str], exclude: list[str]) -> bool:
    if include and not _matches_any(name, include):
        return False
    if exclude and _matches_any(name, exclude):
        return False
    return True


def discover_runs(input_path: Path, include: list[str], exclude: list[str]) -> list[ResultRun]:
    """Return one or more result directories under input_path."""
    input_path = input_path.resolve()
    if _is_result_dir(input_path):
        if not _keep_run_name(input_path.name, include, exclude):
            raise FileNotFoundError(
                f"Input result directory {input_path.name!r} was filtered out "
                "by --include/--exclude."
            )
        return [_make_run(input_path)]

    runs = []
    if input_path.is_dir():
        for child in sorted(input_path.iterdir()):
            if child.is_dir() and _is_result_dir(child) and _keep_run_name(child.name, include, exclude):
                runs.append(_make_run(child))

    if not runs:
        raise FileNotFoundError(
            "No FSOI/OSE result directories found. Expected files like "
            "csv/fsoi_by_instrument.csv or "
            "evaluation/ose_vs_fsoi_comparison.csv under the input path."
        )
    return runs


def _parse_bin_time(value: object) -> pd.Timestamp:
    text = str(value)
    if text.startswith("bin") and len(text) >= 13 and text[3:13].isdigit():
        return pd.to_datetime(text[3:13], format="%Y%m%d%H", errors="coerce")
    return pd.to_datetime(text, errors="coerce")


def _with_cycle_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Attach stable cycle key/time columns used for block resampling."""
    df = df.copy()
    if "curr_bin" in df.columns:
        df["_cycle_key"] = df["curr_bin"].astype(str)
        df["_cycle_time"] = df["curr_bin"].map(_parse_bin_time)
    elif "pair_idx" in df.columns:
        df["_cycle_key"] = df["pair_idx"].astype(str)
        df["_cycle_time"] = pd.NaT
    else:
        raise ValueError("Cycle-level CSV must contain curr_bin or pair_idx")

    if "pair_idx" in df.columns:
        df["_pair_idx_num"] = pd.to_numeric(df["pair_idx"], errors="coerce")
    else:
        df["_pair_idx_num"] = np.arange(len(df), dtype=float)
    return df


def _cycle_order(df: pd.DataFrame) -> list[str]:
    cycles = (
        df[["_cycle_key", "_cycle_time", "_pair_idx_num"]]
        .drop_duplicates("_cycle_key")
        .sort_values(["_cycle_time", "_pair_idx_num", "_cycle_key"], na_position="last")
    )
    return [str(v) for v in cycles["_cycle_key"].tolist()]


def _moving_block_sample(cycles: list[str], block_cycles: int, rng: np.random.Generator) -> list[str]:
    n = len(cycles)
    if n == 0:
        return []
    block_cycles = max(1, min(int(block_cycles), n))
    if block_cycles == 1:
        return [cycles[i] for i in rng.integers(0, n, size=n)]

    sampled: list[str] = []
    n_blocks = int(math.ceil(n / block_cycles))
    starts = rng.integers(0, n, size=n_blocks)
    for start in starts:
        for j in range(block_cycles):
            sampled.append(cycles[(int(start) + j) % n])
            if len(sampled) == n:
                return sampled
    return sampled[:n]


def _finite_array(values: Iterable[object]) -> np.ndarray:
    arr = pd.to_numeric(pd.Series(list(values)), errors="coerce").to_numpy(dtype=float)
    return arr[np.isfinite(arr)]


def _estimate_array(values: np.ndarray, estimator: str) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan")
    if estimator == "mean":
        return float(np.mean(values))
    if estimator == "median":
        return float(np.median(values))
    if estimator == "fraction_positive":
        return float(np.mean(values > 0.0))
    if estimator == "fraction_negative":
        return float(np.mean(values < 0.0))
    raise ValueError(f"Unsupported estimator: {estimator}")


def _ci_from_boot(boot: np.ndarray) -> tuple[float, float]:
    boot = np.asarray(boot, dtype=float)
    boot = boot[np.isfinite(boot)]
    if boot.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boot, [CI_LO, CI_HI])
    return float(lo), float(hi)


def _bootstrap_value_metric(
    df: pd.DataFrame,
    cycles: list[str],
    value_col: str,
    estimator: str,
    block_cycles: int,
    n_boot: int,
    seed: int,
    multiplier: float = 1.0,
) -> dict:
    per_cycle = {
        str(cycle): (g[value_col].to_numpy(dtype=float) * multiplier)
        for cycle, g in df.groupby("_cycle_key", dropna=False)
    }
    all_values = np.concatenate([per_cycle.get(c, np.array([], dtype=float)) for c in cycles])
    point = _estimate_array(all_values, estimator)

    if n_boot <= 0 or len(cycles) <= 1:
        boot = np.array([point], dtype=float)
    else:
        rng = np.random.default_rng(seed)
        boot_values = []
        for _ in range(n_boot):
            sample_cycles = _moving_block_sample(cycles, block_cycles, rng)
            sampled = np.concatenate(
                [per_cycle.get(c, np.array([], dtype=float)) for c in sample_cycles]
            )
            boot_values.append(_estimate_array(sampled, estimator))
        boot = np.asarray(boot_values, dtype=float)

    ci_low, ci_high = _ci_from_boot(boot)
    n_values = int(np.isfinite(all_values).sum())
    n_cycles_present = int(sum(c in per_cycle and np.isfinite(per_cycle[c]).any() for c in cycles))
    return {
        "point": float(point),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "boot": boot,
        "n_values": n_values,
        "n_cycles_present": n_cycles_present,
        "n_cycles_total": int(len(cycles)),
    }


def _bootstrap_frame_metric(
    df: pd.DataFrame,
    cycles: list[str],
    metric_func: Callable[[pd.DataFrame], float],
    block_cycles: int,
    n_boot: int,
    seed: int,
) -> dict:
    per_cycle = {str(cycle): g for cycle, g in df.groupby("_cycle_key", dropna=False)}
    point = float(metric_func(df))

    if n_boot <= 0 or len(cycles) <= 1:
        boot = np.array([point], dtype=float)
    else:
        rng = np.random.default_rng(seed)
        boot_values = []
        empty = df.iloc[0:0]
        for _ in range(n_boot):
            sample_cycles = _moving_block_sample(cycles, block_cycles, rng)
            sampled_frames = [per_cycle.get(c, empty) for c in sample_cycles]
            sampled = pd.concat(sampled_frames, ignore_index=True)
            boot_values.append(float(metric_func(sampled)))
        boot = np.asarray(boot_values, dtype=float)

    ci_low, ci_high = _ci_from_boot(boot)
    return {
        "point": point,
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "boot": boot,
        "n_values": int(len(df)),
        "n_cycles_present": int(df["_cycle_key"].nunique()),
        "n_cycles_total": int(len(cycles)),
    }


def _collapse_target_variable_rows(
    df: pd.DataFrame,
    keys: tuple[str, ...] = ("pair_idx", "instrument"),
) -> pd.DataFrame:
    """Lightweight copy of fsoi_utils.collapse_target_variable_rows.

    Kept local so this uncertainty script only requires pandas/numpy and does
    not import torch via fsoi_utils.
    """
    if "target_variable" not in df.columns or df["target_variable"].nunique(dropna=True) <= 1:
        return df
    key_cols = [k for k in keys if k in df.columns]
    if not key_cols:
        return df

    strat_cols = {
        "target_variable",
        "target_channel",
        "p_idx",
        "p_hpa",
        "ea_p",
        "eb_p",
        "ea_total",
        "eb_total",
    }
    count_cols = {
        "n_observations",
        "raw_n_observations",
        "sampled_n_observations",
        "n_channels",
        "instrument_id",
        "sample_scale",
        "is_subsampled",
        "n_valid_values",
        "n_total_values",
        "total_count",
        "raw_total_count",
        "total_count_scaled",
    }
    agg: dict[str, str] = {}
    for col in df.columns:
        if col in key_cols or col in strat_cols:
            continue
        if col in count_cols or not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]):
            agg[col] = "first"
        else:
            agg[col] = "mean"
    return df.groupby(key_cols, dropna=False).agg(agg).reset_index()


def _impact_col(df: pd.DataFrame) -> str:
    return "sum_impact_scaled" if "sum_impact_scaled" in df.columns else "sum_impact"


def _read_fsoi_by_instrument(run: ResultRun) -> pd.DataFrame | None:
    path = run.csv_dir / "fsoi_by_instrument.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty or "instrument" not in df.columns:
        return None
    df = _collapse_target_variable_rows(df)
    return _with_cycle_columns(df)


def _read_innovation_diagnostics(run: ResultRun) -> pd.DataFrame | None:
    path = run.evaluation_dir / "innovation_diagnostics.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty or "instrument" not in df.columns:
        return None
    return _with_cycle_columns(df)


def _read_closure_diagnostics(run: ResultRun) -> pd.DataFrame | None:
    path = run.evaluation_dir / "fsoi_closure_diagnostics.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    return _with_cycle_columns(df)


def _read_ose_comparison(run: ResultRun) -> pd.DataFrame | None:
    path = run.evaluation_dir / "ose_vs_fsoi_comparison.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    if df.empty:
        return None
    if "denied_instrument" not in df.columns:
        if "instrument" in df.columns:
            df["denied_instrument"] = df["instrument"]
        elif "denied_instruments" in df.columns:
            df["denied_instrument"] = df["denied_instruments"]
        else:
            df["denied_instrument"] = "unknown"
    return _with_cycle_columns(df)


def _format_group(group: object) -> str:
    if isinstance(group, tuple):
        return "|".join(str(v) for v in group)
    return str(group)


def _value_metric_records(
    frames: list[tuple[ResultRun, pd.DataFrame]],
    table_name: str,
    metric_name: str,
    value_col: str,
    estimator: str,
    group_col: str,
    block_cycles: int,
    n_boot: int,
    seed: int,
    multiplier: float = 1.0,
    units: str = "",
) -> list[dict]:
    records: list[dict] = []
    aggregate_boots: dict[str, list[np.ndarray]] = {}
    aggregate_points: dict[str, list[float]] = {}
    aggregate_cycles: dict[str, int] = {}
    aggregate_values: dict[str, int] = {}

    for run, frame in frames:
        if value_col not in frame.columns or group_col not in frame.columns:
            continue
        cycles = _cycle_order(frame)
        for group, g in frame.groupby(group_col, dropna=False):
            group_name = _format_group(group)
            result = _bootstrap_value_metric(
                g,
                cycles,
                value_col=value_col,
                estimator=estimator,
                block_cycles=block_cycles,
                n_boot=n_boot,
                seed=_stable_seed(seed, table_name, metric_name, run.name, group_name),
                multiplier=multiplier,
            )
            records.append(
                {
                    "scope": "run",
                    "run_name": run.name,
                    "result_dir": str(run.path),
                    "table": table_name,
                    "metric": metric_name,
                    "group": group_name,
                    "estimator": estimator,
                    "value": result["point"],
                    "ci95_low": result["ci95_low"],
                    "ci95_high": result["ci95_high"],
                    "units": units,
                    "n_boot": n_boot,
                    "block_cycles": block_cycles,
                    "n_cycles_present": result["n_cycles_present"],
                    "n_cycles_total": result["n_cycles_total"],
                    "n_values": result["n_values"],
                }
            )
            aggregate_boots.setdefault(group_name, []).append(result["boot"])
            aggregate_points.setdefault(group_name, []).append(result["point"])
            aggregate_cycles[group_name] = aggregate_cycles.get(group_name, 0) + result["n_cycles_total"]
            aggregate_values[group_name] = aggregate_values.get(group_name, 0) + result["n_values"]

    if len(frames) > 1:
        for group_name, boots in sorted(aggregate_boots.items()):
            boot_matrix = np.vstack(boots)
            boot_mean = np.nanmean(boot_matrix, axis=0)
            point = float(np.nanmean(np.asarray(aggregate_points[group_name], dtype=float)))
            ci_low, ci_high = _ci_from_boot(boot_mean)
            records.append(
                {
                    "scope": "equal_run_mean",
                    "run_name": "ALL_RUNS",
                    "result_dir": "",
                    "table": table_name,
                    "metric": metric_name,
                    "group": group_name,
                    "estimator": estimator,
                    "value": point,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "units": units,
                    "n_boot": n_boot,
                    "block_cycles": block_cycles,
                    "n_cycles_present": aggregate_cycles[group_name],
                    "n_cycles_total": aggregate_cycles[group_name],
                    "n_values": aggregate_values[group_name],
                }
            )

    return records


def _bool_mean(series: pd.Series) -> float:
    if series.empty:
        return float("nan")
    if series.dtype == bool:
        values = series
    else:
        values = series.astype(str).str.lower().map(
            {"true": True, "1": True, "1.0": True, "yes": True, "false": False, "0": False, "0.0": False, "no": False}
        )
    values = values.dropna()
    if values.empty:
        return float("nan")
    return float(values.astype(bool).mean())


def _finite_col(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(dtype=float)
    values = pd.to_numeric(frame[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return values.dropna()


def _valid_signal_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if "signal_valid" not in frame.columns:
        return frame
    valid = frame["signal_valid"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])
    return frame[valid]


def _frame_metric_records(
    frames: list[tuple[ResultRun, pd.DataFrame]],
    table_name: str,
    metric_name: str,
    group_col: str | None,
    metric_func: Callable[[pd.DataFrame], float],
    block_cycles: int,
    n_boot: int,
    seed: int,
    units: str = "",
) -> list[dict]:
    records: list[dict] = []
    aggregate_boots: dict[str, list[np.ndarray]] = {}
    aggregate_points: dict[str, list[float]] = {}
    aggregate_cycles: dict[str, int] = {}
    aggregate_values: dict[str, int] = {}

    for run, frame in frames:
        cycles = _cycle_order(frame)
        grouped = [(None, frame)] if group_col is None else list(frame.groupby(group_col, dropna=False))
        for group, g in grouped:
            group_name = "all" if group is None else _format_group(group)
            result = _bootstrap_frame_metric(
                g,
                cycles,
                metric_func=metric_func,
                block_cycles=block_cycles,
                n_boot=n_boot,
                seed=_stable_seed(seed, table_name, metric_name, run.name, group_name),
            )
            records.append(
                {
                    "scope": "run",
                    "run_name": run.name,
                    "result_dir": str(run.path),
                    "table": table_name,
                    "metric": metric_name,
                    "group": group_name,
                    "estimator": "cycle_frame",
                    "value": result["point"],
                    "ci95_low": result["ci95_low"],
                    "ci95_high": result["ci95_high"],
                    "units": units,
                    "n_boot": n_boot,
                    "block_cycles": block_cycles,
                    "n_cycles_present": result["n_cycles_present"],
                    "n_cycles_total": result["n_cycles_total"],
                    "n_values": result["n_values"],
                }
            )
            aggregate_boots.setdefault(group_name, []).append(result["boot"])
            aggregate_points.setdefault(group_name, []).append(result["point"])
            aggregate_cycles[group_name] = aggregate_cycles.get(group_name, 0) + result["n_cycles_total"]
            aggregate_values[group_name] = aggregate_values.get(group_name, 0) + result["n_values"]

    if len(frames) > 1:
        for group_name, boots in sorted(aggregate_boots.items()):
            boot_matrix = np.vstack(boots)
            boot_mean = np.nanmean(boot_matrix, axis=0)
            point = float(np.nanmean(np.asarray(aggregate_points[group_name], dtype=float)))
            ci_low, ci_high = _ci_from_boot(boot_mean)
            records.append(
                {
                    "scope": "equal_run_mean",
                    "run_name": "ALL_RUNS",
                    "result_dir": "",
                    "table": table_name,
                    "metric": metric_name,
                    "group": group_name,
                    "estimator": "cycle_frame",
                    "value": point,
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "units": units,
                    "n_boot": n_boot,
                    "block_cycles": block_cycles,
                    "n_cycles_present": aggregate_cycles[group_name],
                    "n_cycles_total": aggregate_cycles[group_name],
                    "n_values": aggregate_values[group_name],
                }
            )

    return records


def build_fsoi_uncertainty(
    runs: list[ResultRun],
    block_cycles: int,
    n_boot: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    frames: list[tuple[ResultRun, pd.DataFrame]] = []
    for run in runs:
        frame = _read_fsoi_by_instrument(run)
        if frame is not None:
            frames.append((run, frame))
    if not frames:
        return {}

    records: list[dict] = []
    first_impact_col = _impact_col(frames[0][1])
    records.extend(
        _value_metric_records(
            frames,
            table_name="fsoi_impact",
            metric_name="mean_scaled_impact",
            value_col=first_impact_col,
            estimator="mean",
            group_col="instrument",
            block_cycles=block_cycles,
            n_boot=n_boot,
            seed=seed,
            units="forecast-error metric",
        )
    )
    if "positive_frac" in frames[0][1].columns:
        records.extend(
            _value_metric_records(
                frames,
                table_name="fsoi_impact",
                metric_name="mean_positive_fraction",
                value_col="positive_frac",
                estimator="mean",
                group_col="instrument",
                block_cycles=block_cycles,
                n_boot=n_boot,
                seed=seed,
                units="fraction",
            )
        )

    amp_records: list[dict] = []
    for col, metric in [
        ("innovation_abs_mean", "mean_absolute_innovation"),
        ("innovation_rms", "innovation_rms"),
    ]:
        if col in frames[0][1].columns:
            amp_records.extend(
                _value_metric_records(
                    frames,
                    table_name="table_6_innovation_amplitude",
                    metric_name=metric,
                    value_col=col,
                    estimator="mean",
                    group_col="instrument",
                    block_cycles=block_cycles,
                    n_boot=n_boot,
                    seed=seed,
                    units="normalized training sigma",
                )
            )

    return {
        "bootstrap_fsoi_impact.csv": pd.DataFrame(records),
        "bootstrap_table6_innovation_amplitude.csv": pd.DataFrame(amp_records),
    }


def build_innovation_uncertainty(
    runs: list[ResultRun],
    block_cycles: int,
    n_boot: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    frames: list[tuple[ResultRun, pd.DataFrame]] = []
    for run in runs:
        frame = _read_innovation_diagnostics(run)
        if frame is not None:
            frames.append((run, frame))
    if not frames:
        return {}

    robust_records: list[dict] = []
    for col, metric, units in [
        ("innovation_median", "median_bias", "normalized training sigma"),
        ("innovation_iqr_scaled", "iqr_scaled_spread", "normalized training sigma"),
        ("innovation_bowley_skewness", "bowley_skewness", "unitless"),
    ]:
        if col in frames[0][1].columns:
            robust_records.extend(
                _value_metric_records(
                    frames,
                    table_name="table_7_robust_innovation",
                    metric_name=metric,
                    value_col=col,
                    estimator="median",
                    group_col="instrument",
                    block_cycles=block_cycles,
                    n_boot=n_boot,
                    seed=seed,
                    units=units,
                )
            )

    bg_records: list[dict] = []
    if "normalized_rmse" in frames[0][1].columns:
        bg_records.extend(
            _value_metric_records(
                frames,
                table_name="table_8_background_departure",
                metric_name="median_background_departure",
                value_col="normalized_rmse",
                estimator="median",
                group_col="instrument",
                block_cycles=block_cycles,
                n_boot=n_boot,
                seed=seed,
                multiplier=100.0,
                units="percent",
            )
        )

    return {
        "bootstrap_table7_robust_innovation.csv": pd.DataFrame(robust_records),
        "bootstrap_table8_background_departure.csv": pd.DataFrame(bg_records),
    }


def build_closure_uncertainty(
    runs: list[ResultRun],
    block_cycles: int,
    n_boot: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    frames: list[tuple[ResultRun, pd.DataFrame]] = []
    for run in runs:
        frame = _read_closure_diagnostics(run)
        if frame is not None:
            frames.append((run, frame))
    if not frames:
        return {}

    def sign_agreement(frame: pd.DataFrame) -> float:
        if "sign_agree" not in frame.columns:
            return float("nan")
        return _bool_mean(frame["sign_agree"])

    def median_closure_ratio(frame: pd.DataFrame) -> float:
        values = _finite_col(frame, "closure_ratio")
        return float(values.median()) if not values.empty else float("nan")

    def median_relative_error(frame: pd.DataFrame) -> float:
        values = _finite_col(frame, "relative_abs_closure_error")
        return float(values.median()) if not values.empty else float("nan")

    records: list[dict] = []
    for metric_name, func, units in [
        ("sign_agreement_fraction", sign_agreement, "fraction"),
        ("median_closure_ratio", median_closure_ratio, "unitless"),
        ("median_relative_abs_closure_error", median_relative_error, "fraction"),
    ]:
        records.extend(
            _frame_metric_records(
                frames,
                table_name="closure",
                metric_name=metric_name,
                group_col=None,
                metric_func=func,
                block_cycles=block_cycles,
                n_boot=n_boot,
                seed=seed,
                units=units,
            )
        )

    return {"bootstrap_closure.csv": pd.DataFrame(records)}


def build_ose_uncertainty(
    runs: list[ResultRun],
    block_cycles: int,
    n_boot: int,
    seed: int,
) -> dict[str, pd.DataFrame]:
    frames: list[tuple[ResultRun, pd.DataFrame]] = []
    for run in runs:
        frame = _read_ose_comparison(run)
        if frame is not None:
            frames.append((run, frame))
    if not frames:
        return {}

    def denial_reduces_error(frame: pd.DataFrame) -> float:
        values = _finite_col(frame, "delta_j_actual")
        return float((values > 0.0).mean()) if not values.empty else float("nan")

    def direct_sign_agreement(frame: pd.DataFrame) -> float:
        valid = _valid_signal_rows(frame)
        if "sign_agree" not in valid.columns:
            return float("nan")
        return _bool_mean(valid["sign_agree"])

    def median_closure_ratio(frame: pd.DataFrame) -> float:
        valid = _valid_signal_rows(frame)
        values = _finite_col(valid, "closure_ratio")
        return float(values.median()) if not values.empty else float("nan")

    def median_abs_ratio(frame: pd.DataFrame) -> float:
        valid = _valid_signal_rows(frame)
        values = _finite_col(valid, "abs_magnitude_ratio")
        return float(values.median()) if not values.empty else float("nan")

    def near_zero_excluded_fraction(frame: pd.DataFrame) -> float:
        if "signal_valid" not in frame.columns:
            return 0.0
        valid = frame["signal_valid"].astype(str).str.lower().isin(["true", "1", "1.0", "yes"])
        return float((~valid).mean())

    records: list[dict] = []
    for metric_name, func, units in [
        ("denial_reduces_error_fraction", denial_reduces_error, "fraction"),
        ("direct_sign_agreement_fraction", direct_sign_agreement, "fraction"),
        ("median_closure_ratio", median_closure_ratio, "unitless"),
        ("median_abs_magnitude_ratio", median_abs_ratio, "unitless"),
        ("near_zero_excluded_fraction", near_zero_excluded_fraction, "fraction"),
    ]:
        records.extend(
            _frame_metric_records(
                frames,
                table_name="matched_ose",
                metric_name=metric_name,
                group_col="denied_instrument",
                metric_func=func,
                block_cycles=block_cycles,
                n_boot=n_boot,
                seed=seed,
                units=units,
            )
        )

    return {"bootstrap_ose_validation.csv": pd.DataFrame(records)}


def _autocorr(values: np.ndarray, lag: int) -> float:
    values = np.asarray(values, dtype=float)
    if values.size <= lag + 2:
        return float("nan")
    x = values[:-lag]
    y = values[lag:]
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    if x.size <= 2 or np.std(x) <= EPS or np.std(y) <= EPS:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _ordered_series(df: pd.DataFrame, value_col: str, reducer: str = "mean") -> pd.DataFrame:
    if value_col not in df.columns:
        return pd.DataFrame()
    tmp = df[["_cycle_key", "_cycle_time", "_pair_idx_num", value_col]].copy()
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    if reducer == "sum":
        grouped = tmp.groupby(["_cycle_key", "_cycle_time", "_pair_idx_num"], dropna=False)[value_col].sum()
    else:
        grouped = tmp.groupby(["_cycle_key", "_cycle_time", "_pair_idx_num"], dropna=False)[value_col].mean()
    return grouped.reset_index().sort_values(["_cycle_time", "_pair_idx_num", "_cycle_key"], na_position="last")


def build_autocorrelation(
    runs: list[ResultRun],
    max_lag_cycles: int,
    cycles_per_day: float,
    threshold: float,
) -> dict[str, pd.DataFrame]:
    records: list[dict] = []

    def add_series(run: ResultRun, source: str, group: str, metric: str, series: pd.Series) -> None:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        for lag in range(1, max_lag_cycles + 1):
            records.append(
                {
                    "run_name": run.name,
                    "result_dir": str(run.path),
                    "source": source,
                    "group": group,
                    "metric": metric,
                    "lag_cycles": lag,
                    "lag_days": lag / cycles_per_day,
                    "autocorrelation": _autocorr(values, lag),
                }
            )

    for run in runs:
        fsoi = _read_fsoi_by_instrument(run)
        if fsoi is not None:
            impact = _impact_col(fsoi)
            total = _ordered_series(fsoi, impact, reducer="sum")
            if not total.empty:
                add_series(run, "fsoi_by_instrument", "all", "total_scaled_impact", total[impact])
            for inst, g in fsoi.groupby("instrument", dropna=False):
                series = _ordered_series(g, impact, reducer="mean")
                if not series.empty:
                    add_series(run, "fsoi_by_instrument", str(inst), "mean_scaled_impact", series[impact])

        innov = _read_innovation_diagnostics(run)
        if innov is not None and "normalized_rmse" in innov.columns:
            for inst, g in innov.groupby("instrument", dropna=False):
                series = _ordered_series(g, "normalized_rmse", reducer="mean")
                if not series.empty:
                    add_series(run, "innovation_diagnostics", str(inst), "mean_normalized_rmse", series["normalized_rmse"])

        ose = _read_ose_comparison(run)
        if ose is not None:
            for inst, g in ose.groupby("denied_instrument", dropna=False):
                for col in ["delta_j_actual", "fsoi_predicted", "closure_ratio"]:
                    if col in g.columns:
                        series = _ordered_series(g, col, reducer="mean")
                        if not series.empty:
                            add_series(run, "ose_vs_fsoi_comparison", str(inst), col, series[col])

    acf = pd.DataFrame(records)
    if acf.empty:
        return {}

    summary_records = []
    for keys, g in acf.groupby(["run_name", "source", "group", "metric"], dropna=False):
        valid = g.dropna(subset=["autocorrelation"]).sort_values("lag_cycles")
        first_below = np.nan
        suggested = np.nan
        if not valid.empty:
            below = valid[valid["autocorrelation"].abs() <= threshold]
            if not below.empty:
                first_below = int(below["lag_cycles"].iloc[0])
                suggested = first_below
            else:
                suggested = int(valid["lag_cycles"].max())
        summary_records.append(
            {
                "run_name": keys[0],
                "source": keys[1],
                "group": keys[2],
                "metric": keys[3],
                "threshold_abs_acf": threshold,
                "first_lag_cycles_below_threshold": first_below,
                "suggested_block_cycles": suggested,
                "suggested_block_days": suggested / cycles_per_day if np.isfinite(suggested) else np.nan,
            }
        )

    return {
        "cycle_autocorrelation.csv": acf,
        "cycle_autocorrelation_summary.csv": pd.DataFrame(summary_records),
    }


def write_outputs(outputs: dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, df in outputs.items():
        if df is None or df.empty:
            continue
        path = output_dir / filename
        df.to_csv(path, index=False)
        print(f"Wrote {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Block-bootstrap uncertainty from existing FSOI/OSE cycle-level CSVs."
    )
    parser.add_argument("--input", required=True, type=Path, help="Result directory or parent of result directories")
    parser.add_argument("--output", type=Path, default=None, help="Output directory for uncertainty CSVs")
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Run-name glob to include, e.g. 'radiosonde_*'. May be repeated.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Run-name glob to exclude. May be repeated.",
    )
    parser.add_argument("--n_boot", type=int, default=2000, help="Number of bootstrap replicates")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--block_cycles",
        type=int,
        default=None,
        help="Consecutive 12-hour cycles per block. Overrides --block_days.",
    )
    parser.add_argument(
        "--block_days",
        type=float,
        default=3.0,
        help="Block length in days when --block_cycles is not set.",
    )
    parser.add_argument(
        "--cycles_per_day",
        type=float,
        default=2.0,
        help="Cycle frequency used to convert days to cycles.",
    )
    parser.add_argument(
        "--max_lag_cycles",
        type=int,
        default=10,
        help="Maximum lag for autocorrelation diagnostics.",
    )
    parser.add_argument(
        "--acf_threshold",
        type=float,
        default=0.2,
        help="Absolute autocorrelation threshold for suggested block length.",
    )
    args = parser.parse_args()

    if args.n_boot < 0:
        raise ValueError("--n_boot must be nonnegative")
    if args.cycles_per_day <= 0:
        raise ValueError("--cycles_per_day must be positive")

    block_cycles = args.block_cycles
    if block_cycles is None:
        block_cycles = int(round(args.block_days * args.cycles_per_day))
    block_cycles = max(1, int(block_cycles))

    runs = discover_runs(args.input, include=args.include, exclude=args.exclude)
    output_dir = args.output if args.output is not None else args.input / "uncertainty"

    print(f"Found {len(runs)} result run(s):")
    for run in runs:
        print(f"  - {run.name}: {run.path}")
    print(f"Bootstrap replicates: {args.n_boot}")
    print(f"Block length: {block_cycles} cycles ({block_cycles / args.cycles_per_day:.2f} days)")
    print(f"Output directory: {output_dir}")

    outputs: dict[str, pd.DataFrame] = {}
    for builder in [
        build_fsoi_uncertainty,
        build_innovation_uncertainty,
        build_closure_uncertainty,
        build_ose_uncertainty,
    ]:
        outputs.update(builder(runs, block_cycles, args.n_boot, args.seed))

    outputs.update(
        build_autocorrelation(
            runs,
            max_lag_cycles=args.max_lag_cycles,
            cycles_per_day=args.cycles_per_day,
            threshold=args.acf_threshold,
        )
    )

    metadata = pd.DataFrame(
        [
            {
                "input": str(args.input),
                "output": str(output_dir),
                "n_runs": len(runs),
                "n_boot": args.n_boot,
                "seed": args.seed,
                "block_cycles": block_cycles,
                "block_days": block_cycles / args.cycles_per_day,
                "cycles_per_day": args.cycles_per_day,
                "max_lag_cycles": args.max_lag_cycles,
                "acf_threshold": args.acf_threshold,
                "run_names": ",".join(run.name for run in runs),
            }
        ]
    )
    outputs["bootstrap_metadata.csv"] = metadata

    write_outputs(outputs, output_dir)
    if not outputs:
        print("No matching CSVs were found for uncertainty analysis.", file=sys.stderr)


if __name__ == "__main__":
    main()
