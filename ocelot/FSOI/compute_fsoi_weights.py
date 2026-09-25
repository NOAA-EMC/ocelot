#!/usr/bin/env python
"""Candidate observation loss weights from the frozen balanced FSOI objective.

Reads the combined (group-weighted) outputs of runs named <target>_<month>:
csv/fsoi_combined_by_instrument.csv and csv/fsoi_combined_by_channel.csv. Negative
FSOI is beneficial. Within each cycle, impact is expressed relative to the control
error J(xa) of that cycle's frozen objective:

  total    HT-weighted instrument total / J    (network contribution)
  per_obs  Hajek mean per valid value / J      (contribution per observed value)

Cycles have equal weight within a month, months within a target, and verification
targets (radiosonde, aircraft, surface) in the pooled result. Cycle-block bootstrap
intervals, resampled within each target-month, classify each instrument or channel
as beneficial, detrimental, or inconclusive. Inconclusive entries keep weight 1;
detrimental entries receive the floor and should be investigated, not just
down-weighted.

These are candidates only. FSOI measures an observation type's value as a model
INPUT, whereas instrument_weights scale its TARGET loss during training. Validate a
candidate by fine-tuning and scoring with the same frozen verification metric.

Usage:
    python FSOI/compute_fsoi_weights.py \
        --fsoi_dirs "$OUT_ROOT/*_*2025" --expected-runs 12 \
        --obs_config configs/observation_config.yaml \
        --output_dir "$OUT_ROOT/fsoi_weights"
"""

import argparse
import copy
import glob
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

INSTRUMENT_CSV = 'fsoi_combined_by_instrument.csv'
CHANNEL_CSV = 'fsoi_combined_by_channel.csv'
REQUIRED = {'curr_bin', 'lead_step', 'instrument', 'sum_impact_ht', 'ea', 'population_scaling_method'}
CYCLE = ['target', 'month', 'cycle']


def find_runs(patterns):
    runs = set()
    for pattern in patterns:
        for match in glob.glob(str(pattern)):
            path = Path(match)
            if path.name == 'csv':
                path = path.parent
            if (path / 'csv' / INSTRUMENT_CSV).is_file():
                runs.add(path.resolve())
    return sorted(runs)


def load_runs(runs, filename):
    frames = []
    for run in runs:
        target, _, month = run.name.rpartition('_')
        if not target:
            raise ValueError(f"{run}: run directories must be named <target>_<month>")
        path = run / 'csv' / filename
        df = pd.read_csv(path)
        missing = REQUIRED - set(df.columns)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}; use runs made with the frozen metric")
        if not df['population_scaling_method'].eq('horvitz_thompson_row_inclusion').all():
            raise ValueError(f"{path}: HT population totals are required")
        if not (np.isfinite(df['ea']) & (df['ea'] > 0)).all():
            raise ValueError(f"{path}: control error J must be finite and positive")
        df['target'], df['month'] = target, month
        df['cycle'] = df['curr_bin'].astype(str) + '_step' + df['lead_step'].astype(str)
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No {filename} files found")
    frame = pd.concat(frames, ignore_index=True)
    if 'target_metric_config' in frame:
        mixed = frame.groupby('target')['target_metric_config'].nunique()
        if (mixed > 1).any():
            raise ValueError(f"Months of one target used different frozen metrics: {mixed[mixed > 1].to_dict()}")
    return frame


def relative_wide(frame, keys, numerator, absent_is_zero):
    """Cycle x key table of impact / J; an absent instrument contributes zero total impact."""
    values = frame.assign(value=frame[numerator] / frame['ea'])
    wide = values.pivot_table(index=CYCLE, columns=keys, values='value', aggfunc='sum')
    return wide.fillna(0.0) if absent_is_zero else wide


def pooled(wide):
    """Equal weight per cycle within a month, per month within a target, per target."""
    by_target = wide.groupby(level=['target', 'month']).mean().groupby(level='target').mean()
    return by_target.mean(), by_target


def bootstrap(wide, n_boot, seed):
    rng = np.random.default_rng(seed)
    blocks = [(t, g.to_numpy()) for (t, _), g in wide.groupby(level=['target', 'month'])]
    targets = sorted({t for t, _ in blocks})
    draws = np.empty((n_boot, wide.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)  # all-NaN per-obs columns
        for b in range(n_boot):
            months = {t: [] for t in targets}
            for t, values in blocks:
                sample = values[rng.integers(len(values), size=len(values))]
                months[t].append(np.nanmean(sample, axis=0))
            draws[b] = np.nanmean([np.nanmean(m, axis=0) for m in months.values()], axis=0)
    return pd.DataFrame(draws, columns=wide.columns)


def classify(wide, n_boot, seed, alpha=0.10):
    estimate, by_target = pooled(wide)
    draws = bootstrap(wide, n_boot, seed)
    out = pd.DataFrame({'impact': estimate,
                        'ci_low': draws.quantile(alpha / 2),
                        'ci_high': draws.quantile(1 - alpha / 2)})
    out['status'] = np.select([out['ci_high'] < 0, out['ci_low'] > 0],
                              ['beneficial', 'detrimental'], 'inconclusive')
    return out, by_target


def candidate_weights(summary, min_weight=0.25, max_weight=4.0, normalize=True):
    """Beneficial entries scale with benefit relative to the mean benefit of all entries;
    inconclusive keep 1; detrimental get the floor. Normalized to mean 1 unless requested."""
    benefit = (-summary['impact']).clip(lower=0)
    beneficial = summary['status'].eq('beneficial')
    raw = pd.Series(1.0, index=summary.index)
    if beneficial.any() and benefit.mean() > 0:
        raw[beneficial] = benefit[beneficial] / benefit.mean()
    raw[summary['status'].eq('detrimental')] = min_weight
    raw = raw.clip(min_weight, max_weight)
    return raw / raw.mean() if normalize else raw


def instrument_features(obs_config):
    return {inst: len(cfg.get('features', []))
            for group in obs_config.get('observation_config', {}).values()
            for inst, cfg in group.items()}


def patch_observation_config(base_config, instrument_weights, channel_weights=None):
    cfg = copy.deepcopy(base_config)
    cfg.setdefault('instrument_weights', {})
    for inst, w in instrument_weights.items():
        cfg['instrument_weights'][inst] = round(float(w), 6)
    for inst, weights in (channel_weights or {}).items():
        cfg.setdefault('channel_weights', {})[inst] = [round(float(v), 6) for v in weights]
    return cfg


def channel_weight_lists(channel_summary, n_features, min_weight, max_weight):
    """Per-instrument lists (1-based channel -> position); unseen channels keep weight 1."""
    lists = {}
    for inst, group in channel_summary.groupby(level='instrument'):
        n = n_features.get(inst, 0)
        if n == 0:
            continue
        group = group.droplevel('instrument')
        values = np.ones(n)
        weights = candidate_weights(group, min_weight, max_weight, normalize=False)
        for channel, w in weights.items():
            if 1 <= int(channel) <= n:
                values[int(channel) - 1] = w
        lists[inst] = (values / values.mean()).tolist()
    return lists


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--fsoi_dirs', nargs='+', required=True,
                        help='Run directories (<target>_<month>) or their csv/ folders; glob patterns allowed')
    parser.add_argument('--expected-runs', type=int, help='Require exactly this many runs')
    parser.add_argument('--obs_config', default='configs/observation_config.yaml')
    parser.add_argument('--output_dir', default='FSOI/fsoi_weights')
    parser.add_argument('--n-boot', type=int, default=2000)
    parser.add_argument('--seed', type=int, default=20250101)
    parser.add_argument('--min-weight', type=float, default=0.25)
    parser.add_argument('--max-weight', type=float, default=4.0)
    args = parser.parse_args()

    runs = find_runs(args.fsoi_dirs)
    if args.expected_runs is not None and len(runs) != args.expected_runs:
        raise ValueError(f"Expected {args.expected_runs} runs, found {len(runs)}: {[r.name for r in runs]}")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inst = load_runs(runs, INSTRUMENT_CSV)
    print(inst.groupby(['target', 'month'])['cycle'].nunique().rename('cycles').to_string())
    total, total_by_target = classify(relative_wide(inst, 'instrument', 'sum_impact_ht', True),
                                      args.n_boot, args.seed)
    per_obs, per_obs_by_target = classify(relative_wide(inst, 'instrument', 'mean_impact_hajek', False),
                                          args.n_boot, args.seed)
    summary = total.add_suffix('_total').join(per_obs.add_suffix('_per_obs'), how='outer')
    summary['weight_total'] = candidate_weights(total, args.min_weight, args.max_weight)
    summary['weight_per_obs'] = candidate_weights(per_obs, args.min_weight, args.max_weight)
    summary.index.name = 'instrument'
    summary.to_csv(out_dir / 'instrument_impact_summary.csv')
    pd.concat({'total': total_by_target, 'per_obs': per_obs_by_target}, names=['measure']).to_csv(
        out_dir / 'instrument_impact_by_target.csv')
    print("\nRelative impact (negative = reduces J) and candidate weights:")
    print(summary.sort_values('impact_total').to_string(float_format=lambda v: f"{v:.3g}"))

    channel_lists = {}
    channel_runs = [r for r in runs if (r / 'csv' / CHANNEL_CSV).is_file()]
    if len(channel_runs) == len(runs):
        channels = load_runs(runs, CHANNEL_CSV)
        channel_summary, _ = classify(relative_wide(channels, ['instrument', 'channel'], 'sum_impact_ht', True),
                                      args.n_boot, args.seed)
        channel_summary.to_csv(out_dir / 'channel_impact_summary.csv')
    else:
        print(f"NOTE: {CHANNEL_CSV} missing in some runs; channel weights not computed")
        channel_summary = None

    obs_config_path = Path(args.obs_config)
    if not obs_config_path.is_file():
        print(f"{obs_config_path} not found; wrote summaries only")
        return
    base_config = yaml.safe_load(obs_config_path.read_text(encoding='utf-8'))
    unknown = sorted(set(summary.index) - set(base_config.get('instrument_weights', {})))
    if unknown:
        print(f"WARNING: instruments absent from the base instrument_weights: {unknown}")
    if channel_summary is not None:
        channel_lists = channel_weight_lists(channel_summary, instrument_features(base_config),
                                             args.min_weight, args.max_weight)
    outputs = {
        'observation_config_candidate_total.yaml': (summary['weight_total'], None),
        'observation_config_candidate_per_obs.yaml': (summary['weight_per_obs'], None),
        'observation_config_candidate_total_channel.yaml': (summary['weight_total'], channel_lists),
    }
    for name, (weights, lists) in outputs.items():
        if lists is not None and not lists:
            continue
        cfg = patch_observation_config(base_config, weights.to_dict(), lists)
        with (out_dir / name).open('w', encoding='utf-8') as f:
            yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
        print(f"Saved {out_dir / name}")
    detrimental = summary.index[summary['status_total'].eq('detrimental')].tolist()
    if detrimental:
        print(f"\nInvestigate net-detrimental inputs before down-weighting them: {detrimental}")
    print("\nCandidates only: validate each by fine-tuning and scoring with the frozen verification metric.")


if __name__ == '__main__':
    main()
