"""Summarize the metadata-only target audit and freeze the supported target groups.

No FSOI signs or losses are used: groups are selected from coverage alone,
before any impact result is examined.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

SETTINGS = ['n_sin_lat', 'n_lon', 'min_observations_per_cell',
            'min_cells_per_group', 'min_observations_per_group']


def load_records(root):
    frames = []
    for path in sorted(Path(root).rglob('target_metric/pair*.csv')):
        df = pd.read_csv(path)
        df['run'] = str(path.parents[2])
        frames.append(df)
    if not frames:
        raise ValueError(f"No target-metric coverage files beneath {root}")
    raw = pd.concat(frames, ignore_index=True)
    raw['eligible'] = raw['eligible'].astype(str).str.lower().eq('true')
    return raw


def run_cycles(raw):
    """Cycles attempted per run; a cycle without a target node counts as unsupported."""
    counts = {}
    for run, group in raw.groupby('run'):
        n = group['pair_idx'].nunique()
        inventory = Path(run) / 'evaluation' / 'target_metric_cycles.csv'
        if inventory.is_file():
            n = max(n, pd.read_csv(inventory)['pair_idx'].nunique())
        counts[run] = n
    return pd.Series(counts, name='run_cycles')


def summarize(root, raw=None):
    raw = load_records(root) if raw is None else raw
    keys = ['run', 'target_instrument', 'variable', 'pressure_hpa']
    summary = raw.groupby(keys, dropna=False).agg(
        cycles=('eligible', 'size'), eligible_cycles=('eligible', 'sum'),
        min_cells=('n_cells', 'min'), median_cells=('n_cells', 'median'),
        min_targets=('n_scored_targets', 'min'), median_targets=('n_scored_targets', 'median'),
        mean_invalid_coordinates=('n_invalid_coordinates', 'mean'),
    ).reset_index()
    summary['run_cycles'] = summary['run'].map(run_cycles(raw))
    summary['eligible_fraction'] = summary['eligible_cycles'] / summary['run_cycles']
    return summary


def parse_restrictions(values):
    """Parse target:variable:min_hpa quality restrictions into {(target, variable): min_hpa}."""
    restrictions = {}
    for value in values or []:
        parts = str(value).split(':')
        if len(parts) != 3:
            raise ValueError(f"Expected target:variable:min_hpa, got {value!r}")
        target, variable, floor = parts
        if float(floor) <= 0:
            raise ValueError(f"Minimum pressure must be positive: {value!r}")
        restrictions[(target, variable)] = float(floor)
    return restrictions


def freeze(raw, summary, min_fraction=0.95, restrictions=None):
    """Keep groups eligible in at least min_fraction of cycles in every audit run.

    A restriction additionally drops levels above (lower pressure than) a floor for
    one variable, for observations that coverage cannot judge, such as radiosonde
    humidity that operational QC never checks in the stratosphere.
    """
    if not 0 < min_fraction <= 1:
        raise ValueError("min_fraction must lie in (0, 1]")
    restrictions = restrictions or {}
    cycles = run_cycles(raw)
    spec = {}
    for target, group in summary.groupby('target_instrument'):
        runs = sorted(group['run'].unique())
        records = raw[raw['target_instrument'].eq(target)].assign(
            pressure_key=lambda d: d['pressure_hpa'].fillna(-1.0))
        settings = records[SETTINGS].drop_duplicates()
        if len(settings) != 1:
            raise ValueError(f"{target}: audit runs used different coverage settings")
        worst = group.groupby(['variable', 'pressure_hpa'], dropna=False).agg(
            worst_fraction=('eligible_fraction', 'min'), n_runs=('run', 'nunique')).reset_index()
        worst['kept'] = worst['worst_fraction'].ge(min_fraction) & worst['n_runs'].eq(len(runs))
        worst['reason'] = np.where(worst['kept'], 'ok', 'below_coverage_threshold')
        target_restrictions = {v: p for (t, v), p in restrictions.items() if t == target}
        for variable, floor in target_restrictions.items():
            if variable not in set(worst['variable']):
                raise ValueError(f"{target}: restriction names unknown variable {variable!r}")
            restricted = worst['variable'].eq(variable) & worst['pressure_hpa'].lt(floor) & worst['kept']
            worst.loc[restricted, ['kept', 'reason']] = [False, 'quality_restriction']
        order = records.groupby('variable')['target_channel'].min().sort_values().index
        variables = [v for v in order if worst.loc[worst['variable'].eq(v), 'kept'].any()]
        if not variables:
            raise ValueError(f"{target}: no target group reaches eligible fraction {min_fraction}")
        chosen = worst[worst['kept'] & worst['variable'].isin(variables)]
        levels = {}
        if chosen['pressure_hpa'].notna().any():
            levels = {v: sorted(chosen.loc[chosen['variable'].eq(v), 'pressure_hpa'].astype(float),
                                reverse=True) for v in variables}
        # A cycle is scored only if every chosen group is eligible in it.
        keys = chosen.assign(pressure_key=chosen['pressure_hpa'].fillna(-1.0))[['variable', 'pressure_key']]
        selected = records.merge(keys, on=['variable', 'pressure_key'])
        per_cycle = selected.groupby(['run', 'pair_idx'])['eligible'].agg(['size', 'all'])
        complete = (per_cycle['all'] & per_cycle['size'].eq(len(keys))).groupby(level='run').sum()
        retention = (complete.reindex(runs).fillna(0) / cycles.reindex(runs)).fillna(0.0)
        dropped = worst[~worst['kept']]
        spec[target] = dict(
            target_variables=list(variables), levels_by_variable=levels,
            min_eligible_fraction=min_fraction,
            quality_restrictions={v: p for v, p in target_restrictions.items()},
            metric_settings={k: int(settings.iloc[0][k]) for k in SETTINGS},
            audit_runs=runs,
            expected_cycle_retention={run: float(retention[run]) for run in runs},
            dropped_groups=[dict(variable=row.variable,
                                 pressure_hpa=None if pd.isna(row.pressure_hpa) else float(row.pressure_hpa),
                                 worst_eligible_fraction=float(row.worst_fraction),
                                 reason=row.reason)
                            for row in dropped.itertuples()],
        )
    return spec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--freeze-output', type=Path,
                        help='Write the frozen target groups (JSON keyed by target) for make_target_metric_config.py')
    parser.add_argument('--min-eligible-fraction', type=float, default=0.95)
    parser.add_argument('--restrict', action='append', metavar='TARGET:VARIABLE:MIN_HPA',
                        help='Drop levels above this pressure for one variable, e.g. '
                             'radiosonde:dewpoint_temperature:300 for unchecked stratospheric humidity')
    args = parser.parse_args()
    raw = load_records(args.root)
    result = summarize(args.root, raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    if args.freeze_output is None:
        print("Select supported groups from coverage, then freeze the definition before examining impact results.")
        return
    spec = freeze(raw, result, args.min_eligible_fraction, parse_restrictions(args.restrict))
    args.freeze_output.parent.mkdir(parents=True, exist_ok=True)
    with args.freeze_output.open('x', encoding='utf-8') as f:
        json.dump(spec, f, indent=2)
    for target, entry in spec.items():
        print(f"\n{target}: variables={entry['target_variables']}")
        for variable, levels in entry['levels_by_variable'].items():
            print(f"  {variable}: {[int(p) for p in levels]} hPa")
        print(f"  dropped groups: {len(entry['dropped_groups'])}")
        low = {Path(r).name: round(v, 3) for r, v in entry['expected_cycle_retention'].items() if v < 0.9}
        if low:
            print(f"  WARNING: fewer than 90% of cycles keep every chosen group: {low}")
    print(f"\nWrote {args.freeze_output}")


if __name__ == '__main__':
    main()
