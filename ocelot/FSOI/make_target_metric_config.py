"""Generate explicit, standalone configurations for target-coverage audits and runs.

Run configurations are generated from the frozen coverage specification written by
summarize_target_coverage.py --freeze-output, so every season uses the same groups.
"""

import argparse
import hashlib
import json
from pathlib import Path

import yaml

from fsoi_target_metric import PRESSURES, pressure_indices


def make_config(base, *, mode='audit', n_sin_lat=18, n_lon=36,
                min_cell=2, min_cells=3, min_targets=10, levels=None, levels_by_variable=None,
                variables=None, provenance=None):
    config = json.loads(json.dumps(base))
    forecast = config['forecast']
    targets = forecast.get('target_instruments')
    if not isinstance(targets, list) or len(targets) != 1:
        raise ValueError("The base configuration must select one observation-space target")
    if targets[0] not in {'radiosonde', 'aircraft', 'surface_obs'}:
        raise ValueError("Supported targets: radiosonde, aircraft, surface_obs")
    selected_levels = PRESSURES.tolist() if levels is None else list(levels)
    pressure_indices(selected_levels)
    for value in (levels_by_variable or {}).values():
        pressure_indices(value)
    if variables is not None:
        variables = list(variables)
        if not variables or len(set(variables)) != len(variables):
            raise ValueError("Target variables must be a nonempty list of unique names")
        if set(levels_by_variable or {}) - set(variables):
            raise ValueError("levels_by_variable names variables outside target_variables")
        forecast['target_variables'] = variables
    if mode not in {'audit', 'run'}:
        raise ValueError("Invalid metric mode")
    if any(int(x) != x or x < 1 for x in (n_sin_lat, n_lon, min_cell, min_cells, min_targets)):
        raise ValueError("Grid dimensions and coverage thresholds must be positive integers")
    forecast.pop('use_area_weights', None)
    forecast.update(use_instrument_weights=False,
                    use_channel_weights=False, loss_reduction='mean', impact_factor=0.5,
                    stratify_by_variable=True, stratify_by_pressure=targets[0] != 'surface_obs')
    forecast['verification_metric'] = dict(
        spatial_weighting='equal_area', group_balancing='variable_level',
        n_sin_lat=n_sin_lat, n_lon=n_lon, min_observations_per_cell=min_cell,
        min_cells_per_group=min_cells, min_observations_per_group=min_targets,
        pressure_levels_hpa=selected_levels, levels_by_variable=levels_by_variable or {},
        sparse_policy='exclude_cycle', audit_only=mode == 'audit',
    )
    if mode == 'audit':
        validation = config.setdefault('validation', {})
        for flag in ('finite_difference_check', 'directional_derivative_check', 'float64_fd_check'):
            validation[flag] = False
    if provenance:
        config['target_metric_freeze'] = dict(provenance)
    config.setdefault('output', {})['save_csv'] = True
    return config


def load_frozen_spec(path, target):
    """Return the frozen groups for one target and a provenance record."""
    raw = Path(path).read_bytes()
    spec = json.loads(raw).get(target)
    if not spec:
        raise ValueError(f"{path} contains no frozen specification for {target}")
    provenance = dict(spec_file=str(path), spec_sha256=hashlib.sha256(raw).hexdigest(),
                      min_eligible_fraction=spec['min_eligible_fraction'],
                      audit_runs=spec['audit_runs'],
                      expected_cycle_retention=spec['expected_cycle_retention'])
    return spec, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', choices=['radiosonde', 'aircraft', 'surface_obs'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['audit', 'run'], default='audit')
    parser.add_argument('--frozen-spec', type=Path,
                        help='JSON from summarize_target_coverage.py --freeze-output (required for --mode run)')
    parser.add_argument('--allow-unfrozen', action='store_true',
                        help='Permit a run config without a frozen spec (development only)')
    parser.add_argument('--n-sin-lat', type=int, default=18)
    parser.add_argument('--n-lon', type=int, default=36)
    parser.add_argument('--min-observations-per-cell', type=int, default=2)
    parser.add_argument('--min-cells-per-group', type=int, default=3)
    parser.add_argument('--min-observations-per-group', type=int, default=10)
    parser.add_argument('--levels-hpa', type=float, nargs='+')
    parser.add_argument('--levels-by-variable', type=Path, help='JSON object mapping variable names to fixed hPa lists')
    args = parser.parse_args()
    name = 'radiosonde_all' if args.target == 'radiosonde' else args.target
    base_file = Path(__file__).parent / 'configs' / f'fsoi_config_{name}.yaml'
    with base_file.open(encoding='utf-8') as f:
        base = yaml.safe_load(f)
    grid = dict(n_sin_lat=args.n_sin_lat, n_lon=args.n_lon,
                min_cell=args.min_observations_per_cell, min_cells=args.min_cells_per_group,
                min_targets=args.min_observations_per_group)
    levels, mapping, variables, provenance = args.levels_hpa, None, None, None
    if args.frozen_spec:
        if args.levels_hpa or args.levels_by_variable:
            parser.error("--frozen-spec already fixes the levels; do not pass --levels-hpa/--levels-by-variable")
        spec, provenance = load_frozen_spec(args.frozen_spec, args.target)
        s = spec['metric_settings']  # Score with exactly the thresholds that were audited.
        grid = dict(n_sin_lat=s['n_sin_lat'], n_lon=s['n_lon'],
                    min_cell=s['min_observations_per_cell'], min_cells=s['min_cells_per_group'],
                    min_targets=s['min_observations_per_group'])
        mapping, variables = spec['levels_by_variable'] or None, spec['target_variables']
        if mapping:
            used = {float(p) for values in mapping.values() for p in values}
            levels = [p for p in PRESSURES.tolist() if p in used]
    elif args.mode == 'run' and not args.allow_unfrozen:
        parser.error("--mode run requires --frozen-spec; freeze the groups from the coverage audit first")
    elif args.levels_by_variable:
        with args.levels_by_variable.open(encoding='utf-8') as f:
            mapping = json.load(f)
    config = make_config(base, mode=args.mode, levels=levels, levels_by_variable=mapping,
                         variables=variables, provenance=provenance, **grid)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as f:
        yaml.safe_dump(config, f, sort_keys=False)
    frozen = f"frozen from {args.frozen_spec}" if args.frozen_spec else "not frozen"
    print(f"Wrote {args.output}; mode={args.mode}; {frozen}.")


if __name__ == '__main__':
    main()
