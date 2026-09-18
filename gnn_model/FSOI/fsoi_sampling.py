"""Row sampling and design-weighted summaries for FSOI decoder subsamples.

HT weights correct unequal selection probabilities for fixed row contributions.
They do not remove dependence of a gradient on the sampled intervention path.
"""

from __future__ import annotations

import numpy as np


def sample_rows(lat, lon, n_raw, max_n, seed=42, grid_deg=10.0):
    """Sample whole rows without replacement, returning indices and their design.

    Allocation is deterministic given cell counts; sampling within each cell is
    uniform. Invalid coordinates, sparse support, or a cap smaller than the
    number of occupied cells use simple random sampling of all rows.
    """
    n_raw, max_n = int(n_raw), int(max_n)
    if n_raw <= 0 or max_n <= 0:
        raise ValueError("Row population and decoder cap must be positive")
    if not np.isfinite(grid_deg) or grid_deg <= 0:
        raise ValueError("grid_deg must be positive and finite")
    n = min(max_n, n_raw)
    rng = np.random.default_rng(seed)
    method = "simple_random_without_replacement"
    cell_id = np.zeros(n_raw, dtype=np.int64)
    counts = np.array([n_raw], dtype=np.int64)
    cells = np.array([-1], dtype=np.int64)
    inverse = np.zeros(n_raw, dtype=np.int64)
    if lat is not None and lon is not None:
        lat, lon = np.asarray(lat).reshape(-1), np.asarray(lon).reshape(-1)
        if (lat.size == lon.size == n_raw and np.isfinite(lat).all()
                and np.isfinite(lon).all() and (np.abs(lat) <= 90).all()):
            lon = (lon + 180) % 360 - 180
            n_lat, n_lon = int(np.ceil(180 / grid_deg)), int(np.ceil(360 / grid_deg))
            iy = np.clip(np.floor((lat + 90) / grid_deg).astype(int), 0, n_lat - 1)
            ix = np.clip(np.floor((lon + 180) / grid_deg).astype(int), 0, n_lon - 1)
            cell_id = iy * n_lon + ix
            cells, inverse, counts = np.unique(cell_id, return_inverse=True, return_counts=True)
            if len(cells) >= 4 and n >= len(cells):
                method = "stratified_srs_without_replacement"
    if n == n_raw:
        idx = np.arange(n_raw)
        allocation = counts.copy()
        method = "census"
    elif method == "stratified_srs_without_replacement":
        quota = n * counts / n_raw
        allocation = np.minimum(counts, np.maximum(1, np.rint(quota).astype(int)))
        while allocation.sum() > n:
            excess = np.where(allocation > 1, allocation - quota, -np.inf)
            allocation[np.argmax(excess)] -= 1
        while allocation.sum() < n:
            deficit = np.where(allocation < counts, quota - allocation, -np.inf)
            allocation[np.argmax(deficit)] += 1
        idx = np.sort(np.concatenate([
            rng.choice(np.flatnonzero(inverse == h), size=int(take), replace=False)
            for h, take in enumerate(allocation)
        ]))
    else:
        idx = np.sort(rng.choice(n_raw, size=n, replace=False))
        cells, counts, allocation = np.array([-1]), np.array([n_raw]), np.array([n])
        inverse = np.zeros(n_raw, dtype=np.int64)
    pi_by_cell = allocation / counts
    selected_cell = inverse[idx]
    design = dict(
        raw_n_observations=n_raw, sampled_n_observations=n,
        sample_scale=float(n_raw / n), is_subsampled=n < n_raw,
        sampling_design=method, sampling_seed=int(seed), sampling_grid_deg=float(grid_deg),
        row_indices=idx, stratum_id=cells[selected_cell],
        stratum_population=counts[selected_cell], stratum_sample_size=allocation[selected_cell],
        inclusion_probability=pi_by_cell[selected_cell],
    )
    assert len(idx) == n and len(np.unique(idx)) == n
    return idx, design


def population_summary(values, valid, design, row_mask=None, channel=None):
    """Compute HT totals and valid-value counts with the same row probabilities.

    A missing channel contributes zero; a valid observed zero remains eligible.
    The HT total divided by a known full-population valid count is distinguished
    from a Hajek ratio using the estimated valid count.
    """
    values = np.asarray(values)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 2 or valid.shape != values.shape:
        raise ValueError("Expected aligned [sampled row, channel] values and validity")
    n, c = values.shape
    design = design or {}
    pi = np.asarray(design.get("inclusion_probability", np.ones(n)), dtype=float)
    raw_n = int(design.get("raw_n_observations", n))
    if "inclusion_probability" not in design and raw_n != n:
        raise ValueError("Population expansion requires saved row inclusion probabilities")
    if pi.shape != (n,) or not np.isfinite(pi).all() or np.any((pi <= 0) | (pi > 1)):
        raise ValueError("Every sampled row must have an inclusion probability in (0, 1]")
    row_mask = np.ones(n, dtype=bool) if row_mask is None else np.asarray(row_mask, dtype=bool)
    if row_mask.shape != (n,):
        raise ValueError("Row-domain mask is not aligned with sampled rows")
    if channel is not None:
        values = values[:, channel:channel + 1]
        valid = valid[:, channel:channel + 1]
    choose = valid & row_mask[:, None]
    if not np.isfinite(values[choose]).all():
        raise ValueError("Non-finite FSOI at a valid observation component")
    contribution = np.where(choose, values, 0.0)
    ht = float(np.sum(contribution / pi[:, None], dtype=np.float64))
    estimated_count = float(np.sum(choose / pi[:, None], dtype=np.float64))
    population_count = float("nan")
    if raw_n == n:
        population_count = int(choose.sum())
    elif row_mask.all() and "population_valid_counts_by_channel" in design:
        counts = np.asarray(design["population_valid_counts_by_channel"])
        population_count = int(counts.sum() if channel is None else counts[channel])
    raw_sum = float(np.sum(contribution, dtype=np.float64))
    return dict(
        sum_impact=raw_sum, sum_impact_ht=ht, sum_impact_scaled=ht,
        sum_impact_scaled_uniform=raw_sum * raw_n / n,
        population_scaling_method="horvitz_thompson_row_inclusion",
        estimated_valid_values_ht=estimated_count,
        population_valid_values=population_count,
        mean_impact_population=ht / population_count if population_count > 0 else float("nan"),
        mean_impact_hajek=ht / estimated_count if estimated_count > 0 else float("nan"),
    )
