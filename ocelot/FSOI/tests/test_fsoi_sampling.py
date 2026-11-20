"""Sampling-design checks runnable without OCELOT, CUDA, or PyTorch."""

import itertools
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fsoi_sampling import population_summary, sample_rows  # noqa: E402


def coordinates(counts):
    return np.repeat([-75., -35., 15., 55.], counts), np.zeros(sum(counts))


class SamplingTests(unittest.TestCase):
    def test_unequal_probabilities_and_exact_size(self):
        lat, lon = coordinates([1, 1, 1, 7])
        for seed in range(30):
            idx, info = sample_rows(lat, lon, 10, 4, seed=seed)
            self.assertEqual(len(idx), 4)
            self.assertEqual(len(set(idx)), 4)
            np.testing.assert_allclose(info['inclusion_probability'], [1, 1, 1, 1/7])
            value = population_summary(np.ones((4, 2)), np.ones((4, 2), bool), info)
            self.assertAlmostEqual(value['sum_impact_ht'], 20)

    def test_empirical_probabilities_include_high_indices(self):
        lat, lon = coordinates([1, 1, 1, 7])
        hits = np.zeros(10)
        for seed in range(2000):
            idx, _ = sample_rows(lat, lon, 10, 4, seed=seed)
            hits[idx] += 1
        np.testing.assert_allclose(hits[:3], 2000)
        np.testing.assert_allclose(hits[3:] / 2000, np.full(7, 1/7), atol=0.035)

    def test_ht_unbiased_by_exhaustive_enumeration_with_missing_channels(self):
        lat, lon = coordinates([1, 2, 3, 6])
        _, info = sample_rows(lat, lon, 12, 6)
        groups = [np.arange(0, 1), np.arange(1, 3), np.arange(3, 6), np.arange(6, 12)]
        takes = info['stratum_sample_size'][np.r_[True, np.diff(info['stratum_id']) != 0]]
        population = np.column_stack([np.arange(12)**2, np.arange(12) - 5.])
        valid = np.ones_like(population, bool)
        valid[[1, 3, 8], 0] = False
        valid[[0, 2, 4, 6, 8, 10], 1] = False
        estimates = []
        for pieces in itertools.product(*[list(itertools.combinations(g, int(k))) for g, k in zip(groups, takes)]):
            idx = np.concatenate(pieces)
            design = dict(info, population_valid_counts_by_channel=valid.sum(axis=0))
            estimates.append([population_summary(population[idx], valid[idx], design, channel=c)['sum_impact_ht'] for c in range(2)])
        np.testing.assert_allclose(np.mean(estimates, axis=0), np.where(valid, population, 0).sum(axis=0), atol=1e-12)

    def test_zero_is_valid_and_unknown_value_is_excluded(self):
        values = np.array([[0., np.nan], [4., 6.]])
        valid = np.array([[True, False], [True, True]])
        design = dict(raw_n_observations=4, inclusion_probability=np.array([.5, .5]),
                      population_valid_counts_by_channel=np.array([4, 2]))
        result = population_summary(values, valid, design, channel=0)
        self.assertEqual(result['estimated_valid_values_ht'], 4)
        self.assertEqual(result['mean_impact_population'], 2)
        result = population_summary(values, valid, design, channel=1)
        self.assertEqual(result['population_valid_values'], 2)
        self.assertEqual(result['mean_impact_population'], 6)

    def test_census_and_small_cap_fallback(self):
        lat, lon = coordinates([1, 1, 1, 7])
        idx, info = sample_rows(lat, lon, 10, 12)
        self.assertEqual(info['sampling_design'], 'census')
        np.testing.assert_equal(info['inclusion_probability'], 1)
        idx, info = sample_rows(lat, lon, 10, 2)
        self.assertEqual(info['sampling_design'], 'simple_random_without_replacement')
        np.testing.assert_allclose(info['inclusion_probability'], .2)

    def test_invalid_coordinates_use_srs(self):
        lat, lon = coordinates([1, 1, 1, 7])
        lat[0] = np.nan
        _, info = sample_rows(lat, lon, 10, 4)
        np.testing.assert_allclose(info['inclusion_probability'], .4)

    def test_no_unrecorded_population_expansion(self):
        with self.assertRaises(ValueError):
            population_summary(np.ones((2, 1)), np.ones((2, 1), bool), {'raw_n_observations': 4})
        for pi in ([0, .5], [1, 2], [np.nan, 1]):
            with self.assertRaises(ValueError):
                population_summary(np.ones((2, 1)), np.ones((2, 1), bool), {'inclusion_probability': pi})

    def test_completely_missing_channel_has_zero_total_and_undefined_mean(self):
        _, design = sample_rows(None, None, 2, 2)
        result = population_summary(np.full((2, 1), np.nan), np.zeros((2, 1), bool), design)
        self.assertEqual(result['sum_impact_ht'], 0)
        self.assertEqual(result['population_valid_values'], 0)
        self.assertTrue(np.isnan(result['mean_impact_population']))

    def test_raw_and_expanded_totals_use_consistent_accumulation(self):
        values = np.array([[1e8], [1.], [-1e8]], dtype=np.float32)
        design = dict(raw_n_observations=6, inclusion_probability=np.full(3, .5))
        result = population_summary(values, np.ones_like(values, dtype=bool), design)
        self.assertEqual(result['sum_impact'], 1.)
        self.assertEqual(result['sum_impact_scaled_uniform'], 2.)
        self.assertEqual(result['sum_impact_ht'], 2.)


if __name__ == '__main__':
    unittest.main()
