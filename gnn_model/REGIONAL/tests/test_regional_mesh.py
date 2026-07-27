"""Unit tests for the regional mesh creation and obs-mesh connectivity.

Run from repo root::

    python -m unittest tests.test_regional_mesh -v
"""

from __future__ import annotations

import types
import unittest

import numpy as np

try:
    import torch
    from ocelot.create_mesh_graph_regional import (
        create_regional_mesh,
        obs_mesh_conn_regional,
        project_coords,
    )
    _AVAILABLE = True
except ImportError:  # pragma: no cover
    _AVAILABLE = False


def _make_xy(H=27, W=27, extent=1e6):
    """Return [2, H, W] projected coordinate array covering [0, extent]^2."""
    x = np.linspace(0, extent, W)
    y = np.linspace(0, extent, H)
    X, Y = np.meshgrid(x, y)
    return np.array([X, Y])


def _flat_args(**kwargs):
    base = dict(levels=None, hierarchical=False, plot=False,
                cutoff_factor=0.67, num_neighbors=4)
    base.update(kwargs)
    return types.SimpleNamespace(**base)


@unittest.skipUnless(_AVAILABLE, "torch / ocelot not installed")
class TestCreateRegionalMesh(unittest.TestCase):

    def test_flat_mesh_keys(self):
        ms = create_regional_mesh(_make_xy(), _flat_args())
        self.assertIn("m2m_graphs", ms)
        self.assertIn("mesh_pos", ms)
        self.assertIn("G_bottom_mesh", ms)
        self.assertIn("all_mesh_nodes", ms)

    def test_flat_mesh_single_graph(self):
        ms = create_regional_mesh(_make_xy(), _flat_args())
        self.assertEqual(len(ms["m2m_graphs"]), 1)

    def test_flat_mesh_node_positions_shape(self):
        ms = create_regional_mesh(_make_xy(), _flat_args())
        pos = ms["m2m_graphs"][0].pos
        # 27x27 grid → nlev=3, mesh_levels=2, flat merges to one graph
        self.assertEqual(pos.shape[1], 2)
        self.assertGreater(pos.shape[0], 0)

    def test_flat_mesh_edge_attr_shape(self):
        ms = create_regional_mesh(_make_xy(), _flat_args())
        ea = ms["m2m_graphs"][0].edge_attr
        # edge_attr = [length, vdiff_x, vdiff_y]
        self.assertEqual(ea.shape[1], 3)
        self.assertGreater(ea.shape[0], 0)

    def test_flat_mesh_edge_lengths_positive(self):
        ms = create_regional_mesh(_make_xy(), _flat_args())
        lengths = ms["m2m_graphs"][0].edge_attr[:, 0]
        self.assertTrue((lengths > 0).all())

    def test_flat_mesh_pos_within_grid_extent(self):
        extent = 1e6
        ms = create_regional_mesh(_make_xy(extent=extent), _flat_args())
        pos = ms["m2m_graphs"][0].pos
        self.assertTrue((pos >= 0).all())
        self.assertTrue((pos <= extent).all())

    def test_levels_cap(self):
        # Capping levels=1 should give fewer mesh nodes than levels=None
        ms_full = create_regional_mesh(_make_xy(), _flat_args(levels=None))
        ms_cap  = create_regional_mesh(_make_xy(), _flat_args(levels=1))
        n_full = ms_full["m2m_graphs"][0].pos.shape[0]
        n_cap  = ms_cap["m2m_graphs"][0].pos.shape[0]
        self.assertLessEqual(n_cap, n_full)

    def test_hierarchical_mesh_multiple_levels(self):
        ms = create_regional_mesh(_make_xy(), _flat_args(hierarchical=True))
        self.assertGreater(len(ms["m2m_graphs"]), 1)
        self.assertIn("mesh_up_graphs", ms)
        self.assertIn("mesh_down_graphs", ms)

    def test_hierarchical_mesh_level_sizes_decrease(self):
        ms = create_regional_mesh(_make_xy(), _flat_args(hierarchical=True))
        sizes = [g.pos.shape[0] for g in ms["m2m_graphs"]]
        # Coarser levels have fewer nodes
        for a, b in zip(sizes, sizes[1:]):
            self.assertGreater(a, b)

    def test_mesh_pos_matches_m2m_graphs(self):
        ms = create_regional_mesh(_make_xy(), _flat_args())
        for g, p in zip(ms["m2m_graphs"], ms["mesh_pos"]):
            self.assertEqual(g.pos.shape, p.shape)

    def test_g_bottom_mesh_has_pos_attributes(self):
        ms = create_regional_mesh(_make_xy(), _flat_args())
        G = ms["G_bottom_mesh"]
        for n in G.nodes:
            self.assertIn("pos", G.nodes[n])
            self.assertEqual(len(G.nodes[n]["pos"]), 2)


@unittest.skipUnless(_AVAILABLE, "torch / ocelot not installed")
class TestObsMeshConnRegional(unittest.TestCase):

    def setUp(self):
        xy = _make_xy()
        self.ms = create_regional_mesh(xy, _flat_args())
        self.G = self.ms["G_bottom_mesh"]
        self.args = _flat_args()
        rng = np.random.default_rng(42)
        self.obs_xy = rng.uniform(0, 1e6, (50, 2))

    def test_o2m_returns_two_tensors(self):
        ei, ea = obs_mesh_conn_regional(self.obs_xy, self.G, self.args, o2m=True)
        self.assertIsInstance(ei, torch.Tensor)
        self.assertIsInstance(ea, torch.Tensor)

    def test_o2m_edge_index_shape(self):
        ei, _ = obs_mesh_conn_regional(self.obs_xy, self.G, self.args, o2m=True)
        self.assertEqual(ei.shape[0], 2)
        self.assertGreater(ei.shape[1], 0)

    def test_o2m_edge_attr_4_features(self):
        _, ea = obs_mesh_conn_regional(self.obs_xy, self.G, self.args, o2m=True)
        self.assertEqual(ea.shape[1], 4)

    def test_o2m_edge_attr_4th_feature_is_zero(self):
        _, ea = obs_mesh_conn_regional(self.obs_xy, self.G, self.args, o2m=True)
        self.assertTrue((ea[:, 3] == 0).all())

    def test_o2m_distances_positive(self):
        _, ea = obs_mesh_conn_regional(self.obs_xy, self.G, self.args, o2m=True)
        self.assertTrue((ea[:, 0] > 0).all())

    def test_o2m_obs_indices_in_range(self):
        ei, _ = obs_mesh_conn_regional(self.obs_xy, self.G, self.args, o2m=True)
        n_obs = self.obs_xy.shape[0]
        n_mesh = len(list(self.G.nodes))
        self.assertTrue((ei[0] >= 0).all() and (ei[0] < n_obs).all())
        self.assertTrue((ei[1] >= 0).all() and (ei[1] < n_mesh).all())

    def test_m2o_flips_edge_direction(self):
        ei_o2m, _ = obs_mesh_conn_regional(self.obs_xy, self.G, self.args, o2m=True)
        ei_m2o, _ = obs_mesh_conn_regional(self.obs_xy, self.G, self.args, o2m=False)
        # Same edges, just src/dst swapped
        self.assertEqual(ei_o2m.shape[1], ei_m2o.shape[1])
        self.assertTrue((ei_o2m[0] == ei_m2o[1]).all())
        self.assertTrue((ei_o2m[1] == ei_m2o[0]).all())

    def test_m2o_edge_attr_unchanged(self):
        _, ea_o2m = obs_mesh_conn_regional(self.obs_xy, self.G, self.args, o2m=True)
        _, ea_m2o = obs_mesh_conn_regional(self.obs_xy, self.G, self.args, o2m=False)
        self.assertTrue(torch.allclose(ea_o2m, ea_m2o))

    def test_knn_fallback_for_distant_obs(self):
        # Radius search finds nothing; KNN fallback must still produce edges.
        far_obs = np.array([[1e9, 1e9]])
        args = _flat_args(cutoff_factor=1e-10, num_neighbors=1)
        ei, ea = obs_mesh_conn_regional(far_obs, self.G, args, o2m=True)
        self.assertEqual(ei.shape[0], 2)
        self.assertGreater(ei.shape[1], 0)
        self.assertEqual(ea.shape[1], 4)

    def test_larger_cutoff_gives_more_edges(self):
        ei_small, _ = obs_mesh_conn_regional(
            self.obs_xy, self.G, _flat_args(cutoff_factor=0.1), o2m=True
        )
        ei_large, _ = obs_mesh_conn_regional(
            self.obs_xy, self.G, _flat_args(cutoff_factor=2.0), o2m=True
        )
        self.assertGreaterEqual(ei_large.shape[1], ei_small.shape[1])


@unittest.skipUnless(_AVAILABLE, "torch / ocelot not installed")
class TestProjectCoords(unittest.TestCase):

    def test_returns_2d_array(self):
        lat = np.array([38.5, 39.0])
        lon = np.array([-97.5, -97.0])
        xy = project_coords(lat, lon)
        self.assertEqual(xy.shape, (2, 2))

    def test_origin_near_zero(self):
        # The projection origin (lat=38.5, lon=-97.5) should map close to (0, 0)
        xy = project_coords(np.array([38.5]), np.array([-97.5]))
        self.assertAlmostEqual(float(xy[0, 0]), 0.0, delta=1.0)
        self.assertAlmostEqual(float(xy[0, 1]), 0.0, delta=1.0)

    def test_longitude_increases_x(self):
        lat = np.array([38.5, 38.5])
        lon = np.array([-98.0, -97.0])
        xy = project_coords(lat, lon)
        self.assertGreater(xy[1, 0], xy[0, 0])

    def test_latitude_increases_y(self):
        lat = np.array([38.0, 39.0])
        lon = np.array([-97.5, -97.5])
        xy = project_coords(lat, lon)
        self.assertGreater(xy[1, 1], xy[0, 1])


if __name__ == "__main__":
    unittest.main()
