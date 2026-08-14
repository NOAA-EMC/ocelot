"""
Standalone smoke test for the regional mesh code added to gnn_model.

Exercises create_regional_mesh_structure() and the obs<->mesh connectivity
path directly, with synthetic data -- no zarr/observation data required.
Run from the gnn_model/ directory (or anywhere; sys.path is fixed up below):

    python tests/test_regional_mesh_smoke.py

Exits non-zero and prints the failing assertion on any problem.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from create_mesh_graph_regional import (
    create_regional_mesh_structure,
    obs_mesh_conn_regional,
    project_coords,
)

# Small CONUS-ish bounding box (degrees)
LON_MIN, LON_MAX = -100.0, -90.0
LAT_MIN, LAT_MAX = 35.0, 45.0


def test_flat_mesh_structure_shape():
    ms = create_regional_mesh_structure(
        LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, hierarchical=False, mesh_splits=2,
    )
    assert ms["geometry"] == "regional"
    for key in (
        "m2m_graphs", "mesh_lat_lon_list", "mesh_list",
        "m2m_edge_index_torch", "m2m_features_torch",
        "mesh_features_torch", "mesh_lat_lon_torch",
        "G_bottom_mesh", "cutoff_factor", "num_neighbors",
    ):
        assert key in ms, f"missing key: {key}"

    n_levels = len(ms["m2m_graphs"])
    assert n_levels == 1, f"flat mesh should have exactly 1 level, got {n_levels}"
    for lst in ("mesh_lat_lon_list", "m2m_edge_index_torch", "m2m_features_torch", "mesh_features_torch"):
        assert len(ms[lst]) == n_levels, f"{lst} length {len(ms[lst])} != n_levels {n_levels}"

    mx = ms["mesh_features_torch"][0]
    assert mx.shape[1] == 2, f"expected 2D (x,y) mesh features, got shape {tuple(mx.shape)}"
    assert torch.all(mx >= -1.0 - 1e-5) and torch.all(mx <= 1.0 + 1e-5), (
        f"mesh_features_torch not normalized to [-1, 1]: min={mx.min().item()}, max={mx.max().item()}"
    )

    # edge-direction bug check: m2m_edge_index_torch must be exactly half of
    # the raw (bidirectional) m2m_graphs edge_index at the same level.
    raw_edges = ms["m2m_graphs"][0].edge_index.shape[1]
    kept_edges = ms["m2m_edge_index_torch"][0].shape[1]
    assert kept_edges == raw_edges // 2, (
        f"expected m2m_edge_index_torch to keep half the bidirectional edges "
        f"(raw={raw_edges}, kept={kept_edges}) -- symmetrization in gnn_datamodule "
        f"would double-count edges otherwise"
    )
    assert ms["m2m_features_torch"][0].shape[0] == kept_edges

    print(f"[OK] flat mesh: {mx.shape[0]} nodes, {kept_edges} undirected edges")
    return ms


def test_hierarchical_mesh_structure_shape():
    ms = create_regional_mesh_structure(
        LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, hierarchical=True, mesh_splits=3,
    )
    n_levels = len(ms["m2m_graphs"])
    assert n_levels == 3, f"expected 3 hierarchical levels, got {n_levels}"
    assert len(ms["mesh_up_ei_list"]) == n_levels - 1
    assert len(ms["mesh_down_ei_list"]) == n_levels - 1
    assert len(ms["mesh_up_features_list"]) == n_levels - 1
    assert len(ms["mesh_down_features_list"]) == n_levels - 1

    # up/down edge counts should match per level-pair
    for i in range(n_levels - 1):
        assert ms["mesh_up_ei_list"][i].shape[1] == ms["mesh_down_ei_list"][i].shape[1]

    print(f"[OK] hierarchical mesh: {n_levels} levels, "
          f"node counts={[g.pos.shape[0] for g in ms['m2m_graphs']]}")
    return ms


def test_obs_mesh_connectivity(ms):
    rng = np.random.default_rng(0)
    n_obs = 50
    obs_lat = rng.uniform(LAT_MIN + 1, LAT_MAX - 1, size=n_obs)
    obs_lon = rng.uniform(LON_MIN + 1, LON_MAX - 1, size=n_obs)
    obs_xy = project_coords(obs_lat, obs_lon)
    assert obs_xy.shape == (n_obs, 2)

    edge_index_o2m, edge_attr_o2m = obs_mesh_conn_regional(
        obs_xy, ms["G_bottom_mesh"],
        cutoff_factor=ms["cutoff_factor"], num_neighbors=ms["num_neighbors"],
        o2m=True,
    )
    assert edge_index_o2m.shape[0] == 2
    assert edge_attr_o2m.shape[1] == 4
    assert edge_index_o2m.shape[1] > 0, "expected at least one obs->mesh edge"
    assert torch.all(edge_attr_o2m[:, 3] == 0), "4th edge_attr column should be zero-padded"
    # obs indices must be valid
    assert edge_index_o2m[0].max().item() < n_obs

    edge_index_m2o, _ = obs_mesh_conn_regional(
        obs_xy, ms["G_bottom_mesh"],
        cutoff_factor=ms["cutoff_factor"], num_neighbors=ms["num_neighbors"],
        o2m=False,
    )
    # o2m=False should just flip src/dst relative to o2m=True
    assert torch.equal(edge_index_m2o[0], edge_index_o2m[1])
    assert torch.equal(edge_index_m2o[1], edge_index_o2m[0])

    print(f"[OK] connectivity: {n_obs} obs -> {edge_index_o2m.shape[1]} obs-mesh edges")


def test_datamodule_dispatch():
    """Exercise the _mesh_conn dispatcher in gnn_datamodule.py directly."""
    from gnn_datamodule import _mesh_conn

    ms = create_regional_mesh_structure(
        LON_MIN, LON_MAX, LAT_MIN, LAT_MAX, hierarchical=False, mesh_splits=2,
    )
    rng = np.random.default_rng(1)
    obs_lat = rng.uniform(LAT_MIN + 1, LAT_MAX - 1, size=10)
    obs_lon = rng.uniform(LON_MIN + 1, LON_MAX - 1, size=10)

    edge_index, edge_attr = _mesh_conn(obs_lat, obs_lon, ms, o2m=True)
    assert edge_index.shape[0] == 2 and edge_index.shape[1] > 0
    assert edge_attr.shape[1] == 4

    # Global dispatch path should still work unchanged (regression check).
    from create_mesh_graph_global import create_mesh
    global_ms = create_mesh(splits=2, levels=1, hierarchical=False, plot=False)
    g_edge_index, g_edge_attr = _mesh_conn(obs_lat, obs_lon, global_ms, o2m=True)
    assert g_edge_index.shape[0] == 2 and g_edge_index.shape[1] > 0
    assert g_edge_attr.shape[1] == 4

    print("[OK] gnn_datamodule._mesh_conn dispatch (regional + global regression)")


if __name__ == "__main__":
    ms_flat = test_flat_mesh_structure_shape()
    test_hierarchical_mesh_structure_shape()
    test_obs_mesh_connectivity(ms_flat)
    test_datamodule_dispatch()
    print("\nAll regional mesh smoke tests passed.")
