# Testing regional mesh support

## 1. Standalone smoke test (no real data needed) — run this first

`tests/test_regional_mesh_smoke.py` exercises everything the regional port
added (mesh construction, both flat and hierarchical, obs<->mesh
connectivity, the `_mesh_conn` dispatcher, and a global-path regression
check) against synthetic data. From `gnn_model/`:

```
python tests/test_regional_mesh_smoke.py
```

Needs `torch`, `torch_geometric`, `networkx`, `scipy`, and `pyproj` installed
(per `requirements.txt`). It prints `[OK] ...` per check and `All regional
mesh smoke tests passed.` at the end, or raises an `AssertionError` naming
exactly what broke — including a dedicated check for the bidirectional-edge
bug that was caught during implementation (regional mesh graphs are built
bidirectionally, but the datamodule's edge symmetrization assumes the
one-direction-per-edge convention that the global mesh uses; left
unhandled, this would silently double every mesh-to-mesh edge for regional
runs).

## 2. Full model construction (needs your real `configs/observation_config.yaml`, no zarr data needed)

Once step 1 passes, sanity-check that `GNNLightning` actually builds in
regional mode:

```python
from weight_utils import load_weights_from_yaml
from gnn_model import GNNLightning

observation_config, feature_stats, instrument_weights, channel_weights, _ = load_weights_from_yaml("configs/observation_config.yaml")

model = GNNLightning(
    observation_config=observation_config,
    hidden_dim=32,
    mesh_geometry="regional",
    lon_min=-100.0, lon_max=-90.0, lat_min=35.0, lat_max=45.0,
    mesh_levels=2,
    feature_stats=feature_stats,
    instrument_weights=instrument_weights,
    channel_weights=channel_weights,
)
print(model.mesh_structure["geometry"], model.mesh_x.shape)
```

This checks the mesh buffers register correctly against the real
observation config (embedder dims, etc.) — the part step 1 can't reach.

## 3. Real training smoke run (needs actual zarr data)

Once you have data access:

```
python train_gnn.py --mesh_geometry regional \
  --lon_min -100 --lon_max -90 --lat_min 35 --lat_max 45 \
  --mesh_levels 2 --max_epochs 1 --data_path <your_zarr_path>
```

Watch for NaNs in the first few steps — that's the tell for the
fp16-overflow issue the `[-1,1]` normalization exists to prevent.

## Known gaps

- `precompute_mesh_edges.py` and `FSOI/fsoi_model_extensions.py` still call
  `obs_mesh_conn` (global) directly and would break on a regional model —
  they were out of scope for this pass.
- `origin/feature/regional` has a separate, pre-existing regional
  implementation as a standalone `gnn_model/REGIONAL/` pipeline, unmerged.
  Worth comparing notes against if questions come up.
