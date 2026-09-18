"""
Utilities for heterogeneous weather graphs: node-type naming, edge relation constants,
and raw **input** node feature layout (shared by ``GraphDataset`` and AR feedback in the model).
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from ocelot.constants import FEATURES_AUX_DIM

# ---------------------------------------------------------------------------
# Naming (dataset + model must agree)
# ---------------------------------------------------------------------------

# When ``num_target_bins > 1``, each target node type also has obs→mesh edges built with the
# **same** ``obs_mesh_conn(..., o2m=True)`` recipe as ``(obs_input, "to", mesh)``, using that
# horizon's target lat/lon. Used to inject predictions at target locations back into the mesh
# (same graph structure as the encoder path). Relation must differ from ``"to"`` so PyG keeps
# a separate edge store.
ENCODE_OBS_TO_MESH_REL = "encode_obs_to_mesh"


def target_node_type(
        obs_type: str,
        inst_name: str,
        horizon: int,
        num_target_bins: int) -> str:
    """
    Node type name for decoder targets. Must end with ``_target`` so ``decode()`` picks it up.

    - ``num_target_bins <= 1``: ``{obs}_{inst}_target`` (backward compatible).
    - ``num_target_bins > 1``: ``{obs}_{inst}_h{horizon}_target`` for horizon in ``0 .. k-1``
      (bin at offset ``idx + 1 + horizon`` from the input bin index).
    """
    if num_target_bins <= 1:
        return f"{obs_type}_{inst_name}_target"
    return f"{obs_type}_{inst_name}_h{horizon}_target"


# ---------------------------------------------------------------------------
# Input feature vectors (``input_dim`` layout)
# ---------------------------------------------------------------------------


def build_satellite_input_x(
    features_norm: torch.Tensor,
    features_meta: torch.Tensor,
    features_aux: torch.Tensor,
) -> torch.Tensor:
    """
    Satellite input node features: ``[features_norm, features_meta, features_aux]``.

    Matches ``GraphDataset`` (graph_dataset.py) satellite branch for ``*_input``.
    """
    return torch.cat([features_norm, features_meta, features_aux], dim=1)


def build_conventional_input_x(
    features_norm: torch.Tensor,
    features_aux: torch.Tensor,
    features_valid_mask: torch.Tensor,
    features_meta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Conventional input node features: ``[features_norm, features_meta?, features_aux, mask]``.

    Matches ``GraphDataset`` conventional branch for ``*_input``.
    """
    input_parts = [features_norm]
    if features_meta is not None and torch.is_tensor(features_meta):
        input_parts.append(features_meta.float())
    input_parts.extend([features_aux, features_valid_mask])
    return torch.cat(input_parts, dim=1)


def build_feedback_input_x_from_target(
    obs_type: str,
    pred_norm: torch.Tensor,
    target_store: Any,
    num_metadata: int,
) -> torch.Tensor:
    """
    Build pseudo-input ``x`` for AR feedback using the **same** layout as real inputs.

    ``target_store`` is the PyG node store for the previous horizon (``…_h*_target``):
    satellite ``.x`` is ``cat(meta, aux)``; conventional ``.x`` is ``cat(meta, aux)``
    or ``aux`` only; ``valid_mask`` is used for conventional like the dataset.

    Args:
        obs_type: ``"satellite"`` or other (conventional).
        pred_norm: Predicted or teacher ``features_norm`` at target rows, shape ``(N, target_dim)``.
        target_store: ``data[prev_target_type]`` with ``.x`` and (for conventional) ``.valid_mask``.
        num_metadata: ``len(inst_cfg.get("metadata", []))`` for this instrument.
    """
    tx = target_store.x
    if obs_type == "satellite":
        if num_metadata > 0:
            meta = tx[:, :num_metadata]
            aux = tx[:, num_metadata: num_metadata + FEATURES_AUX_DIM]
        else:
            meta = pred_norm.new_zeros(pred_norm.size(0), 0)
            aux = tx
        return build_satellite_input_x(pred_norm, meta, aux)

    # Conventional: same order as GraphDataset lines 208–213
    vm = target_store.valid_mask.float()
    if num_metadata > 0:
        meta = tx[:, :num_metadata]
        aux = tx[:, num_metadata:]
        return build_conventional_input_x(
            pred_norm, aux, vm, features_meta=meta)
    return build_conventional_input_x(pred_norm, tx, vm, features_meta=None)
