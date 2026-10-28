"""Synthetic-window construction for AR rollout (feedback of predictions as inputs)."""

import copy
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch_geometric.data import HeteroData

from ar.input_features import _build_input_rows, _instrument_layout


_ALIGN_DIAG_DONE: set = set()
_MASK_DIAG_DONE: set = set()
_MATCH_DIAG_DONE: set = set()
_SATCOUNT_DONE: set = set()


def _find_target_steps(window: HeteroData, inst_name: str) -> List[Tuple[int, str]]:
    """Return (step_index, node_type) pairs sorted by step index for inst_name."""
    steps = []
    for nt in window.node_types:
        if nt.startswith(f"{inst_name}_target_step"):
            suffix = nt[len(f"{inst_name}_target_step"):]
            try:
                steps.append((int(suffix), nt))
            except ValueError:
                pass
    return sorted(steps)


def _recompute_target_metadata_times(
    window: HeteroData,
    ar_offset_hours: float,
    latent_step_hours: float,
    init_time_unix: Optional[int],
    target_time_feature_dim: int = 5,
    verbose: bool = False,
) -> None:
    """Refresh target-side time conditioning and zero viewing geometry on all
    ``*_target_step{i}`` nodes in `window`, in place.
    """
    if init_time_unix is None:
        if verbose:
            print("[AR-META] init_time_unix is None; skipping recompute")
        return

    from process_timeseries import _encode_target_time_features

    for node_type in window.node_types:
        if "_target_step" not in node_type:
            continue
        try:
            step_idx = int(node_type.split("_target_step")[1])
        except ValueError:
            continue

        store = window[node_type]

        if hasattr(store, "x") and store.x is not None and store.x.numel() > 0:
            if verbose:
                pre_absmax = float(store.x.abs().max())
                pre_nonzero = int((store.x != 0).sum())
            store.x = torch.zeros_like(store.x)
            if verbose:
                print(f"[AR-META] {node_type}: scan/x zeroed "
                      f"(was absmax={pre_absmax:.4f}, nonzero={pre_nonzero}/{store.x.numel()}); "
                      f"now absmax={float(store.x.abs().max()):.4f}")

        tm = getattr(store, "target_metadata", None)
        if tm is None or tm.numel() == 0 or tm.size(1) < target_time_feature_dim:
            if verbose:
                print(f"[AR-META] {node_type}: no usable target_metadata; time features unchanged")
            continue

        N = tm.shape[0]
        lead_seconds = (float(ar_offset_hours) + (step_idx + 0.5) * float(latent_step_hours)) * 3600.0
        target_time_unix = int(round(init_time_unix + lead_seconds))

        lon = getattr(store, "lon", None)
        if lon is not None and lon.numel() == N:
            lons_deg = lon.detach().cpu().numpy()
        else:
            lons_deg = np.zeros(N, dtype=np.float64)

        if verbose:
            pre_tail = tm[0, -target_time_feature_dim:].detach().cpu().numpy().copy() if N else None

        target_times_unix = np.full(N, target_time_unix, dtype=np.int64)
        time_feat_np = _encode_target_time_features(target_times_unix, lons_deg)
        time_feat = torch.from_numpy(time_feat_np).to(dtype=tm.dtype, device=tm.device)
        tm[:, -target_time_feature_dim:] = time_feat
        store.target_metadata = tm

        if verbose:
            import datetime as _dt
            vt = _dt.datetime.utcfromtimestamp(target_time_unix).strftime("%Y-%m-%d %H:%M")
            post_tail = tm[0, -target_time_feature_dim:].detach().cpu().numpy() if N else None
            print(f"[AR-META] {node_type}: step={step_idx} "
                  f"ar_offset={ar_offset_hours:.1f}h lead={lead_seconds/3600:.1f}h "
                  f"valid={vt} N={N}")
            if N:
                print(f"[AR-META]   time_feat[0] pre={np.array2string(pre_tail, precision=4)} "
                      f"post={np.array2string(post_tail, precision=4)}")

        obs_ts = getattr(store, "obs_time_unix", None)
        if obs_ts is not None and obs_ts.numel() == N:
            store.obs_time_unix = torch.full(
                (N,), target_time_unix, dtype=obs_ts.dtype, device=obs_ts.device
            )


def build_synthetic_window(
    predictions: Dict[str, List[torch.Tensor]],
    prev_window: HeteroData,
    template_window: HeteroData,
    observation_config: Optional[dict] = None,
    recompute_target_metadata: bool = False,
    ar_offset_hours: float = 0.0,
    latent_step_hours: Optional[float] = None,
    init_time_unix: Optional[int] = None,
    target_time_feature_dim: int = 5,
    verbose: bool = False,
) -> HeteroData:
    """Build a synthetic observation graph for AR feedback.

    The input observations for window k+1 are the decoded predictions of
    window k, placed at window k's target node locations.  All latent steps
    are flattened into a single input node set, matching the 12h coverage
    of a real first-window input.

    For each instrument:
      - synthetic[{inst}_input].x        <- predictions, feature-aligned to input_dim
      - synthetic[{inst}_input].lat/lon  <- prev_window[{inst}_target_step{i}].lat/lon
      - synthetic[{inst}_input -> mesh]  <- prev_window[mesh -> {inst}_target_step{i}]
                                           edge_index flipped, with per-step offset

    Args:
        predictions:     Dict[base_type, List[Tensor]] from GNNLightning.forward().
                         One tensor per latent step, shape [N_step, target_dim].
        prev_window:     Window k-1 HeteroData.  Provides target node positions,
                         metadata, and decoder edges.
        template_window: Window k HeteroData.  Provides mesh nodes, target nodes
                         for window k's forward pass, and decoder edges for window k.
        observation_config: Per-instrument config (metadata, features, sat_ids),
                         used to place predictions in the real input-row layout.
        recompute_target_metadata: Refresh target-side time conditioning and
                         zero viewing geometry on the copied target nodes.
                         Required when template_window is itself synthetic.
        ar_offset_hours: Hours from the rollout init to the start of the window
                         being built. Used only if recompute_target_metadata.
        latent_step_hours: Hours per latent step. Required if
                         recompute_target_metadata.
        init_time_unix:  Rollout init time, Unix seconds. Required if
                         recompute_target_metadata.
        target_time_feature_dim: Width of the trailing time-feature block in
                         target_metadata.

    Returns:
        HeteroData with synthetic input nodes and real window-k target structure.
    """
    if observation_config is None:
        raise ValueError("build_synthetic_window requires observation_config to build input rows.")
    synthetic = copy.deepcopy(template_window)

    for base_type, pred_list in predictions.items():
        if not pred_list:
            continue

        # Resolve inst_name
        if "_target_step" in base_type:
            inst_name = base_type.split("_target_step")[0]
        elif base_type.endswith("_target"):
            inst_name = base_type[: -len("_target")]
        else:
            continue

        input_node_type = f"{inst_name}_input"
        if input_node_type not in template_window.node_types:
            continue

        template_x = template_window[input_node_type].x
        if template_x is None or template_x.numel() == 0:
            continue

        input_dim = template_x.shape[1]
        device = template_x.device
        dtype = template_x.dtype
        layout = _instrument_layout(observation_config, inst_name, input_dim)

        target_steps = _find_target_steps(prev_window, inst_name)
        if not target_steps:
            continue

        all_x: List[torch.Tensor] = []
        all_lat: List[torch.Tensor] = []
        all_lon: List[torch.Tensor] = []
        all_pressure_level: List[torch.Tensor] = []
        all_input_times: List[torch.Tensor] = []
        all_mask: List[torch.Tensor] = []
        all_enc_edge_index: List[torch.Tensor] = []
        all_enc_edge_attr: List[torch.Tensor] = []
        node_offset = 0

        template_mask = getattr(template_window[input_node_type], "input_channel_mask", None)
        n_mask_ch = template_mask.shape[1] if template_mask is not None and template_mask.ndim == 2 else None

        for step_idx, step_nt in target_steps:
            if step_idx >= len(pred_list):
                continue
            step_pred = pred_list[step_idx]
            if step_pred is None or step_pred.numel() == 0:
                continue

            step_pred = step_pred.to(device=device, dtype=dtype)
            N_step = step_pred.shape[0]
            tgt = prev_window[step_nt]

            n_tgt = int(tgt.y.shape[0]) if hasattr(tgt, "y") and tgt.y is not None else -1
            if n_tgt != N_step or not hasattr(tgt, "lat"):
                raise RuntimeError(
                    f"{step_nt}: pred rows={N_step} target rows={n_tgt} "
                    f"has_lat={hasattr(tgt, 'lat')} n_preds={len(pred_list)} "
                    f"steps={[s for s, _ in target_steps]}"
                )

            # Channel validity of each prediction row, from the matching target node
            tmask = getattr(tgt, "target_channel_mask", None)
            if tmask is not None and tmask.ndim == 2 and tmask.shape == (N_step, layout["n_ch"]):
                chan_mask = tmask.to(device=device, dtype=torch.bool)
            else:
                print(f"[MASKDIAG] {step_nt}: fallback all-valid mask "
                      f"(tmask={None if tmask is None else tuple(tmask.shape)}, N_step={N_step})", flush=True)
                chan_mask = torch.ones(N_step, layout["n_ch"], dtype=torch.bool, device=device)

            all_x.append(_build_input_rows(step_pred, chan_mask, tgt, template_x, layout, inst_name))

            if hasattr(tgt, "lat") and tgt.lat is not None and tgt.lat.numel() > 0:
                all_lat.append(tgt.lat[:N_step].to(device=device, dtype=torch.float32))
            if hasattr(tgt, "lon") and tgt.lon is not None and tgt.lon.numel() > 0:
                all_lon.append(tgt.lon[:N_step].to(device=device, dtype=torch.float32))
            if hasattr(tgt, "pressure_level") and tgt.pressure_level is not None and tgt.pressure_level.numel() > 0:
                all_pressure_level.append(tgt.pressure_level[:N_step].to(device=device))
#            if hasattr(tgt, "target_times") and tgt.target_times is not None and tgt.target_times.numel() > 0:
#                all_input_times.append(tgt.target_times[:N_step].to(device=device))
            if hasattr(tgt, "obs_time_unix") and tgt.obs_time_unix is not None and tgt.obs_time_unix.numel() > 0:
                all_input_times.append(tgt.obs_time_unix[:N_step].to(device=device))

            if n_mask_ch is not None:
                all_mask.append(chan_mask)

            dec_etype = ("mesh", "to", step_nt)
            if dec_etype in prev_window.edge_types:
                dec = prev_window[dec_etype]
                if hasattr(dec, "edge_index") and dec.edge_index is not None and dec.edge_index.numel() > 0:
                    ei = dec.edge_index.to(device=device)  # [2,E]: row0=mesh, row1=target
                    enc_ei = torch.stack([ei[1] + node_offset, ei[0]], dim=0)
                    all_enc_edge_index.append(enc_ei)
                    if hasattr(dec, "edge_attr") and dec.edge_attr is not None and dec.edge_attr.numel() > 0:
                        all_enc_edge_attr.append(dec.edge_attr.to(device=device, dtype=dtype))

            node_offset += N_step

        if not all_x:
            continue

        synthetic[input_node_type].x = torch.cat(all_x, dim=0)
        if all_mask:
            synthetic[input_node_type].input_channel_mask = torch.cat(all_mask, dim=0)
            _nx = synthetic[input_node_type].x.shape[0]
            _nm = synthetic[input_node_type].input_channel_mask.shape[0]
            _tm = template_mask.shape[0] if template_mask is not None else -1
            if input_node_type not in _MASK_DIAG_DONE or _nx != _nm:
                _MASK_DIAG_DONE.add(input_node_type)
                print(f"[MASKCHECK] {input_node_type}: synth x rows={_nx} synth mask rows={_nm} "
                      f"template mask rows={_tm} -> {'OK' if _nx == _nm else 'MISMATCH'}", flush=True)
        if input_node_type not in _ALIGN_DIAG_DONE:   # once per instrument per process
            _ALIGN_DIAG_DONE.add(input_node_type)
            print(f"[ALIGNDIAG] {input_node_type} layout={layout}", flush=True)
            print(f"[ALIGNDIAG] real  x[0] = {template_x[0].detach().float().cpu().numpy().round(3)}", flush=True)
            print(f"[ALIGNDIAG] synth x[0] = {synthetic[input_node_type].x[0].detach().float().cpu().numpy().round(3)}", flush=True)
        if all_lat:
            synthetic[input_node_type].lat = torch.cat(all_lat, dim=0)
        if all_lon:
            synthetic[input_node_type].lon = torch.cat(all_lon, dim=0)
        if all_pressure_level:
            synthetic[input_node_type].pressure_level = torch.cat(all_pressure_level, dim=0)
        if all_input_times:
            synthetic[input_node_type].input_times = torch.cat(all_input_times, dim=0)
        if all_mask:
            synthetic[input_node_type].input_channel_mask = torch.cat(all_mask, dim=0)

        if layout["k"] and input_node_type not in _SATCOUNT_DONE:
            _SATCOUNT_DONE.add(input_node_type)
            _s0, _k = layout["sat0"], layout["k"]
            print(f"[SATCOUNT] {input_node_type} real={template_x[:, _s0:_s0+_k].sum(0).int().tolist()} "
                  f"synth={synthetic[input_node_type].x[:, _s0:_s0+_k].sum(0).int().tolist()}", flush=True)

        # ---- [MATCHDIAG] once per instrument: synth row 0 vs nearest real input row ----
        if input_node_type not in _MATCH_DIAG_DONE and all_lat and all_lon:
            _MATCH_DIAG_DONE.add(input_node_type)
            _fmt = lambda v: np.array2string(v.detach().float().cpu().numpy(), precision=3,
                                             suppress_small=True, max_line_width=100000)
            _rlat = template_window[input_node_type].lat.float()
            _rlon = template_window[input_node_type].lon.float()
            _slat = synthetic[input_node_type].lat[0].float()
            _slon = synthetic[input_node_type].lon[0].float()
            _j = int(((_rlat - _slat) ** 2 + (_rlon - _slon) ** 2).argmin())
            print(f"[MATCHDIAG] {input_node_type} synth[0] lat/lon=({float(_slat):.3f},{float(_slon):.3f}) "
                  f"nearest real[{_j}] lat/lon=({float(_rlat[_j]):.3f},{float(_rlon[_j]):.3f})", flush=True)
            print(f"[MATCHDIAG] {input_node_type} real  x[{_j}] = {_fmt(template_x[_j])}", flush=True)
            print(f"[MATCHDIAG] {input_node_type} synth x[0]  = {_fmt(synthetic[input_node_type].x[0])}", flush=True)


        enc_etype = (input_node_type, "to", "mesh")
        if all_enc_edge_index:
            synthetic[enc_etype].edge_index = torch.cat(all_enc_edge_index, dim=1)
            if all_enc_edge_attr:
                synthetic[enc_etype].edge_attr = torch.cat(all_enc_edge_attr, dim=0)

    if recompute_target_metadata:
        _recompute_target_metadata_times(
            synthetic,
            ar_offset_hours=ar_offset_hours,
            latent_step_hours=latent_step_hours,
            init_time_unix=init_time_unix,
            target_time_feature_dim=target_time_feature_dim,
            verbose=verbose,
        )

    return synthetic
