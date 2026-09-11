#!/usr/bin/env python
"""Synthetic common-support target diagnostic for OCELOT FSOI.

This diagnostic tests whether cross-target ATMS sign differences persist after
forcing radiosonde and aircraft verification targets onto the same locations,
times, pressure levels, and verifying values for the variables they share.

The default experiment uses radiosonde target rows as the support template and
constructs:

  - a radiosonde-target loss on the selected rows, and
  - an aircraft-target loss on the same rows.

Only variables present in both target definitions are compared: temperature,
u wind, and v wind. Dewpoint is excluded because the aircraft decoder does not
predict dewpoint.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from torch_geometric.loader import DataLoader as PyGDataLoader

_FSOI = Path(__file__).resolve().parent
_GNN_MODEL = _FSOI.parent
if str(_GNN_MODEL) not in sys.path:
    sys.path.insert(0, str(_GNN_MODEL))
if str(_FSOI) not in sys.path:
    sys.path.insert(0, str(_FSOI))

from create_mesh_graph_global import obs_mesh_conn  # noqa: E402
from fsoi_dataset import FSOIDataset, verify_sequential_consistency  # noqa: E402
from fsoi_inference import (  # noqa: E402
    configure_deterministic_inference,
    find_checkpoint,
    load_fsoi_config,
    _as_scalar_bin,
)
from fsoi_model_extensions import freeze_model_for_fsoi, predict_at_targets  # noqa: E402
from fsoi_utils import (  # noqa: E402
    STANDARD_PRESSURE_LEVELS,
    compute_per_level_fsoi_by_variable,
    get_fsoi_input_masks,
    get_fsoi_inputs,
)
from gnn_datamodule import BinDataset, GNNDataModule  # noqa: E402
from gnn_model import GNNLightning  # noqa: E402
from process_timeseries import organize_bins_times  # noqa: E402
from weight_utils import load_weights_from_yaml  # noqa: E402


COMMON_VARIABLES = ("temperature", "u_wind", "v_wind")
TARGET_CHANNELS = {
    "radiosonde": {"temperature": 0, "dewpoint_temperature": 1, "u_wind": 2, "v_wind": 3},
    "aircraft": {"temperature": 0, "u_wind": 1, "v_wind": 2},
}


def _parse_csv_list(text: str, cast=str) -> list:
    return [cast(item.strip()) for item in str(text).split(",") if item.strip()]


def _pressure_indices(levels_hpa: Iterable[float]) -> set[int]:
    indices: set[int] = set()
    for level in levels_hpa:
        matches = np.where(np.isclose(STANDARD_PRESSURE_LEVELS, float(level), atol=1.0))[0]
        if matches.size:
            indices.add(int(matches[0]))
    if not indices:
        raise ValueError(f"No requested pressure levels matched {STANDARD_PRESSURE_LEVELS.tolist()}")
    return indices


def _slice_attr(value, keep: torch.Tensor):
    if torch.is_tensor(value) and value.shape[:1] == keep.shape[:1]:
        return value[keep].clone()
    return value


def _coerce_2d_width(value: torch.Tensor, width: int, *, fill: float = 0.0) -> torch.Tensor:
    value = value.clone()
    if value.dim() == 1:
        value = value.view(-1, 1)
    if value.shape[1] == width:
        return value
    if value.shape[1] > width:
        return value[:, :width].clone()
    pad = torch.full(
        (value.shape[0], width - value.shape[1]),
        fill,
        dtype=value.dtype,
        device=value.device,
    )
    return torch.cat([value, pad], dim=1)


def _target_metadata_width(batch, target_node: str, fallback: int) -> int:
    if target_node in batch.node_types and hasattr(batch[target_node], "target_metadata"):
        meta = batch[target_node].target_metadata
        if torch.is_tensor(meta) and meta.dim() == 2 and meta.shape[1] > 0:
            return int(meta.shape[1])
    return fallback


def _target_x_width(batch, target_node: str, observation_config: dict, target_inst: str) -> int:
    if target_node in batch.node_types and hasattr(batch[target_node], "x"):
        x = batch[target_node].x
        if torch.is_tensor(x) and x.dim() == 2 and x.shape[1] > 0:
            return int(x.shape[1])
    for obs_type, insts in observation_config.items():
        if target_inst in insts:
            return int(insts[target_inst].get("scan_angle_channels", 1) or 1)
    return 1


def build_synthetic_common_target(
    batch,
    *,
    model,
    observation_config: dict,
    template_inst: str,
    target_inst: str,
    lead_step: int,
    pressure_levels_hpa: list[float],
    variables: list[str],
    max_target_rows: int | None,
    seed: int,
) -> tuple[int, list[int]]:
    """Replace one target node with template rows on common support."""
    template_node = f"{template_inst}_target_step{lead_step}"
    target_node = f"{target_inst}_target_step{lead_step}"
    if template_node not in batch.node_types:
        raise ValueError(f"Template target node {template_node!r} is not present")
    if target_node not in batch.node_types:
        raise ValueError(f"Target node {target_node!r} is not present")
    if target_inst not in TARGET_CHANNELS:
        raise ValueError(f"No channel map is defined for target {target_inst!r}")

    template = batch[template_node]
    if not hasattr(template, "y") or template.y is None or template.y.numel() == 0:
        raise ValueError(f"Template target node {template_node!r} has no y values")
    if not hasattr(template, "pressure_level"):
        raise ValueError(f"Template target node {template_node!r} has no pressure_level")

    valid_vars = [v for v in variables if v in TARGET_CHANNELS[template_inst] and v in TARGET_CHANNELS[target_inst]]
    if not valid_vars:
        raise ValueError(
            f"No requested variables are common to {template_inst} and {target_inst}: {variables}"
        )

    pidx = template.pressure_level
    if pidx.dim() > 1:
        pidx = pidx.squeeze(1)
    requested_idx = _pressure_indices(pressure_levels_hpa)
    keep = torch.zeros_like(pidx, dtype=torch.bool)
    for idx in requested_idx:
        keep |= pidx.long().eq(int(idx))

    if hasattr(template, "target_channel_mask"):
        mask = template.target_channel_mask.to(torch.bool)
        for var in valid_vars:
            keep &= mask[:, TARGET_CHANNELS[template_inst][var]]

    n_available = int(keep.sum().item())
    if n_available == 0:
        raise ValueError(
            f"No {template_inst} target rows remain for levels {pressure_levels_hpa} "
            f"and variables {valid_vars}"
        )

    selected_positions = torch.nonzero(keep, as_tuple=False).view(-1)
    if max_target_rows is not None and n_available > int(max_target_rows):
        rng = np.random.default_rng(seed)
        take = np.sort(rng.choice(n_available, size=int(max_target_rows), replace=False))
        selected_positions = selected_positions[torch.as_tensor(take, dtype=torch.long, device=selected_positions.device)]
        keep = torch.zeros_like(keep, dtype=torch.bool)
        keep[selected_positions] = True

    n = int(keep.sum().item())
    target_dim = len(TARGET_CHANNELS[target_inst])
    y = torch.zeros((n, target_dim), dtype=template.y.dtype, device=template.y.device)
    target_mask = torch.zeros((n, target_dim), dtype=torch.bool, device=template.y.device)

    for var in valid_vars:
        src_ch = TARGET_CHANNELS[template_inst][var]
        dst_ch = TARGET_CHANNELS[target_inst][var]
        y[:, dst_ch] = template.y[keep, src_ch]
        target_mask[:, dst_ch] = True

    store = batch[target_node]
    store.y = y.float()
    store.target_channel_mask = target_mask
    store.pressure_level = pidx[keep].long().clone()

    if hasattr(template, "target_pressure_hpa") and template.target_pressure_hpa.numel() > 0:
        store.target_pressure_hpa = template.target_pressure_hpa[keep].float().clone()
    else:
        p_hpa = torch.as_tensor(
            [STANDARD_PRESSURE_LEVELS[int(i)] for i in store.pressure_level.detach().cpu().numpy()],
            dtype=torch.float32,
            device=template.y.device,
        )
        store.target_pressure_hpa = p_hpa

    if hasattr(template, "lat") and hasattr(template, "lon"):
        store.lat = template.lat[keep].float().clone()
        store.lon = template.lon[keep].float().clone()
    else:
        raise ValueError(f"Template node {template_node!r} does not have lat/lon")

    if hasattr(template, "obs_time_unix"):
        store.obs_time_unix = template.obs_time_unix[keep].long().clone()

    meta_width = _target_metadata_width(batch, target_node, fallback=0)
    if hasattr(template, "target_metadata") and template.target_metadata.numel() > 0:
        meta = template.target_metadata[keep].float().clone()
        if meta_width > 0:
            meta = _coerce_2d_width(meta, meta_width)
        store.target_metadata = meta

    x_width = _target_x_width(batch, target_node, observation_config, target_inst)
    store.x = torch.zeros((n, x_width), dtype=torch.float32, device=template.y.device)

    inst_id = model.instrument_name_to_id.get(target_inst)
    if inst_id is not None:
        store.instrument_ids = torch.full((n,), int(inst_id), dtype=torch.long, device=template.y.device)

    # PyG batch attributes may be present because the dataloader wraps a
    # single HeteroData object. Keep them consistent with the new row count.
    if "batch" in store:
        store.batch = torch.zeros((n,), dtype=torch.long, device=template.y.device)
    if "ptr" in store:
        store.ptr = torch.tensor([0, n], dtype=torch.long, device=template.y.device)
    store.num_nodes = n

    lat_np = store.lat.detach().cpu().numpy()
    lon_np = store.lon.detach().cpu().numpy()
    edge_index, edge_attr = obs_mesh_conn(
        lat_np,
        lon_np,
        model.mesh_structure["m2m_graphs"],
        model.mesh_structure["mesh_lat_lon_list"],
        model.mesh_structure["mesh_list"],
        o2m=False,
    )
    batch["mesh", "to", target_node].edge_index = edge_index
    batch["mesh", "to", target_node].edge_attr = edge_attr.to(torch.float16)

    return n, sorted(requested_idx)


def _make_loader(
    *,
    data_path: str,
    start_date: str,
    end_date: str,
    observation_config: dict,
    feature_stats: dict,
    model,
) -> PyGDataLoader:
    datamodule = GNNDataModule(
        data_path=data_path,
        start_date=start_date,
        end_date=end_date,
        observation_config=observation_config,
        mesh_structure=model.mesh_structure,
        batch_size=1,
        feature_stats=feature_stats,
        num_neighbors=3,
        window_size="12h",
        pipeline=None,
    )
    datamodule.setup(stage="test")
    fsoi_summary = organize_bins_times(
        datamodule.z,
        start_date,
        end_date,
        observation_config,
        pipeline_cfg={},
        window_size="12h",
    )
    bin_names = sorted(fsoi_summary.keys())
    verify_sequential_consistency(bin_names, expected_interval_hours=12)

    def create_graph_fn(bin_data):
        return datamodule._create_graph_structure(bin_data)

    base_dataset = BinDataset(
        bin_names=bin_names,
        data_summary=fsoi_summary,
        zarr_store=datamodule.z,
        create_graph_fn=create_graph_fn,
        observation_config=observation_config,
        feature_stats=feature_stats,
        include_persistence_inputs=True,
        tag="COMMON_SUPPORT",
    )
    fsoi_dataset = FSOIDataset(base_dataset=base_dataset, bin_names=bin_names)
    return PyGDataLoader(fsoi_dataset, batch_size=1, shuffle=False, num_workers=0)


def _summarize_atms_channel(results: list[dict], channel: int) -> dict:
    vals = []
    for result in results:
        tensor = result["fsoi_values"].get("atms")
        if tensor is None or tensor.numel() == 0 or tensor.shape[1] < channel:
            continue
        vals.append(tensor[:, channel - 1].detach().cpu().float().reshape(-1))
    if not vals:
        return {"sum_fsoi": np.nan, "mean_fsoi": np.nan, "n_source_values": 0}
    all_vals = torch.cat(vals)
    return {
        "sum_fsoi": float(all_vals.sum().item()),
        "mean_fsoi": float(all_vals.mean().item()),
        "median_fsoi": float(all_vals.median().item()),
        "n_source_values": int(all_vals.numel()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="OCELOT checkpoint path")
    parser.add_argument("--config", default="FSOI/configs/fsoi_config_radiosonde_all.yaml")
    parser.add_argument("--obs-config", default="configs/observation_config.yaml")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default="FSOI/fsoi_outputs/common_support_atms")
    parser.add_argument("--start-date", default="2025-07-01")
    parser.add_argument("--end-date", default="2025-07-03")
    parser.add_argument("--pair-indices", default="0", help="Comma-separated pair indices")
    parser.add_argument("--source-instrument", default="atms")
    parser.add_argument("--source-channel", type=int, default=1, help="1-based channel index")
    parser.add_argument("--template-target", default="radiosonde")
    parser.add_argument("--compare-targets", default="radiosonde,aircraft")
    parser.add_argument("--variables", default="u_wind,v_wind")
    parser.add_argument("--pressure-levels", default="1000,925,850")
    parser.add_argument("--max-source-nodes", type=int, default=50000)
    parser.add_argument("--max-target-rows", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20250911)
    args = parser.parse_args()

    checkpoint = find_checkpoint(args.checkpoint)
    fsoi_config = load_fsoi_config(args.config)
    observation_config, feature_stats, instrument_weights, channel_weights, name_to_id = load_weights_from_yaml(args.obs_config)

    forecast_cfg = fsoi_config.get("forecast", {})
    use_area_weights = bool(forecast_cfg.get("use_area_weights", True))
    loss_reduction = str(forecast_cfg.get("loss_reduction", "mean"))
    impact_factor = float(forecast_cfg.get("impact_factor", 0.5))
    lead_step = int((forecast_cfg.get("lead_steps") or [0])[0])

    if args.source_instrument != "atms":
        raise ValueError("This diagnostic currently reports ATMS channel impacts only")
    if impact_factor != 0.5:
        raise ValueError("Common-support diagnostic requires impact_factor=0.5")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    configure_deterministic_inference(args.seed)

    print(f"Loading checkpoint: {checkpoint}")
    model = GNNLightning.load_from_checkpoint(checkpoint)
    model.to(device)
    model.eval()
    freeze_model_for_fsoi(model)
    model.weights_name_to_id = name_to_id

    data_path = args.data_path or "/scratch4/NAGAPE/gpu-ai4wp/Ronald.McLaren/ocelot/data/v7"
    loader = _make_loader(
        data_path=data_path,
        start_date=args.start_date,
        end_date=args.end_date,
        observation_config=observation_config,
        feature_stats=feature_stats,
        model=model,
    )

    output_dir = Path(args.output_dir)
    (output_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    with open(output_dir / "logs" / "common_support_config_used.yaml", "w") as f:
        yaml.safe_dump(vars(args), f, sort_keys=True)

    wanted_pairs = set(_parse_csv_list(args.pair_indices, int))
    variables = _parse_csv_list(args.variables, str)
    pressure_levels = _parse_csv_list(args.pressure_levels, float)
    compare_targets = _parse_csv_list(args.compare_targets, str)

    rows: list[dict] = []
    for pair_idx, (prev_batch, curr_batch) in enumerate(loader):
        if pair_idx not in wanted_pairs:
            continue

        print(f"\n=== Common-support diagnostic pair {pair_idx} ===")
        prev_bin = _as_scalar_bin(prev_batch.bin_name)
        curr_bin = _as_scalar_bin(curr_batch.bin_name)
        print(f"Previous: {prev_bin}  Current: {curr_bin}")

        prev_dev = prev_batch.to(device)
        curr_dev = curr_batch.to(device)

        xa_all = get_fsoi_inputs(
            curr_dev,
            observation_config,
            model.instrument_name_to_id,
            match_targets=False,
        )
        if args.source_instrument not in xa_all:
            print(f"[SKIP] {args.source_instrument} not present in pair {pair_idx}")
            continue
        xa = {args.source_instrument: xa_all[args.source_instrument]}

        xb_raw, subsample_indices = predict_at_targets(
            model,
            prev_dev,
            curr_dev,
            observation_config,
            forecast_step=lead_step,
            keep_instruments=[args.source_instrument],
            max_decoder_nodes={args.source_instrument: args.max_source_nodes},
        )
        if args.source_instrument not in xb_raw:
            print(f"[SKIP] no background prediction for {args.source_instrument} in pair {pair_idx}")
            continue

        idx = subsample_indices.get(args.source_instrument)
        if idx is not None:
            xa[args.source_instrument] = xa[args.source_instrument][idx].clone().detach().requires_grad_(True)
        xb = {
            args.source_instrument: xb_raw[args.source_instrument].clone().detach().requires_grad_(True)
        }
        valid_masks = get_fsoi_input_masks(
            curr_dev,
            observation_config,
            replace_indices=subsample_indices,
            device=device,
        )
        valid_masks = {
            args.source_instrument: valid_masks[args.source_instrument],
        }

        for target_inst in compare_targets:
            synthetic_batch = curr_batch.clone()
            n_target_rows, pressure_indices = build_synthetic_common_target(
                synthetic_batch,
                model=model,
                observation_config=observation_config,
                template_inst=args.template_target,
                target_inst=target_inst,
                lead_step=lead_step,
                pressure_levels_hpa=pressure_levels,
                variables=variables,
                max_target_rows=args.max_target_rows,
                seed=args.seed + pair_idx,
            )
            synthetic_batch = synthetic_batch.to(device)
            results = compute_per_level_fsoi_by_variable(
                model=model,
                curr_batch=synthetic_batch,
                xa=xa,
                xb=xb,
                observation_config=observation_config,
                forecast_lead_step=lead_step,
                instrument_weights=instrument_weights,
                channel_weights=channel_weights,
                use_area_weights=use_area_weights,
                target_instruments=[target_inst],
                requested_target_variables=variables,
                target_pressure_levels=pressure_levels,
                loss_reduction=loss_reduction,
                impact_factor=impact_factor,
                replace_indices=subsample_indices,
                valid_masks=valid_masks,
            )

            for result in results:
                summary = _summarize_atms_channel([result], args.source_channel)
                rows.append(
                    {
                        "pair_idx": pair_idx,
                        "prev_bin": prev_bin,
                        "curr_bin": curr_bin,
                        "template_target": args.template_target,
                        "synthetic_target": target_inst,
                        "target_variable": result.get("target_variable"),
                        "p_idx": result.get("p_idx"),
                        "p_hpa": result.get("p_hpa"),
                        "source_instrument": args.source_instrument,
                        "source_channel": args.source_channel,
                        "sum_fsoi": summary["sum_fsoi"],
                        "mean_fsoi": summary["mean_fsoi"],
                        "median_fsoi": summary["median_fsoi"],
                        "n_source_values": summary["n_source_values"],
                        "n_target_rows": n_target_rows,
                        "pressure_indices": ",".join(str(v) for v in pressure_indices),
                        "ea": result.get("ea_p"),
                        "eb": result.get("eb_p"),
                        "loss_reduction": loss_reduction,
                        "use_area_weights": use_area_weights,
                        "synthetic_support_note": (
                            "same template target rows, lat/lon, time, pressure levels, "
                            "and verifying values for common variables"
                        ),
                    }
                )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    detail_path = output_dir / "evaluation" / "common_support_atms_channel_detail.csv"
    df.to_csv(detail_path, index=False)
    print(f"\nSaved detail CSV: {detail_path}")

    if not df.empty:
        group_cols = ["synthetic_target", "target_variable", "p_hpa", "source_instrument", "source_channel"]
        summary = (
            df.groupby(group_cols, dropna=False)
            .agg(
                n_pairs=("pair_idx", "nunique"),
                total_sum_fsoi=("sum_fsoi", "sum"),
                median_sum_fsoi=("sum_fsoi", "median"),
                mean_sum_fsoi=("sum_fsoi", "mean"),
                sign_positive_fraction=("sum_fsoi", lambda s: float((s > 0).mean())),
                n_target_rows=("n_target_rows", "median"),
                n_source_values=("n_source_values", "median"),
            )
            .reset_index()
        )
        summary_path = output_dir / "evaluation" / "common_support_atms_channel_summary.csv"
        summary.to_csv(summary_path, index=False)
        print(f"Saved summary CSV: {summary_path}")
        print(summary.to_string(index=False))
    else:
        print("[WARNING] No common-support rows were produced.")


if __name__ == "__main__":
    main()
