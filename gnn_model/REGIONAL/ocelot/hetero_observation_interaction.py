"""
Implementation of a graph neural network model for processing heterogeneous observations
with different variable types and spatial locations.
"""

import logging

import torch
import torch.nn as nn
from typing import Dict, List, Set, Tuple
from torch_geometric.data import HeteroData
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GATConv

logger = logging.getLogger(__name__)

from ocelot.interaction_net import InteractionNet
from ocelot.hierarchical_processor import HierarchicalProcessor
from ocelot.processor_transformer_hierarchical import (
    HierarchicalSlidingWindowTransformer,
)
from ocelot.base_hetero_observation_model import BaseHeteroGraphModel, _validate_graph
from ocelot.constants import FEATURES_AUX_DIM, STATIC_CONTEXT_DIM
from ocelot.processor import Processor
from ocelot import utils
from ocelot.graph_utils import (
    ENCODE_OBS_TO_MESH_REL,
    build_feedback_input_x_from_target,
    target_node_type,
)


def _mem(label: str) -> None:
    """Print current GPU memory allocation with a label."""
    if not torch.cuda.is_available():
        return
    alloc = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    logger.debug("[MEM] %s: allocated=%.2f GiB  reserved=%.2f GiB", label, alloc, reserved)


def _tensor_mb(t: torch.Tensor) -> float:
    return t.element_size() * t.nelement() / 1024**2


class BipartiteGAT(nn.Module):
    """
    GAT layer for bipartite graphs (send -> rec). Same interface as InteractionNet:
    forward(send_rep, rec_rep, edge_rep, edge_index) -> updated rec_rep.
    edge_rep is incorporated as edge features in the attention computation when edge_dim > 0.
    """

    def __init__(
        self,
        send_dim: int,
        rec_dim: int,
        hidden_dim: int,
        num_heads: int = 1,
        edge_dim: int = 0,
        gat_chunk_size: int = 0,
        **kwargs,
    ):
        super().__init__()
        self.rec_dim = rec_dim
        self.num_heads = num_heads
        self.gat_chunk_size = gat_chunk_size
        out_per_head = hidden_dim // num_heads
        if out_per_head * num_heads != hidden_dim:
            out_per_head = hidden_dim
            num_heads = 1
        self._num_heads = num_heads
        self._out_per_head = out_per_head
        self.gat = GATConv(
            (send_dim, rec_dim),
            out_per_head,
            heads=num_heads,
            concat=(num_heads > 1),
            add_self_loops=False,  # bipartite: no self-edges in graph
            dropout=0.0,
            edge_dim=edge_dim if edge_dim > 0 else None,
        )
        if num_heads > 1:
            self.proj = nn.Linear(out_per_head * num_heads, rec_dim)
        else:
            self.proj = nn.Linear(out_per_head, rec_dim)
        self.norm = nn.LayerNorm(rec_dim)

    def forward(self, send_rep, rec_rep, edge_rep, edge_index):
        size = (send_rep.size(0), rec_rep.size(0))
        if self.gat_chunk_size > 0 and rec_rep.size(0) > self.gat_chunk_size:
            return self._forward_chunked(send_rep, rec_rep, edge_rep, edge_index)
        out = self.gat((send_rep, rec_rep), edge_index, edge_attr=edge_rep, size=size)
        out = self.proj(out)
        return self.norm(rec_rep + out)

    def _forward_chunked(self, send_rep, rec_rep, edge_rep, edge_index):
        """Process receiver nodes in chunks to reduce peak GPU memory.

        proj and norm are applied per-chunk so the only full-mesh allocation is
        the pre-allocated output tensor, avoiding a second mesh-sized intermediate
        that would otherwise be live alongside rec_rep during backward recomputation.
        """
        n_rec = rec_rep.size(0)
        chunk_size = self.gat_chunk_size
        # Pre-allocate at rec_dim; proj+norm applied inside the loop
        out = torch.empty(n_rec, self.rec_dim, device=rec_rep.device, dtype=send_rep.dtype)
        for start in range(0, n_rec, chunk_size):
            end = min(start + chunk_size, n_rec)
            mask = (edge_index[1] >= start) & (edge_index[1] < end)
            chunk_ei = edge_index[:, mask].clone()
            chunk_ei[1] = chunk_ei[1] - start
            chunk_ea = edge_rep[mask] if edge_rep is not None else None
            chunk_rec = rec_rep[start:end]
            chunk_size_tuple = (send_rep.size(0), end - start)
            chunk_out = self.gat(
                (send_rep, chunk_rec), chunk_ei,
                edge_attr=chunk_ea, size=chunk_size_tuple,
            )
            chunk_out = self.proj(chunk_out)
            out[start:end] = self.norm(chunk_rec + chunk_out)
        return out


class HeteroObservationGraphModel(BaseHeteroGraphModel): # Or pl.LightningModule if not inheriting ARDOPModel's PL logic

    """
    Graph neural network model for processing heterogeneous observations.
    Handles observations of different types (temperature, wind, pressure, etc.)
    at different spatial locations.

    Workflow:
    1. Multiple observation types -> type-specific embeddings
    2. Map to mesh via spatial graph connections
    3. Process on mesh
    4. Map back to observation space
    """

    def __init__(self, args):
        super().__init__(args)
        # Initialize network dictionaries
        self.processor = None
        self._instrument_order = []
        self._num_target_bins = max(1, int(getattr(args, 'num_target_bins', 1)))
        # Note: scan_angle_embedders, observation_embedders, output_mappers 
        # are already initialized in BaseHeteroGraphModel

        # Set up observation networks if config provided
        if hasattr(args, 'observation_config') and args.observation_config:
            self.setup_observation_networks(args.observation_config)

    def setup_observation_networks(self, observation_config: Dict):
        """
        Creates encoding/decoding networks and initializes the GNN Processor.

        Architecture (GraphCast-style):
        - Encoder: obs_input -> mesh (observation_encoders)
        - Processor: mesh <-> mesh only (refines latent mesh state)
        - Decoder: mesh -> obs_target (observation_decoders + output_mappers)
        """
        # Fixed order for encoder aggregation (deterministic, independent of batch)
        self._encoder_edge_order = []
        k = max(1, int(getattr(self.args, 'num_target_bins', 1)))
        self._num_target_bins = k
        self._instrument_order: List[Tuple[str, str]] = []
        use_gat = getattr(self.args, 'use_gat_encoder_decoder', False)
        num_heads = getattr(self.args, 'num_heads', 1)
        self.scan_angle_embed_dim = getattr(self.args, 'scan_angle_embed_dim', 8)
        gat_chunk_size = int(getattr(self.args, 'gat_chunk_size', 0))

        for obs_type, instruments in observation_config.items():
            for inst_name, config in instruments.items():
                node_type_input = f"{obs_type}_{inst_name}_input"

                input_dim = config.get("input_dim")
                target_dim = config.get("target_dim")
                if input_dim is None or target_dim is None:
                    raise ValueError(f"input_dim or target_dim not specified for {obs_type}.{inst_name}")

                self._instrument_order.append((obs_type, inst_name))

                # Graph-based encoder for input nodes (e.g., obs -> mesh)
                edge_type_tuple = (node_type_input, 'to', 'mesh')
                self._encoder_edge_order.append(edge_type_tuple)
                if use_gat:
                    self.observation_encoders[self._edge_key(edge_type_tuple)] = BipartiteGAT(
                        send_dim=self.hidden_dim,
                        rec_dim=self.hidden_dim,
                        hidden_dim=self.hidden_dim,
                        num_heads=num_heads,
                        edge_dim=4,
                        gat_chunk_size=gat_chunk_size,
                    )
                else:
                    self.observation_encoders[self._edge_key(edge_type_tuple)] = InteractionNet(
                        send_dim=self.hidden_dim,
                        rec_dim=self.hidden_dim,
                        edge_dim=4,
                        hidden_layers=self.args.hidden_layers,
                    )

                # We still need an initial MLP to project raw features to hidden_dim
                self.observation_embedders[node_type_input] = utils.make_mlp(
                    [input_dim] + self.mlp_blueprint_end
                )

                meta_keys = config.get('metadata', [])
                target_context_dim = len(meta_keys) + FEATURES_AUX_DIM
                if obs_type == "state":
                    # DARegionalGraphDataset appends static terrain/slmask context
                    # to the state/anal decoder target node features.
                    target_context_dim += STATIC_CONTEXT_DIM
                input_dim_for_mapper = self.hidden_dim + self.scan_angle_embed_dim
                output_map_layers = (
                    [input_dim_for_mapper] + [self.hidden_dim] * self.args.hidden_layers + [target_dim]
                )

                for h in range(k):
                    node_type_target = target_node_type(obs_type, inst_name, h, k)

                    # GNN-based decoder for target nodes (e.g., mesh -> obs_target)
                    edge_type_tuple = ('mesh', 'to', node_type_target)
                    if use_gat:
                        self.observation_decoders[self._edge_key(edge_type_tuple)] = BipartiteGAT(
                            send_dim=self.hidden_dim,
                            rec_dim=self.hidden_dim,
                            hidden_dim=self.hidden_dim,
                            num_heads=num_heads,
                            edge_dim=4,
                            gat_chunk_size=gat_chunk_size,
                        )
                    else:
                        self.observation_decoders[self._edge_key(edge_type_tuple)] = InteractionNet(
                            send_dim=self.hidden_dim,
                            rec_dim=self.hidden_dim,
                            edge_dim=4,
                            hidden_layers=self.args.hidden_layers,
                        )

                    self.scan_angle_embedders[node_type_target] = utils.make_mlp(
                        [target_context_dim, self.scan_angle_embed_dim]
                    )
                    self.output_mappers[node_type_target] = utils.make_mlp(
                        output_map_layers, layer_norm=False
                    )

        scatter_chunk_size = getattr(self.args, 'scatter_chunk_size', 50000)
        hier_kind = getattr(self.args, 'hierarchical_processor_type', 'interaction')
        if getattr(self.args, 'hierarchical', False):
            if hier_kind not in ('interaction', 'transformer'):
                raise ValueError(
                    "hierarchical_processor_type must be 'interaction' or 'transformer' "
                    f"(got {hier_kind!r})"
                )
            self._hierarchical_processor_type = hier_kind
            if hier_kind == 'interaction':
                self.processor = HierarchicalProcessor(
                    hidden_dim=self.hidden_dim,
                    num_levels=self.num_mesh_levels,
                    num_message_passing_steps=self.args.mesh_gnn_layers,
                    hidden_layers=self.args.hidden_layers,
                )
            else:
                proc_heads = int(getattr(self.args, 'processor_heads', 4))
                if self.hidden_dim % proc_heads != 0:
                    raise ValueError(
                        f"hidden_dim ({self.hidden_dim}) must be divisible by "
                        f"processor_heads ({proc_heads}) for the hierarchical transformer"
                    )
                self.processor = HierarchicalSlidingWindowTransformer(
                    hidden_dim=self.hidden_dim,
                    num_levels=self.num_mesh_levels,
                    window=int(getattr(self.args, 'processor_window', 4)),
                    depth=int(getattr(self.args, 'processor_depth', 2)),
                    num_heads=proc_heads,
                    dropout=float(getattr(self.args, 'processor_dropout', 0.0)),
                    use_causal_mask=not getattr(
                        self.args, 'no_processor_causal_mask', False
                    ),
                    use_cross_scale=not getattr(
                        self.args, 'no_processor_cross_scale', False
                    ),
                    spatial_mixing_steps=int(
                        getattr(self.args, 'spatial_mixing_steps', 1)
                    ),
                    attn_chunk_size=int(
                        getattr(self.args, 'processor_attn_chunk_size', 2048)
                    ),
                )
            self.coarse_to_fine_norm = nn.LayerNorm(self.hidden_dim)
            self.coarse_to_fine_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
            self.coarse_to_fine_gate = nn.Sequential(
                nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                nn.Sigmoid(),
            )
        else:
            self._hierarchical_processor_type = 'none'
            self.processor = Processor(
                hidden_dim=self.hidden_dim,
                node_types=['mesh'],
                edge_types=[('mesh', 'to', 'mesh')],
                num_message_passing_steps=self.args.mesh_gnn_layers,
                scatter_chunk_size=scatter_chunk_size,
                edge_attr_dim=self.mesh_edge_attr.shape[-1],
            )

    def _use_obs_autoregressive(self, k: int) -> bool:
        """When ``num_target_bins`` > 1, AR forward is on unless ``--no_obs_autoregressive``."""
        if k <= 1:
            return False
        return not getattr(self.args, 'no_obs_autoregressive', False)

    def _obs_ar_teacher_forcing(self) -> bool:
        return not getattr(self.args, 'no_obs_ar_teacher_forcing', False)

    def _allowed_dst_types_at_horizon(self, horizon: int, k: int) -> Set[str]:
        return {
            target_node_type(obs_type, inst_name, horizon, k)
            for obs_type, inst_name in self._instrument_order
        }

    def _num_metadata_for_instrument(self, obs_type: str, inst_name: str) -> int:
        oc = getattr(self.args, 'observation_config', None) or {}
        inst_cfg = oc.get(obs_type, {}).get(inst_name, {})
        return len(inst_cfg.get('metadata', []))

    def _embed_raw_inputs(self, data: HeteroData) -> Dict[str, torch.Tensor]:
        _mem("embed_raw_inputs: start")
        embedded_features: Dict[str, torch.Tensor] = {}
        for node_type, x in data.x_dict.items():
            logger.debug("[MEM]   raw input '%s': shape=%s  %.1f MiB  dtype=%s", node_type, tuple(x.shape), _tensor_mb(x), x.dtype)
            if node_type == 'mesh':
                embedded_features[node_type] = self.mesh_embedder(x)
            elif node_type.endswith("_input"):
                embedded_features[node_type] = self.observation_embedders[node_type](x)
        for node_type, t in embedded_features.items():
            logger.debug("[MEM]   embedded '%s': shape=%s  %.1f MiB", node_type, tuple(t.shape), _tensor_mb(t))
        _mem("embed_raw_inputs: done")
        return embedded_features

    def _run_bipartite_encoders(
        self,
        data: HeteroData,
        encoded_mesh_features: torch.Tensor,
        embedded: Dict[str, torch.Tensor],
        edge_order: List[Tuple],
    ) -> torch.Tensor:
        _mem("encoders: start")
        for i, edge_type in enumerate(edge_order):
            src_type = edge_type[0]
            if edge_type not in data.edge_index_dict or src_type not in embedded:
                continue
            edge_index = data.edge_index_dict[edge_type]
            obs_features = embedded[src_type]
            edge_attr = data[edge_type].edge_attr
            encoder_key = self._edge_key(edge_type)
            if encoder_key not in self.observation_encoders:
                continue
            encoder = self.observation_encoders[encoder_key]

            ea_mb = _tensor_mb(edge_attr) if edge_attr is not None else 0.0
            ei_mb = _tensor_mb(edge_index)
            logger.debug(
                "[MEM] encoder[%d] %s:  obs=%s %.1f MiB  edge_index=%s %.1f MiB"
                "  edge_attr=%s %.1f MiB  (checkpoint saves ~%.0f MiB)",
                i, edge_type, tuple(obs_features.shape), _tensor_mb(obs_features),
                tuple(edge_index.shape), ei_mb,
                tuple(edge_attr.shape) if edge_attr is not None else None, ea_mb,
                _tensor_mb(obs_features) + ei_mb + ea_mb,
            )
            _mem(f"encoder[{i}] before")

            if self.training:
                encoded_mesh_features = checkpoint(
                    encoder,
                    obs_features,
                    encoded_mesh_features,
                    edge_attr,
                    edge_index,
                    use_reentrant=False,
                )
            else:
                encoded_mesh_features = encoder(
                    send_rep=obs_features,
                    rec_rep=encoded_mesh_features,
                    edge_rep=edge_attr,
                    edge_index=edge_index,
                )
            _mem(f"encoder[{i}] after")

        _mem("encoders: done")
        return encoded_mesh_features

    def _build_hierarchical_mesh_lists(
        self,
        data: HeteroData,
        current_mesh_features: torch.Tensor,
    ) -> Tuple[List[torch.Tensor], ...]:
        """Batch per-level mesh features and edges for hierarchical processors."""
        num_graphs = int(data.num_graphs)
        hd = self.hidden_dim
        num_levels = self.num_mesh_levels

        mesh_features_list: List[torch.Tensor] = []
        mesh_edge_index_list: List[torch.Tensor] = []
        mesh_edge_attr_list: List[torch.Tensor] = []

        for level in range(num_levels):
            level_mesh_x = getattr(self, f'mesh_x_level_{level}')
            level_mesh_ei = getattr(self, f'mesh_edge_index_level_{level}')
            level_mesh_ea = getattr(self, f'mesh_edge_attr_level_{level}')

            if level == 0:
                mesh_features_list.append(current_mesh_features)
            else:
                n = level_mesh_x.shape[0]
                mesh_features_list.append(
                    torch.zeros(
                        n * num_graphs,
                        hd,
                        device=current_mesh_features.device,
                        dtype=current_mesh_features.dtype,
                    )
                )

            num_nodes_level = level_mesh_x.shape[0]
            batched_ei = [
                level_mesh_ei + i * num_nodes_level for i in range(num_graphs)
            ]
            mesh_edge_index_list.append(torch.cat(batched_ei, dim=1))
            mesh_edge_attr_list.append(level_mesh_ea.repeat(num_graphs, 1))

        up_edge_index_list: List[torch.Tensor] = []
        up_edge_attr_list: List[torch.Tensor] = []
        down_edge_index_list: List[torch.Tensor] = []
        down_edge_attr_list: List[torch.Tensor] = []

        for level in range(num_levels - 1):
            up_ei = getattr(self, f'mesh_up_edge_index_{level}')
            up_ea = getattr(self, f'mesh_up_edge_attr_{level}')
            down_ei = getattr(self, f'mesh_down_edge_index_{level}')
            down_ea = getattr(self, f'mesh_down_edge_attr_{level}')

            num_nodes_fine = getattr(self, f'mesh_x_level_{level}').shape[0]
            num_nodes_coarse = getattr(self, f'mesh_x_level_{level + 1}').shape[0]
            device = up_ei.device

            batched_up = []
            batched_down = []
            for i in range(num_graphs):
                batched_up.append(
                    up_ei
                    + torch.tensor(
                        [[i * num_nodes_fine], [i * num_nodes_coarse]],
                        device=device,
                        dtype=torch.long,
                    )
                )
                batched_down.append(
                    down_ei
                    + torch.tensor(
                        [[i * num_nodes_coarse], [i * num_nodes_fine]],
                        device=device,
                        dtype=torch.long,
                    )
                )
            up_edge_index_list.append(torch.cat(batched_up, dim=1))
            up_edge_attr_list.append(up_ea.repeat(num_graphs, 1))
            down_edge_index_list.append(torch.cat(batched_down, dim=1))
            down_edge_attr_list.append(down_ea.repeat(num_graphs, 1))

        return (
            mesh_features_list,
            mesh_edge_index_list,
            mesh_edge_attr_list,
            up_edge_index_list,
            up_edge_attr_list,
            down_edge_index_list,
            down_edge_attr_list,
        )

    def _processor_rollout_mesh(self, data: HeteroData, encoded_mesh_features: torch.Tensor) -> torch.Tensor:
        if getattr(self.args, 'hierarchical', False):
            return self._processor_rollout_mesh_hierarchical(data, encoded_mesh_features)

        processor_input = {'mesh': encoded_mesh_features}
        mesh_edge_key = ('mesh', 'to', 'mesh')
        processor_edge_index = {mesh_edge_key: data.edge_index_dict[mesh_edge_key]}
        processor_edge_attr = {mesh_edge_key: data[mesh_edge_key].edge_attr}

        for _, features in processor_input.items():
            if torch.isnan(features).any():
                logger.info("NaN detected in encoded_features for mesh")

        num_processor_rollout_steps = getattr(self.args, 'num_processor_rollout_steps', 1)
        _alpha = getattr(self.args, 'processor_rollout_alpha', None)
        processor_rollout_alpha = (
            (1.0 / max(1, num_processor_rollout_steps)) if _alpha is None else float(_alpha)
        )
        current_mesh_dict = {key: val.clone() for key, val in processor_input.items()}
        for _ in range(num_processor_rollout_steps):
            # No outer checkpoint: the processor already checkpoints each of its
            # mesh_gnn_layers steps internally.  Wrapping it in a second
            # use_reentrant=False checkpoint creates nested non-reentrant
            # checkpoints whose dict-valued ctx.inputs form reference cycles
            # that Python's reference-count GC cannot collect, leaking
            # ~1.5 GiB of checkpoint saves per training step.
            out = self.processor(current_mesh_dict, processor_edge_index, processor_edge_attr)
            for key in out:
                current_mesh_dict[key] = (
                    (1.0 - processor_rollout_alpha) * current_mesh_dict[key]
                    + processor_rollout_alpha * out[key]
                )
        out_mesh = current_mesh_dict['mesh']
        if torch.isnan(out_mesh).any():
            logger.info("NaN detected in processed_features for mesh")
        logger.debug(
            "[PROCESSOR] Processed mesh shape after %d rollout step(s) × %d layers: %s",
            num_processor_rollout_steps, self.args.mesh_gnn_layers, out_mesh.shape,
        )
        return out_mesh

    def _processor_rollout_mesh_hierarchical(
        self, data: HeteroData, encoded_mesh_features: torch.Tensor
    ) -> torch.Tensor:
        """Multi-level mesh processor with coarse→fine gating (see gnn_model)."""
        hd = self.hidden_dim
        num_levels = self.num_mesh_levels
        hier_ptype = getattr(self, '_hierarchical_processor_type', 'interaction')

        num_processor_rollout_steps = getattr(self.args, 'num_processor_rollout_steps', 1)
        _alpha = getattr(self.args, 'processor_rollout_alpha', None)
        processor_rollout_alpha = (
            (1.0 / max(1, num_processor_rollout_steps)) if _alpha is None else float(_alpha)
        )

        if hier_ptype == 'transformer':
            self.processor.reset()

        state = encoded_mesh_features.clone()
        for _ in range(num_processor_rollout_steps):
            current_mesh_features = state

            (
                mesh_features_list,
                mesh_edge_index_list,
                mesh_edge_attr_list,
                up_edge_index_list,
                _up_edge_attr_list,
                down_edge_index_list,
                _down_edge_attr_list,
            ) = self._build_hierarchical_mesh_lists(data, current_mesh_features)

            if hier_ptype == 'transformer':
                processed_levels = self.processor(
                    mesh_features_list,
                    up_edge_index_list,
                    down_edge_index_list,
                    mesh_edge_index_list,
                    mesh_edge_attr_list,
                )
            else:
                processed_levels = self.processor(
                    mesh_features_list,
                    mesh_edge_index_list,
                    mesh_edge_attr_list,
                    up_edge_index_list,
                    _up_edge_attr_list,
                    down_edge_index_list,
                    _down_edge_attr_list,
                )

            if num_levels > 1:
                fine_features = processed_levels[0]
                coarse_features = processed_levels[1]
                down_edge_index = down_edge_index_list[0]

                coarse_gathered = coarse_features[down_edge_index[0]]
                fine_conditioned = torch.zeros_like(fine_features)
                fine_conditioned.scatter_reduce_(
                    0,
                    down_edge_index[1].unsqueeze(-1).expand(-1, hd),
                    coarse_gathered,
                    reduce='mean',
                )
                fine_conditioned_norm = self.coarse_to_fine_norm(fine_conditioned)
                delta = self.coarse_to_fine_proj(fine_conditioned_norm)
                gate_input = torch.cat([fine_features, fine_conditioned_norm], dim=-1)
                gate = self.coarse_to_fine_gate(gate_input)
                out_fine = fine_features + gate * delta
            else:
                out_fine = processed_levels[0]

            state = (
                (1.0 - processor_rollout_alpha) * state
                + processor_rollout_alpha * out_fine
            )

        if torch.isnan(state).any():
            logger.info("NaN detected in processed_features for mesh (hierarchical)")
        kind = getattr(self, '_hierarchical_processor_type', 'interaction')
        extra = (
            f"transformer w={getattr(self.args, 'processor_window', 4)}"
            if kind == 'transformer'
            else f"{self.args.mesh_gnn_layers} intra steps"
        )
        logger.debug(
            "[PROCESSOR hierarchical:%s] mesh shape after %d rollout step(s) (%s): %s",
            kind, num_processor_rollout_steps, extra, state.shape,
        )
        return state

    def _forward_observations(self, data: HeteroData) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Single entry for observation → mesh → predictions.

        - **Non-AR** (``k==1`` or ``--no_obs_autoregressive``): one input encode, one processor,
          decode all target heads in parallel.
        - **AR** (default when ``num_target_bins`` > 1): loop ``h = 0 … k-1`` with input encode
          at ``h==0``, feedback encode at ``h≥1``, then processor and per-horizon decode.
        """
        k = self._num_target_bins
        use_ar = self._use_obs_autoregressive(k)
        encoder_order = getattr(self, '_encoder_edge_order', [])

        _mem("forward: start (data on GPU)")
        # Log data batch tensor sizes
        for nt, nd in data.node_items():
            if hasattr(nd, 'x') and nd.x is not None:
                logger.debug("[MEM]   data['%s'].x: %s  %.1f MiB", nt, tuple(nd.x.shape), _tensor_mb(nd.x))
        for et, ed in data.edge_items():
            ei = ed.edge_index
            ea = getattr(ed, 'edge_attr', None)
            ea_str = f"  edge_attr={tuple(ea.shape)} {_tensor_mb(ea):.1f} MiB" if ea is not None else ""
            logger.debug("[MEM]   data[%s].edge_index: %s  %.1f MiB%s", et, tuple(ei.shape), _tensor_mb(ei), ea_str)

        embedded_inputs = self._embed_raw_inputs(data)
        for node_type, features in embedded_inputs.items():
            if torch.isnan(features).any():
                logger.info("NaN detected in embedded_features for %s", node_type)

        encoded_mesh_features = embedded_inputs['mesh']

        if not use_ar:
            encoded_mesh_features = self._run_bipartite_encoders(
                data,
                encoded_mesh_features,
                embedded_inputs,
                encoder_order,
            )
            _mem("forward: after encoders, before processor")
            encoded_mesh_features = self._processor_rollout_mesh(data, encoded_mesh_features)
            _mem("forward: after processor, before decode")
            predictions = self.decode(data, {'mesh': encoded_mesh_features}, allowed_dst_types=None)
            _mem("forward: after decode")
            return predictions, encoded_mesh_features

        predictions: Dict[str, torch.Tensor] = {}
        teacher = self._obs_ar_teacher_forcing()

        for h in range(k):
            if h == 0:
                encoded_mesh_features = self._run_bipartite_encoders(
                    data,
                    encoded_mesh_features,
                    embedded_inputs,
                    encoder_order,
                )
            else:
                feedback_embedded: Dict[str, torch.Tensor] = {}
                for obs_type, inst_name in self._instrument_order:
                    prev_tgt = target_node_type(obs_type, inst_name, h - 1, k)
                    if prev_tgt not in data.node_types:
                        continue
                    if data[prev_tgt].num_nodes == 0:
                        continue
                    edge_fb = (prev_tgt, ENCODE_OBS_TO_MESH_REL, 'mesh')
                    if edge_fb not in data.edge_index_dict:
                        continue
                    if teacher and hasattr(data[prev_tgt], 'y') and data[prev_tgt].y.numel() > 0:
                        src_norm = data[prev_tgt].y
                    else:
                        if prev_tgt not in predictions:
                            continue
                        src_norm = predictions[prev_tgt]
                    num_metadata = self._num_metadata_for_instrument(obs_type, inst_name)
                    pseudo_x = build_feedback_input_x_from_target(
                        obs_type, src_norm, data[prev_tgt], num_metadata
                    )
                    input_nt = f"{obs_type}_{inst_name}_input"
                    feedback_embedded[prev_tgt] = self.observation_embedders[input_nt](pseudo_x)

                for obs_type, inst_name in self._instrument_order:
                    prev_tgt = target_node_type(obs_type, inst_name, h - 1, k)
                    if prev_tgt not in feedback_embedded:
                        continue
                    edge_fb = (prev_tgt, ENCODE_OBS_TO_MESH_REL, 'mesh')
                    if edge_fb not in data.edge_index_dict:
                        continue
                    input_edge = (f"{obs_type}_{inst_name}_input", 'to', 'mesh')
                    enc_key = self._edge_key(input_edge)
                    encoder = self.observation_encoders[enc_key]
                    edge_index = data.edge_index_dict[edge_fb]
                    edge_attr = data[edge_fb].edge_attr
                    send_rep = feedback_embedded[prev_tgt]
                    logger.debug(
                        "[DEBUG] AR feedback encode: %s  send_rep=%s  mesh=%s",
                        edge_fb, send_rep.shape, encoded_mesh_features.shape,
                    )
                    if self.training:
                        encoded_mesh_features = checkpoint(
                            encoder,
                            send_rep,
                            encoded_mesh_features,
                            edge_attr,
                            edge_index,
                            use_reentrant=False,
                        )
                    else:
                        encoded_mesh_features = encoder(
                            send_rep=send_rep,
                            rec_rep=encoded_mesh_features,
                            edge_rep=edge_attr,
                            edge_index=edge_index,
                        )

            encoded_mesh_features = self._processor_rollout_mesh(data, encoded_mesh_features)
            allowed = self._allowed_dst_types_at_horizon(h, k)
            preds_h = self.decode(
                data, {'mesh': encoded_mesh_features}, allowed_dst_types=allowed
            )
            predictions.update(preds_h)

        return predictions, encoded_mesh_features

    def forward(self, data: 'HeteroData') -> Dict[str, torch.Tensor]:
        """
        Defines the forward pass of the model with split input/target node types.
        With ``num_target_bins`` = k > 1 and autoregressive mode (default), runs
        encode → processor → decode per horizon, feeding previous targets into the mesh
        via ``encode_obs_to_mesh`` edges (teacher forcing on ground-truth ``y`` by default).
        """
        for node_type, x in data.x_dict.items():
            if torch.isnan(x).any():
                logger.info("[NaN Check] NaN detected in input data for node_type: %s", node_type)

        num_graphs = data.num_graphs
        num_mesh_nodes = self.mesh_x.shape[0]

        data['mesh'].x = self.mesh_x.repeat(num_graphs, 1)
        data['mesh', 'to', 'mesh'].edge_attr = self.mesh_edge_attr.repeat(num_graphs, 1)

        edge_indices = [self.mesh_edge_index + i * num_mesh_nodes for i in range(num_graphs)]
        data['mesh', 'to', 'mesh'].edge_index = torch.cat(edge_indices, dim=1)

        predictions, processed_mesh = self._forward_observations(data)

        for node_type, pred in predictions.items():
            if torch.isnan(pred).any():
                logger.info("NaN detected in predictions for %s within forward pass", node_type)

        if self.debug_gradients:
            _validate_graph(data, {'mesh': processed_mesh})

        # Store processor output so _generic_step can route empty-batch zero losses
        # through the processor checkpoint chain, ensuring backward() traverses and
        # frees all CheckpointBackward nodes even when predictions carry no grad_fn.
        self._last_processed_mesh = processed_mesh

        return predictions
