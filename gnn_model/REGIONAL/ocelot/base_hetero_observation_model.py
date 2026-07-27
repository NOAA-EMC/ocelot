"""
Implementation of a graph neural network model for processing heterogeneous observations
with different variable types and spatial locations.
"""

import gc
import torch
import torch.nn as nn
import os
from typing import Dict, Any, Tuple, Optional, Set
import torch.distributed as dist
from torch_geometric.data import Data, Dataset, Batch
from torch_geometric.data import HeteroData
from torch.utils.checkpoint import checkpoint

from ocelot.ar_dop_model import ARDOPModel, _instrument_key_for_obs_config
from ocelot import utils
from ocelot.create_mesh_graph_global import create_mesh, obs_mesh_conn
from ocelot.metrics_dop import gradient_loss, multiscale_loss
from ocelot.logger import logger

# logger = setup_logger(__name__)


def _validate_graph(data: 'HeteroData', processed_features: Dict[str, torch.Tensor]):
    """
    Performs shape and index validation on the graph data.
    """
    for (src, _, dst), edge_index in data.edge_index_dict.items():
        # Determine the number of nodes for the source and destination types
        if src == "mesh":
            max_src = processed_features['mesh'].size(0) - 1
        elif src.endswith("_input"):
            max_src = data[src].x.size(0) - 1
        elif src.endswith("_target"):
            # Target nodes might not have input 'x', but they have a defined number of nodes
            max_src = data[src].num_nodes - 1
        else:
            continue

        if dst == "mesh":
            max_dst = processed_features['mesh'].size(0) - 1
        elif dst.endswith("_input"):
            # This case should not typically happen, but handle for completeness
            max_dst = data[dst].x.size(0) - 1
        elif dst.endswith("_target"):
            max_dst = data[dst].num_nodes - 1
        else:
            continue

        # Assert that edge indices are within the valid range
        assert edge_index[0].max() <= max_src, (
            f"Invalid edge index for src '{src}': max {edge_index[0].max()} exceeds {max_src}"
        )
        assert edge_index[0].min() >= 0, (
            f"Negative edge index for src '{src}'"
        )

        assert edge_index[1].max() <= max_dst, (
            f"Invalid edge index for dst '{dst}': max {edge_index[1].max()} exceeds {max_dst}"
        )
        assert edge_index[1].min() >= 0, (
            f"Negative edge index for dst '{dst}'"
        )

        print(f"[CHECK] Edge {src}->{dst} ok: ({edge_index[0].min()}–{edge_index[0].max()}) to ({edge_index[1].min()}–{edge_index[1].max()})")


class BaseHeteroGraphModel(ARDOPModel):
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
        # ... (your existing __init__ content) ...
        
        # For storing outputs from validation/test steps
        self.validation_step_outputs = []
        self.test_step_outputs = []

        self.args = args
        self.debug_gradients = args.debug_gradients if hasattr(args, 'debug_gradients') else False
        self.hidden_dim = args.hidden_dim
        self.num_heads = args.num_heads
        self.mlp_blueprint_end = [args.hidden_dim] * (args.hidden_layers + 1)

        # Initialize network dictionaries
        self.observation_embedders = nn.ModuleDict()  # Embeds raw obs features
        self.observation_encoders = nn.ModuleDict()  # GNNs: Encodes obs features onto mesh
        self.observation_decoders = nn.ModuleDict()  # GNNs: Decodes mesh features to obs
        self.output_mappers = nn.ModuleDict()       # MLPs: Maps hidden state to output dim
        self.scan_angle_embedders = nn.ModuleDict() # Embeds scan angle for satellite data

        # A dedicated embedder for the static mesh features
        self.mesh_embedder = utils.make_mlp([args.mesh_feature_dim] + self.mlp_blueprint_end)

        # GNN processor
        self.processor = None
        self.is_hierarchical = False
        self.num_mesh_levels = 1

        # Initialize graph structures
        self.mesh_structure = None  # Will be set in setup_mesh
        self.mesh_graph = None      # Will be set in setup_mesh

        self.create_mesh_structures()
        
        # Set up observation networks if config provided
        if hasattr(args, 'observation_config') and args.observation_config:
            self.setup_observation_networks(args.observation_config)

    def setup(self, stage=None):
        """
        Called by PyTorch Lightning before training/testing. This is the right place to
        initialize the static mesh data.
        """
        self.setup_mesh()

    def setup_mesh(self):
        """
        Overrides the base method to create and register static mesh data as buffers.
        This ensures the mesh data is part of the model's state and moved to the correct device.
        """
        # Finest mesh (index 0) is used for obs ↔ mesh encode/decode.
        mesh_x = self.mesh_graph.featurs[0]
        mesh_edge_index = self.mesh_graph.edge_index[0]
        mesh_edge_attr = self.mesh_graph.edge_attr[0]

        self.register_buffer('mesh_x', mesh_x)
        self.register_buffer('mesh_edge_index', mesh_edge_index)
        self.register_buffer('mesh_edge_attr', mesh_edge_attr)

        if getattr(self, 'is_hierarchical', False):
            ms = self.mesh_structure
            for i in range(self.num_mesh_levels):
                self.register_buffer(
                    f'mesh_x_level_{i}', ms['mesh_features_torch'][i].clone()
                )
                self.register_buffer(
                    f'mesh_edge_index_level_{i}',
                    ms['m2m_edge_index_torch'][i].clone(),
                )
                self.register_buffer(
                    f'mesh_edge_attr_level_{i}',
                    ms['m2m_features_torch'][i].clone(),
                )
            for i, up_ei in enumerate(ms['mesh_up_ei_list']):
                self.register_buffer(f'mesh_up_edge_index_{i}', up_ei.clone())
            for i, up_f in enumerate(ms['mesh_up_features_list']):
                self.register_buffer(f'mesh_up_edge_attr_{i}', up_f.clone())
            for i, down_ei in enumerate(ms['mesh_down_ei_list']):
                self.register_buffer(f'mesh_down_edge_index_{i}', down_ei.clone())
            for i, down_f in enumerate(ms['mesh_down_features_list']):
                self.register_buffer(f'mesh_down_edge_attr_{i}', down_f.clone())

    def setup_observation_networks(self, observation_config: Dict):
        """
        Creates encoding/decoding networks and initializes the GNN Processor.
        """

        raise NotImplementedError("No prediction step implemented")

    def create_mesh_structures(self) -> None:
        """
        Create or load the static mesh structure for the domain.
        This initializes both self.mesh_structure and self.mesh_graph.
        
        Raises:
            FileNotFoundError: If grid coordinates file cannot be found
            RuntimeError: If mesh creation fails
        """
        try:
            mesh_splits = int(getattr(self.args, 'mesh_splits', 6))
            levels = getattr(self.args, 'levels', None)
            hierarchical = bool(getattr(self.args, 'hierarchical', False))

            mesh_structure = create_mesh(
                splits=mesh_splits,
                levels=levels,
                hierarchical=hierarchical,
                plot=False,
            )

            self.mesh_structure = mesh_structure
            self.is_hierarchical = hierarchical
            self.num_mesh_levels = len(mesh_structure['m2m_graphs'])

            self.mesh_graph = Data(
                featurs=mesh_structure['mesh_features_torch'],
                pos=torch.from_numpy(mesh_structure['mesh_lat_lon_list'][0]).float(),
                m2m_graphs=mesh_structure['m2m_graphs'],
                edge_index=mesh_structure['m2m_edge_index_torch'],
                edge_attr=mesh_structure['m2m_features_torch'],
            )

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Could not find grid coordinates file: {e}")
        except Exception as e:
            raise RuntimeError(f"Failed to create mesh structure: {e}")

        self.setup_mesh()

    @staticmethod
    def _edge_key(edge_type: Tuple[str, str, str]) -> str:
        """Converts an edge_type tuple to a string key for ModuleDict."""
        return f'{edge_type[0]}__{edge_type[1]}__{edge_type[2]}'

    def forward(self, data: 'HeteroData') -> Dict[str, torch.Tensor]:
        """
        Defines the forward pass of the model with split input/target node types.
        Includes shape and index sanity checks.
        """
        raise NotImplementedError("No prediction step implemented")

    def decode(
        self,
        data: 'HeteroData',
        processed_features: Dict[str, torch.Tensor],
        allowed_dst_types: Optional[Set[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decodes processed mesh features back to the target observation space.

        Args:
            allowed_dst_types: If set, only decode mesh -> dst edges whose ``dst_type`` is
                in this set (used for autoregressive per-horizon decode).
        """
        def _mem_decode(label):
            if torch.cuda.is_available():
                a = torch.cuda.memory_allocated() / 1024**3
                r = torch.cuda.memory_reserved() / 1024**3
                logger.debug(f"[MEM] decode {label}: allocated={a:.2f} GiB  reserved={r:.2f} GiB")

        predictions = {}
        mesh_features = processed_features['mesh']

        for edge_type, edge_index in data.edge_index_dict.items():
            src_type, _, dst_type = edge_type
            logger.info(f"Processing edge_type: {edge_type}, src_type: {src_type}, dst_type: {dst_type}")
            if src_type == 'mesh' and dst_type.endswith('_target'):
                if allowed_dst_types is not None and dst_type not in allowed_dst_types:
                    continue
                target_features = torch.zeros(
                    data[dst_type].num_nodes, self.hidden_dim, device=self.device
                )

                # Get the correct GNN decoder
                decoder = self.observation_decoders[self._edge_key(edge_type)]

                edge_attr = data[edge_type].edge_attr

                def _tmb(t):
                    return t.element_size() * t.nelement() / 1024**2

                ei_mb = _tmb(edge_index)
                ea_mb = _tmb(edge_attr) if edge_attr is not None else 0.0
                logger.debug(
                    f"[MEM] decoder {edge_type}:"
                    f"  tgt={tuple(target_features.shape)} {_tmb(target_features):.1f} MiB"
                    f"  edge_index={tuple(edge_index.shape)} {ei_mb:.1f} MiB"
                    f"  edge_attr={tuple(edge_attr.shape) if edge_attr is not None else None} {ea_mb:.1f} MiB"
                    f"  (checkpoint saves ~{ei_mb + ea_mb:.0f} MiB)"
                )
                _mem_decode(f"before {edge_type}")

                # Decode from mesh to target observation nodes.
                # When there are no target nodes the decoder output is trivially empty.
                # Skipping checkpoint in that case is critical: checkpoint saves
                # mesh_features [num_mesh_nodes × hidden_dim] so it can recompute the
                # decoder during backward.  That save keeps the entire processor
                # checkpoint chain alive.  With zero target nodes backward never
                # traverses this path (the loss doesn't connect to empty predictions),
                # so the checkpoint saves would accumulate ~1.5 GiB per step.
                if self.training:
                    if target_features.shape[0] == 0:
                        # No target nodes → output is trivially empty; skip checkpoint
                        # to avoid retaining mesh_features and the full processor graph.
                        decoded_target_features = torch.zeros_like(target_features)
                    else:
                        decoded_target_features = checkpoint(
                            decoder,
                            mesh_features,
                            target_features,
                            edge_attr,
                            edge_index,
                            use_reentrant=False
                        )
                else:
                    decoded_target_features = decoder(
                        send_rep=mesh_features,
                        rec_rep=target_features,
                        edge_rep=edge_attr,
                        edge_index=edge_index
                    )
                _mem_decode(f"after {edge_type}")

                # Combine decoded mesh state with embedded target context (target .x: meta + aux).
                target_context = data[dst_type].x
                context_embedded = self.scan_angle_embedders[dst_type](target_context)
                final_features = torch.cat([decoded_target_features, context_embedded], dim=-1)
                predictions[dst_type] = self.output_mappers[dst_type](final_features)

                if (
                    self.debug_gradients
                    and dst_type == "state_ges_target"
                    and predictions[dst_type].requires_grad
                ):
                    predictions[dst_type].register_hook(print_grad(f"predictions[{dst_type}]"))

        return predictions

    def _zero_loss_with_graph(
        self, predictions: Dict[str, torch.Tensor], mesh: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Return a scalar 0 that still depends on the autograd graph.

        A standalone leaf tensor (even with requires_grad=True) does **not**
        participate in AMP GradScaler's inf/unscale bookkeeping. Prefer routing
        through `mesh` (the processor output) so backward() explicitly calls
        CheckpointBackward.apply() on every processor checkpoint node, freeing
        their saved tensors. Without this, C++/Python reference cycles formed by
        use_reentrant=False checkpoint prevent Python's GC from breaking the cycle.
        """
        if mesh is not None and mesh.numel() > 0:
            return mesh.sum() * 0.0
        for pred in predictions.values():
            if isinstance(pred, torch.Tensor) and pred.numel() > 0:
                return pred.sum() * 0.0
        for p in self.parameters():
            if p.requires_grad and p.numel() > 0:
                return p.sum() * 0.0
        return torch.zeros((), device=self.device, dtype=torch.float32)

    def _get_instrument_weight(self, node_type: str) -> float:
        """
        Resolve the per-instrument loss weight for a target node type
        (e.g. ``satellite_diag_atms_target`` or ``conventional_surface_obs_h0_target``)
        via ``args.instrument_weights``, which is keyed the same as
        ``observation_config`` (e.g. ``diag_atms``). Defaults to 1.0 when no
        weight is configured for the instrument.
        """
        instrument_weights = getattr(self.args, 'instrument_weights', None)
        if not instrument_weights:
            return 1.0

        base = node_type[:-len('_target')] if node_type.endswith('_target') else node_type
        _obs_category, *inst_parts = base.split('_')
        cfg_key = _instrument_key_for_obs_config('_'.join(inst_parts))

        return float(instrument_weights.get(cfg_key, 1.0))

    def _generic_step(self, batch: 'HeteroData', batch_idx: int, stage: str) -> torch.Tensor:
        """
        Generic step for training, validation, and testing.

        Args:
            batch (HeteroData): The data batch.
            batch_idx (int): The index of the batch.
            stage (str): The stage ('train', 'val', or 'test').

        Returns:
            torch.Tensor: The calculated loss.
        """
        # DEBUG: Log batch structure for the first batch of the training stage
        if stage == 'train' and batch_idx == 0:
            logger.info("=" * 80)
            logger.info("DEBUG: Batch Structure")
            logger.info("=" * 80)
            logger.info(f"Batch node types: {batch.node_types}")
            logger.info(f"Batch edge types: {batch.edge_types}")
            for node_type in batch.node_types:
                node_data = batch[node_type]
                logger.info(f"\nNode Type: {node_type}")
                logger.info(f"  num_nodes: {node_data.num_nodes if hasattr(node_data, 'num_nodes') else 'N/A'}")
                if hasattr(node_data, 'x'):
                    logger.info(f"  x shape: {tuple(node_data.x.shape)}")
                if hasattr(node_data, 'y'):
                    logger.info(f"  y shape: {tuple(node_data.y.shape)}")
                if hasattr(node_data, 'pos'):
                    logger.info(f"  pos shape: {tuple(node_data.pos.shape)}")
            logger.info("=" * 80)
        
        predictions = self.forward(batch)
        # Retrieve and immediately clear the processor output stored by forward().
        # This mesh tensor retains the full grad_fn chain through every processor
        # checkpoint node, so routing the zero loss through it forces backward() to
        # call CheckpointBackward.apply() on all nodes — the only way to break the
        # C++/Python reference cycles that prevent Python GC from freeing saves.
        _stored_mesh = getattr(self, '_last_processed_mesh', None)
        self._last_processed_mesh = None
        total_loss: Optional[torch.Tensor] = None
        total_weight = 0.0

        # DEBUG: Print stats for the first batch of the training stage
        if stage == 'train' and batch_idx == 0:
            logger.info("=" * 80)
            logger.info("DEBUG: Inspecting First Training Batch")
            logger.info("=" * 80)
            def _stats(t):
                if t.numel() == 0:
                    return "empty"
                t = t.float()
                return f"Mean: {t.mean():.4f}, Std: {t.std():.4f}, Min: {t.min():.4f}, Max: {t.max():.4f}"
            for node_type, pred in predictions.items():
                logger.info(f"Node Type: {node_type}")
                if hasattr(batch[node_type], 'x') and hasattr(batch[node_type], 'y'):
                    inputs = batch[node_type].x
                    targets = batch[node_type].y
                    logger.info(f"  Input Features  | Shape: {tuple(inputs.shape)} | {_stats(inputs)}")
                    logger.info(f"  Target Labels   | Shape: {tuple(targets.shape)} | {_stats(targets)}")
                    logger.info(f"  Predictions     | Shape: {tuple(pred.shape)} | {_stats(pred)}")
                else:
                    missing_attrs = []
                    if not hasattr(batch[node_type], 'x'):
                        missing_attrs.append('x')
                    if not hasattr(batch[node_type], 'y'):
                        missing_attrs.append('y')
                    logger.info(f"  (Missing attributes: {', '.join(missing_attrs) if missing_attrs else 'unknown'})")
                logger.info("-" * 80)
            logger.info("=" * 80)

        if not predictions:
            loss = self._zero_loss_with_graph({}, mesh=_stored_mesh)
            sync = stage in ('val', 'test')
            self.log(f'{stage}_loss', loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=1, sync_dist=sync)
            return loss

        # Calculate loss for each observation type
        for node_type, pred in predictions.items():
            # Ensure the node type has ground truth labels in the batch
            if hasattr(batch[node_type], 'y'):
                target = batch[node_type].y
                pred_std = self._get_std_for_loss(pred)

                mask = batch[node_type].valid_mask if hasattr(batch[node_type], 'valid_mask') else None

                # Diagnostic: log valid-mask coverage for surface obs node types
                if mask is not None and 'surface_obs' in node_type:
                    n_total = mask.numel()
                    n_valid = mask.sum().item()
                    logger.warning(
                        f"[mask] {stage} {node_type}: {n_valid}/{n_total} valid "
                        f"({100*n_valid/max(n_total,1):.1f}%) shape={tuple(mask.shape)}"
                    )

                # The loss function will handle the masking internally.
                # We check for numel on the original prediction tensor before masking.
                if pred.numel() > 0:
                    logger.debug(f"Computing loss for {node_type} | pred: {tuple(pred.shape)}, "
                                f"target: {tuple(target.shape)}, pred_std: {tuple(pred_std.shape)}, "
                                f"mask: {tuple(mask.shape) if mask is not None else None}")
                    loss = self.loss(pred, target, pred_std, mask=mask)

                    # Optional auxiliary terms (see possible_future_improvement.md):
                    # neighbor-difference loss to sharpen fronts/discontinuities, and
                    # multi-scale loss so synoptic-scale structure is scored directly
                    # rather than relying on the per-node loss to get it right.
                    grad_weight = float(getattr(self.args, 'gradient_loss_weight', 0.0) or 0.0)
                    if grad_weight > 0:
                        edge_type = (node_type, 'grad', node_type)
                        grad_edge_index = (
                            batch[edge_type].edge_index if edge_type in batch.edge_types else None
                        )
                        loss = loss + grad_weight * gradient_loss(pred, target, grad_edge_index, mask=mask)

                    multiscale_weight = float(getattr(self.args, 'multiscale_loss_weight', 0.0) or 0.0)
                    if multiscale_weight > 0 and hasattr(batch[node_type], 'pos'):
                        scales = getattr(self.args, 'multiscale_scales', None) or (0.5, 1.0, 2.0, 4.0)
                        loss = loss + multiscale_weight * multiscale_loss(
                            pred, target, batch[node_type].pos, mask=mask, scales=scales
                        )

                    # A NaN loss can occur if the mask is all False, which is valid.
                    if not torch.isnan(loss):
                        weight = self._get_instrument_weight(node_type)
                        weighted_loss = loss * weight
                        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
                        total_weight += weight
                        # Detach before logging: torchmetrics MeanMetric accumulates tensors
                        # with on_epoch=True by summing them into a running total, keeping
                        # every step's grad_fn chain (predictions → decoder checkpoint →
                        # processor saves ~1.28 GiB) alive for the entire epoch.
                        self.log(f'{stage}_loss_{node_type}', loss.detach(), on_step=True, on_epoch=True, prog_bar=False,
                                 logger=True, batch_size=1)

        # Weighted average of the per-instrument losses that had predictions.
        if total_loss is None:
            avg_loss = self._zero_loss_with_graph(predictions, mesh=_stored_mesh)
        elif total_weight > 0:
            avg_loss = total_loss / total_weight
        else:
            # All contributing instrument weights are 0 (e.g. explicitly zeroed
            # out); total_loss is already ~0 but still connected to the
            # autograd graph, so use it directly rather than dividing by 0.
            avg_loss = total_loss
        # sync_dist=True for val/test so all DDP ranks agree on the epoch metric.
        # ModelCheckpoint generates the checkpoint filename from val_loss; if ranks
        # see different values (sync_dist=False), they compute different filenames
        # and enter file_exists() a different number of times → collective mismatch.
        sync = stage in ('val', 'test')
        self.log(f'{stage}_loss', avg_loss.detach(), on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=1, sync_dist=sync)

        # Log metrics and visualizations for val/test stages
        if stage in ['val', 'test']:
            self._log_and_visualize(batch, predictions, stage)

        return avg_loss

    def training_step(self, batch: 'HeteroData', batch_idx: int) -> torch.Tensor:
        """
        Training step.
        """
        return self._generic_step(batch, batch_idx, 'train')

    def validation_step(self, batch: 'HeteroData', batch_idx: int) -> torch.Tensor:
        """
        Validation step.
        """
        return self._generic_step(batch, batch_idx, 'val')

    def test_step(self, batch: 'HeteroData', batch_idx: int) -> torch.Tensor:
        """
        Test step.
        """
        return self._generic_step(batch, batch_idx, 'test')

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Force Python GC and return freed CUDA blocks to the pool each step.
        # Checkpoint nodes with dict inputs can form reference cycles that the
        # reference-count GC misses; the cyclic GC breaks them here.
        gc.collect()
        torch.cuda.empty_cache()

    @staticmethod
    def _first_weight(module):
        """First 'weight' parameter of a module, searched in registration order.

        Encoders/decoders may be plain ``nn.Sequential`` MLPs (indexable) or
        graph modules like ``InteractionNet``/``BipartiteGAT`` (not indexable),
        so we can't assume ``module[0]`` works.
        """
        if module is None:
            return None
        for name, param in module.named_parameters():
            if name.endswith('.weight') or name == 'weight':
                return param
        return None

    @staticmethod
    def _last_weight(module):
        """Last 'weight' parameter of a module, searched in registration order."""
        if module is None:
            return None
        last = None
        for name, param in module.named_parameters():
            if name.endswith('.weight') or name == 'weight':
                last = param
        return last

    def on_after_backward(self):
        """
        Hook to inspect gradients after the backward pass.
        """
        if not self.debug_gradients:
            return

        # We only need to check this for a few steps
        if self.trainer.global_step < 3:
            print(f"\n--- DEBUG: Gradient Check at Step {self.trainer.global_step} ---")

            # Check gradient of an encoder (first weight found)
            if self.observation_encoders:
                first_encoder_key = next(iter(self.observation_encoders.keys()))
                encoder_weight = self._first_weight(self.observation_encoders[first_encoder_key])
                if encoder_weight is None:
                    print(f"  Encoder ('{first_encoder_key}') | No weight parameter found!")
                elif encoder_weight.grad is not None:
                    grad_mean = encoder_weight.grad.mean()
                    grad_std = encoder_weight.grad.std()
                    print(f"  Encoder ('{first_encoder_key}') | Grad Mean: {grad_mean:.2e}, Grad Std: {grad_std:.2e}")
                else:
                    print(f"  Encoder ('{first_encoder_key}') | Grad is None!")

            # Check gradient of a decoder (last weight found)
            if self.observation_decoders:
                first_decoder_key = next(iter(self.observation_decoders.keys()))
                decoder_weight = self._last_weight(self.observation_decoders[first_decoder_key])
                if decoder_weight is None:
                    print(f"  Decoder ('{first_decoder_key}') | No weight parameter found!")
                elif decoder_weight.grad is not None:
                    grad_mean = decoder_weight.grad.mean()
                    grad_std = decoder_weight.grad.std()
                    print(f"  Decoder ('{first_decoder_key}') | Grad Mean: {grad_mean:.2e}, Grad Std: {grad_std:.2e}")
                else:
                    print(f"  Decoder ('{first_decoder_key}') | Grad is None!")

            self._debug_state_gradients()
            print("--- END GRADIENT CHECK ---\n")

    def _debug_state_gradients(self):
        """
        Gradient stats for every module touching the ``state_ges`` (RRFS
        analysis-increment) pathway: input embedder -> encoder -> processor
        (shared with all node types) -> decoder -> scan-angle embedder ->
        output mapper. Print for each so a vanishing/exploding/None stage
        pinpoints where the state branch stops learning.
        """
        def _report(label, weight):
            if weight is None:
                print(f"  [state] {label} | not found")
                return
            if weight.grad is None:
                print(f"  [state] {label} | Grad is None!")
                return
            g = weight.grad
            nan_flag = " NaN!" if torch.isnan(g).any() else ""
            inf_flag = " Inf!" if torch.isinf(g).any() else ""
            print(
                f"  [state] {label} | Grad Mean: {g.mean():.2e}, Std: {g.std():.2e}, "
                f"AbsMax: {g.abs().max():.2e}{nan_flag}{inf_flag}"
            )

        def _lookup(module_dict_name, key):
            module_dict = getattr(self, module_dict_name, None)
            if module_dict is None or key not in module_dict:
                return None
            return module_dict[key]

        input_key = "state_ges_input"
        target_key = "state_ges_target"
        encoder_key = self._edge_key((input_key, "to", "mesh"))
        decoder_key = self._edge_key(("mesh", "to", target_key))

        embedder = _lookup("observation_embedders", input_key)
        _report(f"embedder[{input_key}] first layer", embedder[0].weight if embedder is not None else None)

        encoder = _lookup("observation_encoders", encoder_key)
        _report(f"encoder[{encoder_key}] first layer", self._first_weight(encoder))

        decoder = _lookup("observation_decoders", decoder_key)
        _report(f"decoder[{decoder_key}] last layer", self._last_weight(decoder))

        scan_embedder = _lookup("scan_angle_embedders", target_key)
        _report(f"scan_angle_embedder[{target_key}] first layer", scan_embedder[0].weight if scan_embedder is not None else None)

        output_mapper = _lookup("output_mappers", target_key)
        _report(f"output_mapper[{target_key}] last layer", output_mapper[-1].weight if output_mapper is not None else None)

    def _log_and_visualize(self, batch: 'HeteroData', predictions: Dict[str, torch.Tensor], stage: str):
        """
        Logs metrics and visualizations for a batch of HeteroData.
        This method overrides the parent's implementation to handle the specific structure of HeteroData.
        """

        # --- Save batch and predictions for offline analysis ---
        # This runs for the first validation batch of each validation run.
        if stage == 'val':
            current_epoch = self.trainer.current_epoch
            current_step = self.trainer.global_step
            last_saved_step = getattr(self, '_last_saved_debug_step', -1)

            # Save once per validation run (using global_step to handle step-based validation)
            # Only save the first batch of each validation run
            should_save = current_step > last_saved_step
            
            if should_save:
                version = getattr(self.args, 'version', 'version_0')
                output_dir = os.path.join('debug_outputs', self.args.exp_name, version)
                os.makedirs(output_dir, exist_ok=True)

                rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
                # rank = int(os.environ.get("RANK", 0))
                filename = os.path.join(output_dir, f'debug_data_epoch_{current_epoch}_step_{current_step}_rank{rank}.pt')
                print(f'\n--- SAVING DEBUG DATA (Epoch {current_epoch}, Step {current_step}): Writing batch and predictions to {filename} ---\n')

                # Move tensors to CPU before saving
                cpu_batch = batch.to('cpu')
                cpu_predictions = {k: v.to('cpu') for k, v in predictions.items()}

                torch.save({
                    'batch': cpu_batch,
                    'predictions': cpu_predictions
                }, filename)

                # Update the flag to the current step
                self._last_saved_debug_step = current_step
                
                # Also plot examples only when we save debug data (avoid excessive plotting)
                self.plot_examples(batch, predictions)

    def predict_step(self, batch: Any, dataloader_idx: int = 0) -> Any:
        """
        Prediction step, required by PyTorch Lightning's trainer.predict().
        """
        return self.forward(batch)

    @staticmethod
    def _get_std_for_loss(pred_value_tensor, std_value_representation=None):
        """
        Helper to get a valid standard deviation tensor for loss functions.
        If std_value_representation is None, returns ones (for unweighted loss).
        If std_value_representation is log_var, it converts to std.
        If std_value_representation is already std, it returns it.
        Adjust this based on how your decoders output uncertainty.
        """
        if std_value_representation is None:
            return torch.ones_like(pred_value_tensor)
        
        # Example: if your decoder outputs log_variance for numerical stability
        # actual_std = torch.exp(0.5 * std_value_representation)
        # return actual_std
        
        # For now, assume std_value_representation is already the actual std if not None
        return std_value_representation

    def on_validation_epoch_start(self):
        # Gradient tensors from the last training step stay resident until the
        # next optimizer.zero_grad().  Freeing them here reclaims enough GPU
        # memory to run the validation GAT scatter_add without OOM.
        for p in self.parameters():
            p.grad = None
        gc.collect()
        torch.cuda.empty_cache()

    def on_validation_epoch_end(self):
        pass

    # def configure_optimizers(self):
    #     optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    
    #     scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.4)
    
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": {
    #             "scheduler": scheduler,
    #             "interval": "epoch",  # or "step"
    #             "frequency": 1,
    #         }
    #     }


def print_grad(name):
    def hook(grad):
        print(f"Gradient for '{name}':")
        if grad is None:
            print("  -> Gradient is None. The tensor is detached from the graph.")
        else:
            print(f"  -> Shape: {grad.shape}")
            print(f"  -> Mean: {grad.mean():.2e}, Std: {grad.std():.2e}")
            print(f"  -> Max: {grad.max():.2e}, Min: {grad.min():.2e}")
            # Check for NaNs or Infs, which indicate instability
            if torch.isnan(grad).any():
                print("  -> WARNING: Gradient contains NaNs!")
            if torch.isinf(grad).any():
                print("  -> WARNING: Gradient contains Infs!")
    return hook
