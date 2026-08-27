"""Bipartite graph attention layers used for observation-mesh message passing.

Author: Azadeh Gholoubi
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from torch_geometric.nn import GATv2Conv
from torch_geometric.data import HeteroData

from ocelot.configs.model_config import GatCoderConfig


class BipartiteGAT(nn.Module):
    """
    Multi-layer GATv2 for bipartite edges (src -> dst).
    Mirrors InteractionNet's interface: forward(send_rep, rec_rep, edge_rep, edge_index)

    Args:
      coder_config: instance of CoderConfig containing all necessary hyperparameters.
    """

    def __init__(self, coder_config: GatCoderConfig):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropout = nn.Dropout(coder_config.dropout)

        # For very large bipartite graphs (e.g., mesh -> satellite targets),
        # PyG's GATv2Conv builds per-edge attention tensors that can exceed GPU memory.
        # We mitigate this by chunking over destination nodes (dst) so peak edge
        # attention memory is bounded. Chunking preserves exact results as long as
        # all incoming edges for a dst node are processed together.
        self.dst_chunk_size = coder_config.dst_chunk_size
        self.dst_chunk_threshold = coder_config.dst_chunk_threshold
        self.use_activation_checkpointing = coder_config.use_activation_checkpointing

        in_src = coder_config.send_dim
        in_dst = coder_config.rec_dim
        for li in range(coder_config.layers):
            conv = GATv2Conv(
                in_channels=(in_src, in_dst),   # bipartite (src,dst)
                out_channels=coder_config.hidden_dim,
                heads=coder_config.heads,
                dropout=coder_config.dropout,
                concat=False,                   # shape = [N_dst, hidden_dim]
                edge_dim=coder_config.edge_dim,              # use edge_attr in attention if provided
                share_weights=False,
                add_self_loops=False,           # we are bipartite; no self loops
            )
            self.layers.append(conv)
            self.norms.append(nn.LayerNorm(coder_config.hidden_dim))

            # after first layer, both sides live in hidden_dim
            in_src = coder_config.hidden_dim
            in_dst = coder_config.hidden_dim

        # if the very first dst dim != hidden_dim, build a projection for residual
        self.res_proj = (
            nn.Linear(coder_config.rec_dim, coder_config.hidden_dim)
            if coder_config.rec_dim != coder_config.hidden_dim else nn.Identity()
        )

    @property
    def edge_index(self):
        # kept only for API parity with your InteractionNet usage
        # where you set encoder.edge_index = ...
        return getattr(self, "_edge_index", None)

    @edge_index.setter
    def edge_index(self, ei):
        self._edge_index = ei

    def forward(
        self,
        send_rep: torch.Tensor,   # [N_src, F_src]
        rec_rep: torch.Tensor,    # [N_dst, F_dst]
        edge_rep: Optional[torch.Tensor] = None,  # [E, edge_dim] or None
        edge_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if edge_index is None:
            edge_index = self._edge_index

        x_src, x_dst = send_rep, rec_rep

        # Fast path: small dst, do a single call per layer.
        N_dst = int(x_dst.size(0))
        if (
            edge_index is None
            or (not torch.is_tensor(edge_index))
            or edge_index.numel() == 0
            or N_dst == 0
        ):
            # No edges or empty dst: return residual-projected dst in the expected dim.
            return self.res_proj(x_dst)

        E = int(edge_index.size(1))

        # Heuristic chunk size selection.
        # If dst_chunk_size is not provided, use an auto chunk size for huge dst.
        chunk_size = self.dst_chunk_size
        if chunk_size is None and N_dst >= self.dst_chunk_threshold:
            chunk_size = 10_000

        use_chunking = chunk_size is not None and N_dst >= self.dst_chunk_threshold

        if not use_chunking:
            res0 = self.res_proj(x_dst)
            for conv, norm in zip(self.layers, self.norms):
                x_dst_new = conv((x_src, x_dst), edge_index, edge_rep)
                x_dst_new = norm(x_dst_new + res0)      # pre-norm residual
                x_dst_new = self.dropout(x_dst_new)
                x_src, x_dst, res0 = x_src, x_dst_new, x_dst_new
            return x_dst

        # Chunked path: split over destination nodes.
        # Try to detect the common KNN layout where edges are grouped by dst
        # with a fixed number of neighbors per dst: dst = [0..0, 1..1, 2..2, ...].
        grouped_k: Optional[int] = None
        if N_dst > 0 and E > 0 and (E % N_dst) == 0:
            k = E // N_dst
            # Only treat small k as a likely KNN case (typical: 3-8).
            if 1 <= k <= 32:
                dst = edge_index[1]
                ok = True
                # Check first few dst blocks
                blocks_to_check = min(N_dst, 4)
                for i in range(blocks_to_check):
                    blk = dst[i * k:(i + 1) * k]
                    if blk.numel() != k or not bool((blk == i).all()):
                        ok = False
                        break
                # Check the last dst block
                if ok:
                    last = N_dst - 1
                    blk = dst[last * k:(last + 1) * k]
                    if blk.numel() != k or not bool((blk == last).all()):
                        ok = False
                if ok:
                    grouped_k = int(k)

        for li, (conv, norm) in enumerate(zip(self.layers, self.norms)):
            out_chunks = []

            # Process contiguous dst ranges.
            for dst_start in range(0, N_dst, int(chunk_size)):
                dst_end = min(N_dst, dst_start + int(chunk_size))
                x_dst_chunk = x_dst[dst_start:dst_end]

                # Residual for this layer.
                if li == 0:
                    res_chunk = self.res_proj(x_dst_chunk)
                else:
                    res_chunk = x_dst_chunk

                # Select edges for this dst chunk.
                if grouped_k is not None:
                    e0 = dst_start * grouped_k
                    e1 = dst_end * grouped_k
                    ei = edge_index[:, e0:e1]
                    ea = edge_rep[e0:e1] if edge_rep is not None else None

                    # Reindex dst into [0, chunk_N)
                    if dst_start != 0:
                        ei = ei.clone()
                        ei[1] = ei[1] - dst_start
                else:
                    dst = edge_index[1]
                    mask = (dst >= dst_start) & (dst < dst_end)
                    ei = edge_index[:, mask]
                    ea = edge_rep[mask] if edge_rep is not None else None
                    if dst_start != 0:
                        ei = ei.clone()
                        ei[1] = ei[1] - dst_start

                if self.training and self.use_activation_checkpointing:
                    # `checkpoint.checkpoint` only accepts Tensor inputs; `edge_attr` may be None.
                    if ea is None:
                        def _f(xs: torch.Tensor, xd: torch.Tensor, eix: torch.Tensor) -> torch.Tensor:
                            return conv((xs, xd), eix, None)

                        out_chunk = checkpoint.checkpoint(
                            _f, x_src, x_dst_chunk, ei, use_reentrant=False
                        )
                    else:
                        def _f(xs: torch.Tensor, xd: torch.Tensor, eix: torch.Tensor, ear: torch.Tensor) -> torch.Tensor:
                            return conv((xs, xd), eix, ear)

                        # use_reentrant=False supports non-reentrant checkpointing and plays better
                        # with complex modules; edge_index/edge_attr are treated as tensor inputs.
                        out_chunk = checkpoint.checkpoint(
                            _f, x_src, x_dst_chunk, ei, ea, use_reentrant=False
                        )
                else:
                    out_chunk = conv((x_src, x_dst_chunk), ei, ea)
                out_chunk = norm(out_chunk + res_chunk)
                out_chunk = self.dropout(out_chunk)
                out_chunks.append(out_chunk)

            x_dst = torch.cat(out_chunks, dim=0)

        return x_dst

    def edge_features(
        self,
        data: HeteroData,
        edge_type,
        edge_index: torch.Tensor,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Returns per-edge features in bipartite_edge_attr_dim (raw spatial edge_attr)."""
        E = int(edge_index.size(1))

        # Debug printing: show whether we used real edge_attr or fell back to zeros.
        # Gated by verbose + global_zero and printed at most once per (edge_type, reason).
        def _maybe_print(reason: str, edge_rep_tensor: torch.Tensor | None = None) -> None:
            if not getattr(self, "verbose", False):
                return
            if not self._is_global_zero_safe():
                return
            if not hasattr(self, "_edge_attr_debug_seen") or self._edge_attr_debug_seen is None:
                self._edge_attr_debug_seen = set()
            key = (tuple(edge_type) if isinstance(edge_type, (list, tuple)) else str(edge_type), str(reason))
            if key in self._edge_attr_debug_seen:
                return
            self._edge_attr_debug_seen.add(key)

            msg = f"[EDGE_ATTR] edge_type={edge_type} E={E} used={'edge_attr' if reason == 'ok' else 'zeros'} reason={reason}"
            if edge_rep_tensor is not None and torch.is_tensor(edge_rep_tensor) and edge_rep_tensor.numel() > 0:
                try:
                    t = edge_rep_tensor.detach()
                    mean_v = t.mean().item()
                    std_v = t.std(unbiased=False).item()
                    min_v = t.min().item()
                    max_v = t.max().item()
                    msg += (
                        f" edge_attr_shape={tuple(t.shape)} "
                        f"mean={mean_v:.4g} std={std_v:.4g} min={min_v:.4g} max={max_v:.4g}"
                    )
                except Exception:
                    msg += f" edge_attr_shape={tuple(edge_rep_tensor.shape)}"
            print(msg)

        if not self.use_bipartite_edge_attr:
            _maybe_print("disabled")
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        edge_rep = None
        try:
            if "edge_attr" in data[edge_type]:
                edge_rep = data[edge_type].edge_attr
        except Exception:
            edge_rep = None

        if edge_rep is None:
            _maybe_print("missing")
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        if torch.is_tensor(edge_rep) and edge_rep.numel() == 0:
            _maybe_print("empty")
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        if not torch.is_tensor(edge_rep):
            _maybe_print("non_tensor")
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        if edge_rep.size(0) != E:
            _maybe_print(f"edge_count_mismatch(edge_attr={int(edge_rep.size(0))})")
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        edge_rep = edge_rep.to(device=device, dtype=dtype)
        edge_rep = self._coerce_edge_attr_dim(edge_rep, self.bipartite_edge_attr_dim)

        if edge_rep.size(-1) != self.bipartite_edge_attr_dim:
            _maybe_print(f"dim_mismatch(edge_attr={int(edge_rep.size(-1))})", edge_rep)
            return torch.zeros((E, self.bipartite_edge_attr_dim), device=device, dtype=dtype)

        _maybe_print("ok", edge_rep)
        return edge_rep

    def _coerce_edge_attr_dim(self, edge_attr: Optional[torch.Tensor], dim: int) -> Optional[torch.Tensor]:
        if edge_attr is None:
            return edge_attr
        if edge_attr.dim() == 1:
            edge_attr = edge_attr.unsqueeze(-1)
        if edge_attr.size(-1) == dim:
            return edge_attr
        if edge_attr.size(-1) < dim:
            pad = dim - edge_attr.size(-1)
            return torch.cat(
                [
                    edge_attr,
                    torch.zeros(
                        edge_attr.size(0),
                        pad,
                        device=edge_attr.device,
                        dtype=edge_attr.dtype,
                    ),
                ],
                dim=-1,
            )
        return edge_attr[:, :dim]
