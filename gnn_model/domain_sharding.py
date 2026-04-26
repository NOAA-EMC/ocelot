"""Mesh domain partitioning and halo-exchange utilities.

This module builds a deterministic mesh partition for each distributed rank,
extracts rank-local graph views, and exchanges boundary halo features between
neighboring ranks without gathering the full mesh state.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
from typing import Dict, Iterable, List, Optional

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch_geometric.data import HeteroData


def get_rank_world_size() -> tuple[int, int]:
    if dist.is_available() and dist.is_initialized():
        return int(dist.get_rank()), int(dist.get_world_size())

    rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID", "0")))
    world_size = int(os.environ.get("WORLD_SIZE", os.environ.get("SLURM_NTASKS", "1")))
    return rank, max(world_size, 1)


def _lat_lon_to_unit_xyz(mesh_lat_lon: torch.Tensor) -> torch.Tensor:
    lat = torch.deg2rad(mesh_lat_lon[:, 0].to(torch.float64))
    lon = torch.deg2rad(mesh_lat_lon[:, 1].to(torch.float64))
    cos_lat = torch.cos(lat)
    return torch.stack(
        [
            cos_lat * torch.cos(lon),
            cos_lat * torch.sin(lon),
            torch.sin(lat),
        ],
        dim=1,
    )


def _build_neighbors(num_nodes: int, edge_index: torch.Tensor) -> list[list[int]]:
    neighbors = [set() for _ in range(num_nodes)]
    src = edge_index[0].tolist()
    dst = edge_index[1].tolist()
    for send, recv in zip(src, dst):
        if send == recv:
            continue
        neighbors[send].add(recv)
        neighbors[recv].add(send)
    return [sorted(adj) for adj in neighbors]


def _select_seed_nodes(xyz: torch.Tensor, world_size: int) -> list[int]:
    if world_size <= 1:
        return [0]

    seeds = [int(torch.argmax(xyz[:, 2]).item())]
    min_dist = torch.full((xyz.size(0),), float("inf"), dtype=torch.float64)

    while len(seeds) < world_size:
        last_seed = xyz[seeds[-1]].unsqueeze(0)
        dist_to_last = torch.cdist(xyz, last_seed).squeeze(1)
        min_dist = torch.minimum(min_dist, dist_to_last)
        seeds.append(int(torch.argmax(min_dist).item()))

    return seeds


def _balanced_region_grow(
    neighbors: list[list[int]],
    xyz: torch.Tensor,
    world_size: int,
) -> torch.Tensor:
    num_nodes = xyz.size(0)
    if world_size <= 1 or num_nodes == 0:
        return torch.zeros((num_nodes,), dtype=torch.long)

    seeds = _select_seed_nodes(xyz, world_size)
    seed_xyz = xyz[seeds]
    seed_dist = torch.cdist(seed_xyz, xyz).cpu()

    target_sizes = [num_nodes // world_size for _ in range(world_size)]
    for rank in range(num_nodes % world_size):
        target_sizes[rank] += 1

    owner = torch.full((num_nodes,), -1, dtype=torch.long)
    frontiers = [deque() for _ in range(world_size)]
    counts = [0 for _ in range(world_size)]
    pending = num_nodes

    for rank, seed in enumerate(seeds):
        owner[seed] = rank
        frontiers[rank].append(seed)
        counts[rank] = 1
        pending -= 1

    def grow_one(rank: int) -> bool:
        while frontiers[rank]:
            node = frontiers[rank].popleft()
            candidates = [nbr for nbr in neighbors[node] if owner[nbr] < 0]
            if not candidates:
                continue

            best_neighbor = min(candidates, key=lambda idx: float(seed_dist[rank, idx]))
            owner[best_neighbor] = rank
            counts[rank] += 1
            frontiers[rank].append(node)
            frontiers[rank].append(best_neighbor)
            return True
        return False

    while pending > 0:
        progressed = False
        for rank in range(world_size):
            if counts[rank] >= target_sizes[rank]:
                continue
            if grow_one(rank):
                pending -= 1
                progressed = True

        if progressed:
            continue

        unassigned = torch.nonzero(owner < 0, as_tuple=False).flatten()
        if unassigned.numel() == 0:
            break

        for rank in range(world_size):
            if counts[rank] >= target_sizes[rank]:
                continue
            remaining = torch.nonzero(owner < 0, as_tuple=False).flatten()
            if remaining.numel() == 0:
                break
            nearest = remaining[torch.argmin(seed_dist[rank, remaining])]
            node = int(nearest.item())
            owner[node] = rank
            counts[rank] += 1
            frontiers[rank].append(node)
            pending -= 1

    return owner


def _expand_halo(
    owned_ids: torch.Tensor,
    neighbors: list[list[int]],
    halo_hops: int,
) -> torch.Tensor:
    if halo_hops <= 0 or owned_ids.numel() == 0:
        return owned_ids.new_empty((0,))

    expanded = set(int(idx) for idx in owned_ids.tolist())
    frontier = set(expanded)

    for _ in range(int(halo_hops)):
        next_frontier = set()
        for node in frontier:
            for neighbor in neighbors[node]:
                if neighbor in expanded:
                    continue
                expanded.add(neighbor)
                next_frontier.add(neighbor)
        frontier = next_frontier
        if not frontier:
            break

    halo = sorted(idx for idx in expanded if idx not in set(int(v) for v in owned_ids.tolist()))
    return torch.as_tensor(halo, dtype=torch.long)


def _subset_node_store(src_store, indices: torch.Tensor) -> dict:
    num_nodes = int(getattr(src_store, "num_nodes", 0) or 0)
    subset = {}
    for key, value in src_store.items():
        if torch.is_tensor(value) and value.dim() > 0 and value.size(0) == num_nodes:
            subset[key] = value.index_select(0, indices)
        else:
            subset[key] = value
    return subset


@dataclass(frozen=True)
class MeshPartitionSpec:
    rank: int
    world_size: int
    node_owner: torch.Tensor
    owned_global_ids: torch.Tensor
    local_global_ids: torch.Tensor
    global_to_local: torch.Tensor
    owned_node_count: int
    local_mesh_x: torch.Tensor
    local_mesh_pos: torch.Tensor
    local_mesh_edge_index: torch.Tensor
    local_mesh_edge_attr: torch.Tensor
    all_send_global_ids_by_rank: list[torch.Tensor]
    send_global_ids_by_dst: list[torch.Tensor]
    recv_global_ids_by_owner: Dict[int, torch.Tensor]

    @property
    def halo_node_count(self) -> int:
        return int(self.local_global_ids.numel() - self.owned_node_count)

    @property
    def send_global_ids(self) -> torch.Tensor:
        return self.all_send_global_ids_by_rank[self.rank]


class HaloExchange:
    """Boundary-only halo exchange that preserves autograd through the gather.

    Uses ``all_to_all_single`` so that each rank only sends/receives the exact
    number of boundary nodes its peers need (no padding to ``max_send_count``).
    """

    def __init__(self, spec: MeshPartitionSpec, device: torch.device, group=None):
        self.spec = spec
        self.device = device
        self.group = group
        self.rank = int(spec.rank)
        self.world_size = int(spec.world_size)
        self.owned_node_count = int(spec.owned_node_count)
        self.halo_node_count = int(spec.halo_node_count)

        # Per-destination local indices to send (concatenated in dst order).
        send_local_chunks: list[torch.Tensor] = []
        send_split_sizes: list[int] = []
        for dst_rank in range(self.world_size):
            ids = spec.send_global_ids_by_dst[dst_rank]
            if ids.numel() == 0 or dst_rank == self.rank:
                send_split_sizes.append(0)
                continue
            local_idx = spec.global_to_local.index_select(0, ids).to(
                device=device, dtype=torch.long
            )
            send_local_chunks.append(local_idx)
            send_split_sizes.append(int(local_idx.numel()))

        if send_local_chunks:
            self.send_local_indices = torch.cat(send_local_chunks, dim=0)
        else:
            self.send_local_indices = torch.empty((0,), device=device, dtype=torch.long)
        self.send_split_sizes = send_split_sizes

        # Recv counts (in owner order) and total recv length.  Halo nodes in
        # ``local_global_ids`` are arranged as: owned, then for each owner != self
        # in ascending order, that owner's halo block.  This matches the
        # ``all_to_all_single`` output layout exactly.
        recv_split_sizes: list[int] = []
        for owner in range(self.world_size):
            if owner == self.rank:
                recv_split_sizes.append(0)
                continue
            ids = spec.recv_global_ids_by_owner.get(owner)
            recv_split_sizes.append(0 if ids is None else int(ids.numel()))
        self.recv_split_sizes = recv_split_sizes
        self.total_recv = int(sum(recv_split_sizes))

    def exchange(self, local_state: torch.Tensor) -> torch.Tensor:
        if (
            self.world_size <= 1
            or not dist.is_available()
            or not dist.is_initialized()
        ):
            return local_state

        # NOTE: We must NOT early-return based on this rank's halo or send
        # counts being zero. ``all_to_all_single`` is a collective: every rank
        # has to call it the same number of times in the same order. If this
        # rank silently skipped a no-op exchange while peers expected one,
        # NCCL would deadlock and eventually time out.

        owned_state = local_state[: self.owned_node_count]
        feat_dim = local_state.size(-1)

        # IMPORTANT: Always build send_buffer via ``index_select`` (even when
        # empty) so the tensor is part of ``local_state``'s autograd graph.
        # If we used ``new_zeros`` for the empty case, autograd would not
        # create a backward node for ``_AlltoAllSingle`` on this rank, and
        # during backward this rank would skip the reverse collective while
        # peers issued it — deadlocking NCCL.
        send_buffer = local_state.index_select(0, self.send_local_indices)

        recv_buffer = local_state.new_empty((self.total_recv, feat_dim))
        recv_buffer = dist_nn.all_to_all_single(
            recv_buffer,
            send_buffer,
            output_split_sizes=self.recv_split_sizes,
            input_split_sizes=self.send_split_sizes,
            group=self.group,
        )

        # IMPORTANT: Always return via ``torch.cat`` so the autograd graph
        # topology is identical on every rank. If we conditionally returned
        # ``owned_state`` when ``recv_buffer`` was empty, ``recv_buffer``
        # would be disconnected from downstream computation on that rank,
        # the backward of ``all_to_all_single`` would be pruned, and other
        # ranks would deadlock waiting for the reverse collective.
        #
        # ALSO: even when we always concat, a rank whose downstream graph
        # never *reads* halo positions (e.g. its local mesh edges have no
        # src indices in the halo region) will receive a zero/None gradient
        # for ``recv_buffer`` and autograd may prune ``_AlltoAllSingle``'s
        # backward node on that rank only. The peers still run the reverse
        # collective and the SeqNums diverge → NCCL deadlock at the next
        # collective (often DDP's ALLREDUCE on the gradient bucket).
        # Tie ``owned_state`` to ``recv_buffer`` via a zero-weighted scalar
        # so the backward path through ``recv_buffer`` is always live.
        if self.total_recv > 0 and recv_buffer.requires_grad:
            sentinel = recv_buffer.sum() * 0.0
            owned_state = owned_state + sentinel
        return torch.cat([owned_state, recv_buffer], dim=0)


class DomainGraphSharder:
    """Builds rank-local mesh partitions and slices hetero-graphs accordingly."""

    def __init__(
        self,
        mesh_x: torch.Tensor,
        mesh_pos: torch.Tensor,
        mesh_edge_index: torch.Tensor,
        mesh_edge_attr: torch.Tensor,
        rank: int,
        world_size: int,
        halo_hops: int = 1,
    ):
        self.rank = int(rank)
        self.world_size = int(max(world_size, 1))
        self.halo_hops = int(max(halo_hops, 0))
        self.mesh_x = mesh_x.detach().cpu()
        self.mesh_pos = mesh_pos.detach().cpu()
        self.mesh_edge_index = mesh_edge_index.detach().cpu()
        self.mesh_edge_attr = mesh_edge_attr.detach().cpu()
        self.is_enabled = self.world_size > 1
        self.uses_synced_partition = False

        self.spec = self._build_partition_spec()

    def _resolve_node_owner(
        self,
        neighbors: list[list[int]],
        xyz: torch.Tensor,
    ) -> torch.Tensor:
        if not self.is_enabled:
            self.uses_synced_partition = False
            return torch.zeros((xyz.size(0),), dtype=torch.long)

        if not (dist.is_available() and dist.is_initialized()):
            self.uses_synced_partition = False
            return _balanced_region_grow(neighbors, xyz, self.world_size)

        if int(dist.get_world_size()) != self.world_size:
            self.uses_synced_partition = False
            return _balanced_region_grow(neighbors, xyz, self.world_size)

        num_nodes = int(xyz.size(0))
        backend = str(dist.get_backend())

        if backend == "nccl" and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            device = torch.device(
                "cuda",
                torch.cuda.current_device() if torch.cuda.is_initialized() else local_rank,
            )
            if self.rank == 0:
                node_owner = _balanced_region_grow(neighbors, xyz, self.world_size).to(device=device)
            else:
                node_owner = torch.empty((num_nodes,), device=device, dtype=torch.long)
            dist.broadcast(node_owner, src=0)
            node_owner = node_owner.cpu()
        else:
            payload = [
                _balanced_region_grow(neighbors, xyz, self.world_size).tolist()
                if self.rank == 0
                else None
            ]
            dist.broadcast_object_list(payload, src=0)
            node_owner = torch.as_tensor(payload[0], dtype=torch.long)

        self.uses_synced_partition = True
        return node_owner

    def _build_partition_spec(self) -> MeshPartitionSpec:
        num_nodes = int(self.mesh_x.size(0))
        xyz = _lat_lon_to_unit_xyz(self.mesh_pos)
        neighbors = _build_neighbors(num_nodes, self.mesh_edge_index)
        node_owner = self._resolve_node_owner(neighbors, xyz)

        owned_by_rank = [
            torch.nonzero(node_owner == rank, as_tuple=False).flatten()
            for rank in range(self.world_size)
        ]
        halo_by_rank = [_expand_halo(owned_ids, neighbors, self.halo_hops) for owned_ids in owned_by_rank]

        recv_global_ids_by_owner: Dict[int, torch.Tensor] = {}
        local_halo_parts = []
        for owner in range(self.world_size):
            if owner == self.rank:
                continue
            halo_ids = halo_by_rank[self.rank]
            if halo_ids.numel() == 0:
                owner_halo = halo_ids
            else:
                owner_mask = node_owner.index_select(0, halo_ids) == owner
                owner_halo = halo_ids.index_select(0, torch.nonzero(owner_mask, as_tuple=False).flatten())
            recv_global_ids_by_owner[owner] = owner_halo
            if owner_halo.numel() > 0:
                local_halo_parts.append(owner_halo)

        owned_global_ids = owned_by_rank[self.rank]
        local_global_ids = owned_global_ids
        if local_halo_parts:
            local_global_ids = torch.cat([owned_global_ids] + local_halo_parts, dim=0)

        global_to_local = torch.full((num_nodes,), -1, dtype=torch.long)
        global_to_local[local_global_ids] = torch.arange(local_global_ids.numel(), dtype=torch.long)

        src_local = global_to_local.index_select(0, self.mesh_edge_index[0])
        dst_local = global_to_local.index_select(0, self.mesh_edge_index[1])
        dst_is_owned = node_owner.index_select(0, self.mesh_edge_index[1]) == self.rank
        keep_edge = (src_local >= 0) & (dst_local >= 0) & dst_is_owned

        local_mesh_edge_index = torch.stack(
            [
                src_local[keep_edge],
                dst_local[keep_edge],
            ],
            dim=0,
        )
        local_mesh_edge_attr = self.mesh_edge_attr[keep_edge]

        all_send_global_ids_by_rank = []
        for src_rank in range(self.world_size):
            owned_ids = owned_by_rank[src_rank]
            send_parts = []
            for dst_rank in range(self.world_size):
                if dst_rank == src_rank:
                    continue
                halo_ids = halo_by_rank[dst_rank]
                if halo_ids.numel() == 0:
                    continue
                src_mask = node_owner.index_select(0, halo_ids) == src_rank
                candidate_ids = halo_ids.index_select(0, torch.nonzero(src_mask, as_tuple=False).flatten())
                if candidate_ids.numel() == 0:
                    continue
                send_parts.append(candidate_ids)
            if send_parts:
                send_ids = torch.unique(torch.cat(send_parts, dim=0), sorted=True)
            else:
                send_ids = owned_ids.new_empty((0,))
            all_send_global_ids_by_rank.append(send_ids)

        # Per-destination send IDs from THIS rank: what we must send to each dst_rank.
        # This mirrors recv_global_ids_by_owner observed by dst_rank: it is the subset
        # of dst_rank's halo that this rank owns.
        send_global_ids_by_dst: list[torch.Tensor] = []
        for dst_rank in range(self.world_size):
            if dst_rank == self.rank:
                send_global_ids_by_dst.append(torch.empty((0,), dtype=torch.long))
                continue
            halo_ids = halo_by_rank[dst_rank]
            if halo_ids.numel() == 0:
                send_global_ids_by_dst.append(torch.empty((0,), dtype=torch.long))
                continue
            mask = node_owner.index_select(0, halo_ids) == self.rank
            send_global_ids_by_dst.append(
                halo_ids.index_select(0, torch.nonzero(mask, as_tuple=False).flatten())
            )

        return MeshPartitionSpec(
            rank=self.rank,
            world_size=self.world_size,
            node_owner=node_owner,
            owned_global_ids=owned_global_ids,
            local_global_ids=local_global_ids,
            global_to_local=global_to_local,
            owned_node_count=int(owned_global_ids.numel()),
            local_mesh_x=self.mesh_x.index_select(0, local_global_ids),
            local_mesh_pos=self.mesh_pos.index_select(0, local_global_ids),
            local_mesh_edge_index=local_mesh_edge_index,
            local_mesh_edge_attr=local_mesh_edge_attr,
            all_send_global_ids_by_rank=all_send_global_ids_by_rank,
            send_global_ids_by_dst=send_global_ids_by_dst,
            recv_global_ids_by_owner=recv_global_ids_by_owner,
        )

    def build_halo_exchange(self, device: torch.device, group=None) -> HaloExchange:
        return HaloExchange(self.spec, device=device, group=group)

    def shard_graph(self, data: HeteroData) -> HeteroData:
        if not self.is_enabled:
            return data

        out = HeteroData()
        mesh_store = out["mesh"]
        mesh_store.x = self.spec.local_mesh_x.clone()
        mesh_store.pos = self.spec.local_mesh_pos.clone()
        mesh_store.global_ids = self.spec.local_global_ids.clone()
        mesh_store.owned_node_count = torch.tensor(self.spec.owned_node_count, dtype=torch.long)
        mesh_store.halo_node_count = torch.tensor(self.spec.halo_node_count, dtype=torch.long)

        out["mesh", "to", "mesh"].edge_index = self.spec.local_mesh_edge_index.clone()
        out["mesh", "to", "mesh"].edge_attr = self.spec.local_mesh_edge_attr.clone()

        for node_type in data.node_types:
            if node_type == "mesh":
                continue

            if node_type.endswith("_input"):
                self._shard_input_nodes(data, out, node_type)
                continue

            if "_target" in node_type:
                self._shard_target_nodes(data, out, node_type)
                continue

            src_store = data[node_type]
            node_indices = torch.arange(src_store.num_nodes, dtype=torch.long)
            for key, value in _subset_node_store(src_store, node_indices).items():
                out[node_type][key] = value

        for attr_name in ("bin_name", "init_time", "input_time"):
            if hasattr(data, attr_name):
                setattr(out, attr_name, getattr(data, attr_name))

        return out

    def _shard_input_nodes(self, data: HeteroData, out: HeteroData, node_type: str) -> None:
        edge_type = (node_type, "to", "mesh")
        edge_index = data[edge_type].edge_index
        edge_attr = data[edge_type].edge_attr

        if edge_index.numel() == 0 or data[node_type].num_nodes == 0:
            empty_index = torch.empty((0,), dtype=torch.long)
            for key, value in _subset_node_store(data[node_type], empty_index).items():
                out[node_type][key] = value
            out[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            out[edge_type].edge_attr = edge_attr[:0]
            return

        dst_owner = self.spec.node_owner.index_select(0, edge_index[1])
        keep_edge = dst_owner == self.rank
        kept_edges = torch.nonzero(keep_edge, as_tuple=False).flatten()
        if kept_edges.numel() == 0:
            empty_index = torch.empty((0,), dtype=torch.long)
            for key, value in _subset_node_store(data[node_type], empty_index).items():
                out[node_type][key] = value
            out[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            out[edge_type].edge_attr = edge_attr[:0]
            return

        kept_src = edge_index[0].index_select(0, kept_edges)
        kept_dst = edge_index[1].index_select(0, kept_edges)
        unique_src, local_src = torch.unique(kept_src, sorted=True, return_inverse=True)
        local_dst = self.spec.global_to_local.index_select(0, kept_dst)

        for key, value in _subset_node_store(data[node_type], unique_src).items():
            out[node_type][key] = value

        out[edge_type].edge_index = torch.stack([local_src, local_dst], dim=0)
        out[edge_type].edge_attr = edge_attr.index_select(0, kept_edges)

    def _shard_target_nodes(self, data: HeteroData, out: HeteroData, node_type: str) -> None:
        edge_type = ("mesh", "to", node_type)
        edge_index = data[edge_type].edge_index
        edge_attr = data[edge_type].edge_attr
        num_targets = int(data[node_type].num_nodes)

        if edge_index.numel() == 0 or num_targets == 0:
            empty_index = torch.empty((0,), dtype=torch.long)
            for key, value in _subset_node_store(data[node_type], empty_index).items():
                out[node_type][key] = value
            out[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            out[edge_type].edge_attr = edge_attr[:0]
            return

        src_owner = self.spec.node_owner.index_select(0, edge_index[0])
        target_has_edge = torch.bincount(edge_index[1], minlength=num_targets) > 0
        assigned_owner = torch.zeros((num_targets,), dtype=torch.long)
        best_count = torch.full((num_targets,), -1, dtype=torch.long)

        for rank in range(self.world_size):
            rank_edge_mask = src_owner == rank
            if not rank_edge_mask.any():
                continue
            rank_counts = torch.bincount(edge_index[1][rank_edge_mask], minlength=num_targets)
            better = rank_counts > best_count
            assigned_owner[better] = rank
            best_count[better] = rank_counts[better]

        keep_target = (assigned_owner == self.rank) & target_has_edge
        target_indices = torch.nonzero(keep_target, as_tuple=False).flatten()

        for key, value in _subset_node_store(data[node_type], target_indices).items():
            out[node_type][key] = value

        if target_indices.numel() == 0:
            out[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)
            out[edge_type].edge_attr = edge_attr[:0]
            return

        keep_edge = keep_target.index_select(0, edge_index[1])
        kept_edges = torch.nonzero(keep_edge, as_tuple=False).flatten()
        kept_src = edge_index[0].index_select(0, kept_edges)
        kept_dst = edge_index[1].index_select(0, kept_edges)

        local_src = self.spec.global_to_local.index_select(0, kept_src)
        local_dst = torch.cumsum(keep_target.to(torch.long), dim=0).index_select(0, kept_dst) - 1

        out[edge_type].edge_index = torch.stack([local_src, local_dst], dim=0)
        out[edge_type].edge_attr = edge_attr.index_select(0, kept_edges)