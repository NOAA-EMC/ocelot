"""Lightning data loading and graph-building utilities for Ocelot GNN training.

This module prepares time-binned observation samples, converts them into
heterogeneous graph inputs, and manages train, validation, and prediction data
pipelines through a PyTorch Lightning data module.

Author: Azadeh Gholoubi
"""

import gc
import glob
import os
import json
import hashlib
import time
from collections.abc import Callable
import importlib
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import zarr
from zarr.storage import LRUStoreCache
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
from process_timeseries import extract_features, organize_bins_times
from create_mesh_graph_global import obs_mesh_conn
from domain_sharding import DomainGraphSharder, get_rank_world_size

# Number of columns for latitude and longitude in metadata
LAT_LON_COLUMNS = 2


def _resolve_zarr_path(data_path: str, zname: str, start_date: str) -> tuple[str, bool]:
    base_name = zname[:-5] if zname.endswith(".zarr") else zname
    direct_candidates = [os.path.join(data_path, zname if zname.endswith(".zarr") else f"{zname}.zarr")]

    year = str(start_date).split("-")[0]
    year_tagged_path = os.path.join(data_path, f"{base_name}_{year}.zarr")
    direct_candidates.append(year_tagged_path)

    for candidate in direct_candidates:
        if os.path.isdir(candidate):
            return candidate, False

    matches = sorted(glob.glob(os.path.join(data_path, f"{base_name}_*.zarr")))
    available_matches = [path for path in matches if os.path.isdir(path)]
    if len(available_matches) == 1:
        return available_matches[0], True

    available_match_names = sorted(os.path.basename(path) for path in available_matches)
    raise FileNotFoundError(
        f"Zarr not found for '{zname}' under {data_path}. "
        f"Available matches: {available_match_names or 'none'}"
    )


def _to_unix_seconds(t):
    """Best-effort conversion of a pandas/py datetime-like to unix seconds (UTC)."""
    if t is None:
        return None
    try:
        ts = pd.Timestamp(t)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return int(ts.timestamp())
    except Exception:
        try:
            return int(t)
        except Exception:
            return None


def _t32(x):
    return x.float() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.float32)


def _t64(x):
    return x.long() if torch.is_tensor(x) else torch.as_tensor(x, dtype=torch.long)


# -------------------------
# Dataset per bin
# -------------------------
class BinDataset(Dataset):
    def __init__(
        self,
        bin_names,
        data_summary,
        zarr_store,
        create_graph_fn,
        observation_config,
        feature_stats=None,
        require_targets=True,
        include_persistence_inputs=False,
        tag="TRAIN",
        verbose: bool = False,
        graph_cache_path_fn: Callable[[str], str] | None = None,
        graph_cache_read: bool = False,
        graph_cache_write: bool = False,
    ):
        self.bin_names = list(bin_names) if bin_names is not None else []
        self.data_summary = data_summary
        self.z = zarr_store
        self.create_graph_fn = create_graph_fn
        self.observation_config = observation_config
        self.feature_stats = feature_stats
        self.require_targets = require_targets
        self.include_persistence_inputs = bool(include_persistence_inputs)
        self.tag = tag
        self.verbose = bool(verbose)
        self.graph_cache_path_fn = graph_cache_path_fn
        self.graph_cache_read = bool(graph_cache_read)
        self.graph_cache_write = bool(graph_cache_write)

    def __len__(self):
        return len(self.bin_names)

    def __getitem__(self, idx):
        bin_name = self.bin_names[idx]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if self.verbose and rank == 0 and idx == 0:
            print(f"[Rank {rank}] [{self.tag}] fetching {bin_name} ... ds_id={id(self)} sum_id={id(self.data_summary)}")

        cache_path = self.graph_cache_path_fn(bin_name) if self.graph_cache_path_fn is not None else None
        if cache_path and self.graph_cache_read and os.path.exists(cache_path):
            try:
                graph_data = torch.load(cache_path, map_location="cpu", weights_only=False)
                graph_data.bin_name = bin_name
                return graph_data
            except Exception as e:
                if self.verbose and rank == 0:
                    print(f"[Rank {rank}] [{self.tag}] WARNING failed to load graph cache {cache_path}: {e}")
                try:
                    os.remove(cache_path)
                except OSError:
                    pass

        try:
            out = extract_features(
                self.z,
                self.data_summary,
                bin_name,
                self.observation_config,
                feature_stats=self.feature_stats,
                require_targets=self.require_targets,
                verbose=self.verbose,
                include_persistence_inputs=self.include_persistence_inputs,
            )
            bin_data = out[bin_name]
            graph_data = self.create_graph_fn(bin_data)
            graph_data.bin_name = bin_name

            # Attach bin-level timing metadata for downstream diagnostics (e.g., val_csv outputs).
            # - init_time: start of the target window (forecast init / cycle time)
            # - input_time: start of the input window
            init_time_unix = None
            input_time_unix = None
            try:
                for _obs_type, inst_dict in (bin_data or {}).items():
                    if not isinstance(inst_dict, dict):
                        continue
                    for _inst_name, data_summary_bin in inst_dict.items():
                        if not isinstance(data_summary_bin, dict):
                            continue

                        if input_time_unix is None and "input_time" in data_summary_bin:
                            input_time_unix = _to_unix_seconds(data_summary_bin.get("input_time"))

                        if init_time_unix is None:
                            target_times = data_summary_bin.get("target_times")
                            if isinstance(target_times, (list, tuple)) and len(target_times) > 0:
                                init_time_unix = _to_unix_seconds(target_times[0])

                        if init_time_unix is not None and input_time_unix is not None:
                            raise StopIteration
            except StopIteration:
                pass
            except Exception:
                init_time_unix = None
                input_time_unix = None

            graph_data.init_time = _t64(int(init_time_unix) if init_time_unix is not None else -1)
            graph_data.input_time = _t64(int(input_time_unix) if input_time_unix is not None else -1)

            if cache_path and self.graph_cache_write:
                try:
                    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
                    tmp_path = f"{cache_path}.tmp.{os.getpid()}.{idx}"
                    torch.save(graph_data, tmp_path)
                    os.replace(tmp_path, cache_path)
                except Exception as e:
                    if self.verbose and rank == 0:
                        print(f"[Rank {rank}] [{self.tag}] WARNING failed to write graph cache {cache_path}: {e}")
                    try:
                        if 'tmp_path' in locals() and os.path.exists(tmp_path):
                            os.remove(tmp_path)
                    except OSError:
                        pass

            return graph_data
        except Exception as e:
            print(f"[Rank {rank}] [{self.tag}] ERROR processing {bin_name}: {e}")
            raise


# -------------------------
# DataModule
# -------------------------
class GNNDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_path,
        start_date,
        end_date,
        observation_config,
        mesh_structure,
        batch_size=1,
        num_neighbors=3,
        feature_stats=None,
        latent_step_hours=12,       # latent rollout support
        window_size="12h",          # binning window
        train_val_split_ratio=0.9,  # Default fallback, should be passed from training script
        cache_val_windows: bool = False,
        val_cache_max_entries: int = 16,
        prediction_mode=False,
        require_targets=None,
        verbose: bool = False,
        parallelization_strategy: str = "replicated",
        domain_halo_hops: int = 1,
        data_loader_seed: int = 0,
        zarr_cache_max_size_bytes: int = int(2e9),
        train_num_workers: int = 4,
        val_num_workers: int = 4,
        predict_num_workers: int = 1,
        dataloader_prefetch_factor: int = 4,
        pin_memory: bool | None = None,
        graph_cache_dir: str | None = None,
        graph_cache_read: bool = False,
        graph_cache_write: bool = False,
        precompute_graph_cache: bool = False,
        **kwargs,
    ):
        super().__init__()

        # Normalize to int so Lightning hparams merge is stable across module/datamodule.
        latent_step_hours = int(latent_step_hours) if latent_step_hours is not None else None
        self.save_hyperparameters()
        self.prediction_mode = bool(prediction_mode)
        self.include_persistence_inputs = bool(prediction_mode)

        # If require_targets not specified, default based on prediction_mode
        # prediction_mode=True → require_targets=False (inference)
        # prediction_mode=False → require_targets=True (training/validation)
        if require_targets is None:
            self.require_targets = not prediction_mode
        else:
            self.require_targets = require_targets
        print(f"[DataModule] prediction_mode={prediction_mode}, require_targets={self.require_targets}")

        # Optional cache for validation summaries keyed by (val_start,val_end)
        self._cache_val_windows = bool(cache_val_windows)
        self._val_cache_max_entries = int(val_cache_max_entries)
        self._val_cache: dict[tuple[pd.Timestamp, pd.Timestamp], tuple[dict, list[str]]] = {}
        self._val_cache_lru: list[tuple[pd.Timestamp, pd.Timestamp]] = []

        # On-disk cache for expensive summary builds (DDP-safe).
        # NOTE: SLURM_JOB_ID is intentionally omitted from the directory so that
        # the cache is reused across jobs (the file name itself is a hash of
        # the window dates, observation_config, etc., so it's still correct).
        submit_dir = os.environ.get("SLURM_SUBMIT_DIR") or os.getcwd()
        self._summary_cache_dir = os.path.join(submit_dir, ".summary_cache")

        # Keep as hparam for downstream access
        self.hparams.verbose = bool(verbose)
        self.hparams.pin_memory = torch.cuda.is_available() if pin_memory is None else bool(pin_memory)
        self.hparams.zarr_cache_max_size_bytes = int(zarr_cache_max_size_bytes)
        self.hparams.train_num_workers = max(0, int(train_num_workers))
        self.hparams.val_num_workers = max(0, int(val_num_workers))
        self.hparams.predict_num_workers = max(0, int(predict_num_workers))
        self.hparams.dataloader_prefetch_factor = max(1, int(dataloader_prefetch_factor))
        self.hparams.graph_cache_dir = None if graph_cache_dir is None else str(graph_cache_dir)
        self.hparams.graph_cache_read = bool(graph_cache_read or precompute_graph_cache)
        self.hparams.graph_cache_write = bool(graph_cache_write or precompute_graph_cache)
        self.hparams.precompute_graph_cache = bool(precompute_graph_cache)

        self.mesh_structure = mesh_structure
        # Lazily-built bidirectional fp16 mesh edge tensors (see _create_graph_structure).
        self._m2m_bidir_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self.feature_stats = feature_stats
        self.parallelization_strategy = str(parallelization_strategy)
        self.domain_halo_hops = int(domain_halo_hops)
        self.data_loader_seed = int(data_loader_seed)
        self.domain_sharder: DomainGraphSharder | None = None

        # Zarr handles (stable across window changes)
        self.z = None

        # Separate train/val summaries + bin name lists
        self.train_data_summary = None
        self.val_data_summary = None
        self.train_bin_names = []
        self.val_bin_names = []

        # Version counters (for debugging staleness)
        self._train_version = 0
        self._val_version = 0

        self._precomputed_graph_cache_keys: set[tuple[str, int]] = set()

        # If callbacks want separate windows, they will set these:
        # Default: create non-overlapping train/val split to prevent data leakage
        # Use split ratio passed from training script for consistency
        total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        train_days = int(total_days * train_val_split_ratio)

        default_train_start = pd.to_datetime(start_date)
        default_train_end = default_train_start + pd.Timedelta(days=train_days)
        default_val_start = default_train_end  # Validation starts where training ends
        default_val_end = pd.to_datetime(end_date)

        self.hparams.train_start = pd.to_datetime(kwargs.get("train_start", default_train_start))
        self.hparams.train_end = pd.to_datetime(kwargs.get("train_end", default_train_end))
        self.hparams.val_start = pd.to_datetime(kwargs.get("val_start", default_val_start))
        self.hparams.val_end = pd.to_datetime(kwargs.get("val_end", default_val_end))

        # In prediction mode, use all data (no split)
        if prediction_mode:
            self.hparams.train_start = pd.to_datetime(start_date)
            self.hparams.train_end = pd.to_datetime(end_date)
            self.hparams.val_start = pd.to_datetime(start_date)
            self.hparams.val_end = pd.to_datetime(end_date)
            print(f"[DataModule] Prediction mode: Using entire date range {start_date} to {end_date}")

        # Validate no overlap between train and validation windows to prevent data leakage (training mode only)
        if not prediction_mode and self.hparams.train_end > self.hparams.val_start:
            raise ValueError(
                f"Data leakage detected! Training window ({self.hparams.train_start} to {self.hparams.train_end}) "
                f"overlaps with validation window ({self.hparams.val_start} to {self.hparams.val_end}). "
                f"Ensure train_end <= val_start for proper temporal split."
            )

        # Log the train/val split for transparency
        train_days = (self.hparams.train_end - self.hparams.train_start).days
        val_days = (self.hparams.val_end - self.hparams.val_start).days
        total_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
        denom_days = total_days if total_days != 0 else 1
        print(
            f"[DataModule] Train/Val Split - Train: {train_days} days ({train_days / denom_days * 100:.1f}%), "
            f"Val: {val_days} days ({val_days / denom_days * 100:.1f}%)"
        )
        print(f"[DataModule] Train window: {self.hparams.train_start.date()} to {self.hparams.train_end.date()}")
        print(f"[DataModule] Val window:   {self.hparams.val_start.date()} to {self.hparams.val_end.date()}")

        # Ensure latent_step_hours has a valid value
        if self.hparams.latent_step_hours is None:
            window_hours = int(self.hparams.window_size.replace('h', ''))
            self.hparams.latent_step_hours = window_hours

    def _ddp_info(self) -> tuple[bool, int]:
        is_ddp = bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
        rank = dist.get_rank() if is_ddp else 0
        return is_ddp, int(rank)

    def _is_verbose(self) -> bool:
        return bool(getattr(self.hparams, "verbose", False))

    def _summary_cache_path(self, kind: str, start_dt, end_dt, require_targets: bool) -> str:
        payload = {
            # Increment when binning semantics / summary structure changes.
            # This forces rebuild instead of reusing an incompatible cached summary.
            "summary_version": 3,
            "kind": kind,
            "start": str(pd.to_datetime(start_dt)),
            "end": str(pd.to_datetime(end_dt)),
            "window_size": str(getattr(self.hparams, "window_size", "")),
            "latent_step_hours": int(getattr(self.hparams, "latent_step_hours", 0) or 0),
            "require_targets": bool(require_targets),
            "observation_config": getattr(self.hparams, "observation_config", None),
            "pipeline": getattr(self.hparams, "pipeline", None),
        }
        s = json.dumps(payload, sort_keys=True, default=str)
        h = hashlib.blake2b(s.encode("utf-8"), digest_size=16).hexdigest()
        os.makedirs(self._summary_cache_dir, exist_ok=True)
        return os.path.join(self._summary_cache_dir, f"{kind}_{h}.pt")

    def _load_or_build_summary(self, kind: str, start_dt, end_dt, require_targets: bool) -> tuple[dict, list[str]]:
        is_ddp, rank = self._ddp_info()
        cache_path = self._summary_cache_path(kind, start_dt, end_dt, require_targets=require_targets)
        verbose = self._is_verbose()

        def _build() -> tuple[dict, list[str]]:
            data_summary = organize_bins_times(
                self.z,
                start_dt,
                end_dt,
                self.hparams.observation_config,
                pipeline_cfg=self.hparams.pipeline,
                window_size=self.hparams.window_size,
                latent_step_hours=self.hparams.latent_step_hours,
                require_targets=require_targets,
                verbose=False,
            )
            # Bins are named as `binYYYYMMDDHH` (time-aligned across instruments).
            # Fall back to lexicographic ordering if parsing fails.

            def _bin_sort_key(name: str):
                try:
                    if name.startswith('bin'):
                        return int(name[3:])
                    return int(name)
                except Exception:
                    return name

            bin_names = sorted(data_summary.keys(), key=_bin_sort_key)
            return data_summary, bin_names

        if not is_ddp:
            if os.path.exists(cache_path):
                try:
                    obj = torch.load(cache_path, weights_only=False)
                    return obj["data_summary"], obj["bin_names"]
                except Exception as e:
                    if verbose:
                        print(f"[DM.cache] WARNING: failed to load cache {cache_path}: {e}; rebuilding")
                    try:
                        os.remove(cache_path)
                    except OSError:
                        pass
            data_summary, bin_names = _build()
            built_obj = {"data_summary": data_summary, "bin_names": bin_names}
            tmp = f"{cache_path}.tmp.{os.getpid()}"
            torch.save(built_obj, tmp)
            os.replace(tmp, cache_path)
            return data_summary, bin_names

        built_obj = None
        build_error: str | None = None
        if rank == 0:
            if os.path.exists(cache_path):
                try:
                    built_obj = torch.load(cache_path, weights_only=False)
                except Exception as e:
                    build_error = f"cache load failed: {e}"
            if built_obj is None and build_error is None:
                if verbose:
                    print(f"[DM.cache] building {kind} summary -> {cache_path}")
                try:
                    data_summary, bin_names = _build()
                    built_obj = {"data_summary": data_summary, "bin_names": bin_names}
                    tmp = f"{cache_path}.tmp.{os.getpid()}"
                    torch.save(built_obj, tmp)
                    os.replace(tmp, cache_path)
                except Exception as e:
                    build_error = f"_build() failed: {e}"

        # Broadcast rank-0 failure status so other ranks don't hang in the
        # cache-file poll loop below waiting for a file that will never exist.
        err_list = [build_error]
        try:
            dist.broadcast_object_list(err_list, src=0)
        except Exception:
            # Older PyTorch / unsupported backend -- fall through; the barrier
            # plus polling will still surface the issue eventually.
            pass
        if err_list[0] is not None:
            dist.barrier()
            raise RuntimeError(
                f"[DM.cache] rank 0 failed to build {kind} summary at {cache_path}: {err_list[0]}"
            )

        dist.barrier()

        if rank == 0:
            return built_obj["data_summary"], built_obj["bin_names"]

        for _ in range(120):
            if os.path.exists(cache_path):
                break
            time.sleep(0.25)
        obj = torch.load(cache_path, weights_only=False)
        return obj["data_summary"], obj["bin_names"]

    def _graph_cache_enabled(self) -> bool:
        return bool(getattr(self.hparams, "graph_cache_dir", None))

    def _graph_cache_namespace(self, kind: str) -> str:
        rank, world_size = get_rank_world_size()
        payload = {
            "graph_cache_version": 1,
            "kind": str(kind),
            "window_size": str(getattr(self.hparams, "window_size", "")),
            "latent_step_hours": int(getattr(self.hparams, "latent_step_hours", 0) or 0),
            "observation_config": getattr(self.hparams, "observation_config", None),
            "pipeline": getattr(self.hparams, "pipeline", None),
            "parallelization_strategy": self.parallelization_strategy,
            "domain_halo_hops": int(self.domain_halo_hops),
            "rank": int(rank),
            "world_size": int(world_size),
            "mesh_nodes": int(self.mesh_structure["mesh_features_torch"][0].shape[0]),
        }
        digest = hashlib.blake2b(json.dumps(payload, sort_keys=True, default=str).encode("utf-8"), digest_size=16).hexdigest()
        return f"{kind}/rank{rank:04d}_of_{world_size:04d}/{digest}"

    def _graph_cache_path(self, kind: str, bin_name: str) -> str | None:
        if not self._graph_cache_enabled():
            return None
        safe_bin_name = str(bin_name).replace(os.sep, "_")
        return os.path.join(str(self.hparams.graph_cache_dir), self._graph_cache_namespace(kind), f"{safe_bin_name}.pt")

    def _make_graph_cache_path_fn(self, kind: str):
        if not self._graph_cache_enabled():
            return None
        return lambda bin_name: self._graph_cache_path(kind, bin_name)

    # ------------- Setup / Zarr open -------------

    def setup(self, stage=None):
        _, rank = self._ddp_info()

        self._ensure_domain_sharder()

        # Open Zarrs once
        if self.z is None:
            self.z = {}
            for obs_type, instruments in self.hparams.observation_config.items():
                self.z[obs_type] = {}
                for inst_name, inst_cfg in instruments.items():
                    src = inst_cfg.get("source", "zarr")

                    if src == "zarr":
                        zarr_dir = inst_cfg.get("zarr_dir")
                        if zarr_dir:
                            zarr_path = zarr_dir
                        else:
                            zname = inst_cfg.get("zarr_name", inst_name)
                            zarr_path, used_fallback = _resolve_zarr_path(
                                self.hparams.data_path,
                                zname,
                                self.hparams.start_date,
                            )
                            if used_fallback and rank == 0:
                                print(
                                    f"[ZARR] {obs_type}/{inst_name} requested year {self.hparams.start_date} "
                                    f"not found; using available store {zarr_path}"
                                )

                        if not os.path.isdir(zarr_path):
                            raise FileNotFoundError(f"Zarr not found: {zarr_path}")

                        # Keep the per-process Zarr cache conservative; this code runs once per rank
                        # and dataloader workers can multiply the effective host-memory footprint.
                        store = LRUStoreCache(
                            zarr.DirectoryStore(zarr_path),
                            max_size=int(self.hparams.zarr_cache_max_size_bytes),
                        )
                        self.z[obs_type][inst_name] = zarr.open(store, mode="r")

                        if rank == 0:
                            print(f"[ZARR] {obs_type}/{inst_name} -> {zarr_path}")
                            try:
                                print("       keys:", list(self.z[obs_type][inst_name].keys())[:12])
                            except Exception:
                                pass

                        if obs_type == "conventional" and inst_name == "surface_obs":
                            if not os.path.basename(zarr_path).startswith("raw_surface_obs"):
                                print(f"[WARN] surface_obs expected raw_surface_obs*.zarr but got: {zarr_path}")

                    else:
                        raise ValueError(
                            f"Unknown source '{src}' for {inst_name}. "
                            "NNJA support has been removed from this repo; use src='zarr'."
                        )

        if dist.is_available() and dist.is_initialized():
            dist.barrier()

        # Build TRAIN and VAL summaries for current windows
        print(
            f"[Rank {rank}] [DM.setup stage={stage}] "
            f"train_window={self.hparams.train_start}..{self.hparams.train_end} | "
            f"val_window={self.hparams.val_start}..{self.hparams.val_end}"
        )

        self._rebuild_train_summary()
        self._rebuild_val_summary()

        if self._graph_cache_enabled() and bool(getattr(self.hparams, "precompute_graph_cache", False)):
            self.precompute_graph_cache(stage=stage)

        if stage in (None, "fit"):
            # For now we use the full lists produced by organize_bins_times;
            # callbacks can narrow them by changing windows and triggering reload.
            pass

    # ------------- Summary (re)builders -------------
    def _rebuild_train_summary(self):
        _, rank = self._ddp_info()
        # Drop the previous summary before building a new one so peak RSS does not
        # transiently hold two copies. Important when this is called during epoch
        # resampling on every rank.
        self.train_data_summary = None
        self.train_bin_names = []
        gc.collect()
        self.train_data_summary, self.train_bin_names = self._load_or_build_summary(
            "train",
            self.hparams.train_start,
            self.hparams.train_end,
            require_targets=True,
        )
        gc.collect()
        print(
            f"[Rank {rank}] [DM.train_summary] v{self._train_version} sum_id={id(self.train_data_summary)} "
            f"bins={len(self.train_bin_names)} first={self.train_bin_names[0] if self.train_bin_names else None}"
        )

    def _rebuild_val_summary(self):
        _, rank = self._ddp_info()
        key = (pd.to_datetime(self.hparams.val_start), pd.to_datetime(self.hparams.val_end))
        if self._cache_val_windows and key in self._val_cache:
            self.val_data_summary, self.val_bin_names = self._val_cache[key]
        else:
            # Drop the previous val summary before building a new one.
            self.val_data_summary = None
            self.val_bin_names = []
            gc.collect()
            self.val_data_summary, self.val_bin_names = self._load_or_build_summary(
                "val",
                self.hparams.val_start,
                self.hparams.val_end,
                require_targets=self.require_targets,
            )
            gc.collect()
            if self._cache_val_windows:
                self._val_cache[key] = (self.val_data_summary, self.val_bin_names)
                self._val_cache_lru.append(key)
                while len(self._val_cache_lru) > self._val_cache_max_entries:
                    old = self._val_cache_lru.pop(0)
                    self._val_cache.pop(old, None)
        print(
            f"[Rank {rank}] [DM.val_summary]   v{self._val_version} sum_id={id(self.val_data_summary)} "
            f"bins={len(self.val_bin_names)} first={self.val_bin_names[0] if self.val_bin_names else None}"
        )

    # ------------- Window setters for callbacks -------------
    def set_train_window(self, start_dt, end_dt):
        self.hparams.train_start = pd.to_datetime(start_dt)
        self.hparams.train_end = pd.to_datetime(end_dt)
        self._train_version += 1
        print(f"[DM.set_train_window] v{self._train_version} -> {self.hparams.train_start} .. {self.hparams.train_end}")
        # Rebuild summary/bin names immediately so the *next* dataloader reload sees fresh objects
        self._rebuild_train_summary()
        if self._graph_cache_enabled() and bool(getattr(self.hparams, "precompute_graph_cache", False)):
            train_ds = self._make_dataset(self.train_bin_names, self.train_data_summary, "TRAIN", True, "train")
            self._precompute_dataset_cache("train", train_ds)

    def set_val_window(self, start_dt, end_dt):
        self.hparams.val_start = pd.to_datetime(start_dt)
        self.hparams.val_end = pd.to_datetime(end_dt)
        self._val_version += 1
        print(f"[DM.set_val_window]   v{self._val_version} -> {self.hparams.val_start} .. {self.hparams.val_end}")
        self._rebuild_val_summary()
        if self._graph_cache_enabled() and bool(getattr(self.hparams, "precompute_graph_cache", False)):
            val_ds = self._make_dataset(self.val_bin_names, self.val_data_summary, "VAL", self.require_targets, "val")
            self._precompute_dataset_cache("val", val_ds)

    # ------------- Graph builder -------------

    def _create_graph_structure(self, bin_data):
        data = HeteroData()

        # 1) Mesh nodes and edges
        data["mesh"].x = _t32(self.mesh_structure["mesh_features_torch"][0])
        data["mesh"].pos = _t32(self.mesh_structure["mesh_lat_lon_list"][0])

        # Cache the bidirectional fp16 mesh edge tensors once. They're static
        # across bins and otherwise get re-materialized (and re-allocated) on
        # every __getitem__ inside every worker process.
        if getattr(self, "_m2m_bidir_cache", None) is None:
            m2m_edge_index = self.mesh_structure["m2m_edge_index_torch"][0]
            m2m_edge_attr = self.mesh_structure["m2m_features_torch"][0].to(torch.float16)
            reverse_edges = torch.stack([m2m_edge_index[1], m2m_edge_index[0]], dim=0)
            self._m2m_bidir_cache = (
                torch.cat([m2m_edge_index, reverse_edges], dim=1).contiguous(),
                torch.cat([m2m_edge_attr, m2m_edge_attr], dim=0).contiguous(),
            )
        data["mesh", "to", "mesh"].edge_index = self._m2m_bidir_cache[0]
        data["mesh", "to", "mesh"].edge_attr = self._m2m_bidir_cache[1]

        window_hours = int(self.hparams.window_size.replace('h', ''))

        # Sanity check: ensure window_hours is divisible by latent_step_hours
        if window_hours % self.hparams.latent_step_hours != 0:
            raise ValueError(f"window_size ({window_hours}h) must be divisible by latent_step_hours ({self.hparams.latent_step_hours}h)")

        num_latent_steps = window_hours // self.hparams.latent_step_hours

        # 3) Observation data and mesh connections
        # ALL instruments get the same node structure based on detected batch mode
        for obs_type, instruments in self.hparams.observation_config.items():
            for inst_name, inst_cfg in instruments.items():

                # Check if this instrument has data for this time bin
                if obs_type in bin_data and inst_name in bin_data[obs_type]:
                    inst_dict = bin_data[obs_type][inst_name]
                    self._create_latent_nodes(data, inst_name, inst_dict, num_latent_steps)
                else:
                    # MISSING INSTRUMENT: Create empty nodes with same structure as present instruments
                    self._create_empty_latent_nodes(data, inst_name, inst_cfg, num_latent_steps)

        if self.domain_sharder is not None and self.domain_sharder.is_enabled:
            # Slice the global hetero-graph down to this rank's owned mesh,
            # halo mesh, local targets, and any boundary observations needed.
            data = self.domain_sharder.shard_graph(data)

        return data

    def _ensure_domain_sharder(self):
        if self.parallelization_strategy != "domain":
            return

        rank, world_size = get_rank_world_size()
        sync_ready = bool(dist.is_available() and dist.is_initialized() and world_size > 1)

        if self.domain_sharder is not None:
            same_rank = self.domain_sharder.rank == rank
            same_world_size = self.domain_sharder.world_size == world_size
            already_synced = getattr(self.domain_sharder, "uses_synced_partition", False)
            if same_rank and same_world_size and (not sync_ready or already_synced):
                return

        mesh_x = _t32(self.mesh_structure["mesh_features_torch"][0])
        mesh_pos = _t32(self.mesh_structure["mesh_lat_lon_torch"][0])
        mesh_edge_index = _t64(self.mesh_structure["m2m_edge_index_torch"][0])
        mesh_edge_attr = _t32(self.mesh_structure["m2m_features_torch"][0]).to(torch.float16)

        self.domain_sharder = DomainGraphSharder(
            mesh_x=mesh_x,
            mesh_pos=mesh_pos,
            mesh_edge_index=mesh_edge_index,
            mesh_edge_attr=mesh_edge_attr,
            rank=rank,
            world_size=world_size,
            halo_hops=self.domain_halo_hops,
        )

        if rank == 0:
            spec = self.domain_sharder.spec
            print(
                "[DataModule] Domain sharding enabled: "
                f"world_size={world_size}, owned_nodes={spec.owned_node_count}, "
                f"halo_nodes={spec.halo_node_count}, synced_partition={self.domain_sharder.uses_synced_partition}"
            )

    def _create_latent_nodes(self, data, inst_name, inst_dict, num_latent_steps):
        """Create nodes for instrument with data in latent mode."""
        # Input features (same for all steps)
        node_type_input = f"{inst_name}_input"
        if "input_features_final" in inst_dict:
            data[node_type_input].x = _t32(inst_dict["input_features_final"])

            if "input_features_raw" in inst_dict:
                data[node_type_input].input_features_raw = _t32(inst_dict["input_features_raw"])
            if "input_channel_mask" in inst_dict:
                data[node_type_input].input_channel_mask = torch.as_tensor(
                    inst_dict["input_channel_mask"], dtype=torch.bool
                )
            if "input_time_unix" in inst_dict:
                data[node_type_input].input_times = _t64(inst_dict["input_time_unix"])

            # Store pressure level index for radiosonde and aircraft (if available)
            if "input_pressure_level" in inst_dict:
                data[node_type_input].pressure_level = inst_dict["input_pressure_level"].long()
                if self._is_verbose():
                    print(
                        f"[DATAMODULE] Stored pressure_level for {node_type_input}: "
                        f"shape={data[node_type_input].pressure_level.shape}, "
                        f"range=[{data[node_type_input].pressure_level.min()}, {data[node_type_input].pressure_level.max()}]"
                    )
            elif inst_name in ["radiosonde", "aircraft"] and self._is_verbose():
                print(f"[DATAMODULE] WARNING: No pressure_level found for {node_type_input}! Data may not be preprocessed with new code.")

            # Create encoder edges (observation to mesh)
            if "input_lat_deg" in inst_dict and "input_lon_deg" in inst_dict:
                grid_lat_deg = inst_dict["input_lat_deg"]
                grid_lon_deg = inst_dict["input_lon_deg"]

                # Keep lat/lon on the observation nodes (used by FSOI matching)
                data[node_type_input].lat = _t32(grid_lat_deg)
                data[node_type_input].lon = _t32(grid_lon_deg)

                edge_index_encoder, edge_attr_encoder = obs_mesh_conn(
                    grid_lat_deg,
                    grid_lon_deg,
                    self.mesh_structure["m2m_graphs"],
                    self.mesh_structure["mesh_lat_lon_list"],
                    self.mesh_structure["mesh_list"],
                    o2m=True,
                )
                data[node_type_input, "to", "mesh"].edge_index = edge_index_encoder
                data[node_type_input, "to", "mesh"].edge_attr = edge_attr_encoder.to(torch.float16)

        # Handle target features for each latent step
        if "target_features_final_list" not in inst_dict:
            return

        for step in range(num_latent_steps):
            if step >= len(inst_dict["target_features_final_list"]):
                continue

            node_type_target = f"{inst_name}_target_step{step}"
            target_features = inst_dict["target_features_final_list"][step]

            # Get channel mask and check validity
            target_channel_mask = inst_dict.get("target_channel_mask_list", [None])[step] if step < len(
                inst_dict.get("target_channel_mask_list", [])) else None

            if target_channel_mask is not None:
                target_channel_mask = target_channel_mask.to(torch.bool)
                keep_t = target_channel_mask.any(dim=1)  # Keep rows with ANY valid channel
            else:
                keep_t = torch.ones((target_features.shape[0],), dtype=torch.bool)

            # Handle empty case
            if keep_t.sum() == 0:
                data[node_type_target].y = torch.empty((0, target_features.shape[1]), dtype=torch.float32)
                data[node_type_target].x = torch.empty((0, 1), dtype=torch.float32)
                data[node_type_target].target_metadata = torch.empty((0, 3), dtype=torch.float32)
                data[node_type_target].instrument_ids = torch.empty((0,), dtype=torch.long)
                data[node_type_target].target_channel_mask = torch.empty((0, target_features.shape[1]), dtype=torch.bool)
                data[node_type_target].target_pressure_hpa = torch.empty((0,), dtype=torch.float32)
                data[node_type_target].obs_time_unix = torch.empty((0,), dtype=torch.long)
                continue

            keep_np = keep_t.cpu().numpy()

            # Filter all data
            y_t = target_features[keep_t]
            mask_t = target_channel_mask[keep_t] if target_channel_mask is not None else torch.ones_like(y_t, dtype=torch.bool)

            data[node_type_target].y = _t32(y_t)
            # IMPORTANT: keep as bool to avoid massive memory blow-ups for satellite targets.
            data[node_type_target].target_channel_mask = mask_t.to(torch.bool)

            # Metadata
            if "target_metadata_list" in inst_dict and step < len(inst_dict["target_metadata_list"]):
                tgt_meta = inst_dict["target_metadata_list"][step][keep_t]
                data[node_type_target].target_metadata = _t32(tgt_meta)

            # Scan angle handling per-instrument (config-driven)
            # Determine observation type to look up config
            obs_type = "satellite" if inst_name in self.hparams.observation_config.get("satellite", {}) else "conventional"
            scan_angle_cols = self.hparams.observation_config[obs_type][inst_name].get("scan_angle_channels", 1)

            if "scan_angle_list" in inst_dict and step < len(inst_dict["scan_angle_list"]):
                x_aux = inst_dict["scan_angle_list"][step][keep_t]

                # Validate and pad/truncate to expected dimensions
                if x_aux.shape[-1] != scan_angle_cols:
                    if x_aux.shape[-1] > scan_angle_cols:
                        x_aux = x_aux[:, :scan_angle_cols]
                    else:
                        pad_cols = scan_angle_cols - x_aux.shape[-1]
                        padding = torch.zeros((x_aux.shape[0], pad_cols), dtype=x_aux.dtype, device=x_aux.device)
                        x_aux = torch.cat([x_aux, padding], dim=-1)
            else:
                x_aux = torch.zeros((y_t.shape[0], scan_angle_cols), dtype=torch.float32)
            data[node_type_target].x = _t32(x_aux)

            # Instrument ID
            if "instrument_id" in inst_dict:
                data[node_type_target].instrument_ids = torch.full(
                    (y_t.shape[0],),
                    inst_dict["instrument_id"],
                    dtype=torch.long
                )

            # Pressure data for radiosonde and aircraft (used for evaluation CSV)
            if "target_pressure_hpa_list" in inst_dict and step < len(inst_dict["target_pressure_hpa_list"]):
                pressure_hpa = inst_dict["target_pressure_hpa_list"][step][keep_np]
                data[node_type_target].target_pressure_hpa = _t32(torch.tensor(pressure_hpa, dtype=torch.float32))

            # Per-observation timestamps (unix seconds) for verifying within-window spread
            if "target_time_unix_list" in inst_dict and step < len(inst_dict["target_time_unix_list"]):
                obs_unix = inst_dict["target_time_unix_list"][step]
                obs_unix = np.asarray(obs_unix, dtype=np.int64)
                if obs_unix.size:
                    obs_unix = obs_unix[keep_np]
                data[node_type_target].obs_time_unix = _t64(torch.tensor(obs_unix, dtype=torch.long))
            else:
                data[node_type_target].obs_time_unix = torch.full((y_t.shape[0],), -1, dtype=torch.long)

            # Store pressure level index for radiosonde and aircraft (if available)
            if "target_pressure_level_list" in inst_dict and step < len(inst_dict["target_pressure_level_list"]):
                pressure_level_idx = inst_dict["target_pressure_level_list"][step][keep_t]
                data[node_type_target].pressure_level = pressure_level_idx.long()
                if self._is_verbose():
                    print(
                        f"[DATAMODULE] Stored pressure_level for {node_type_target}: "
                        f"shape={data[node_type_target].pressure_level.shape}, "
                        f"range=[{data[node_type_target].pressure_level.min()}, {data[node_type_target].pressure_level.max()}]"
                    )
            elif inst_name in ["radiosonde", "aircraft"] and self._is_verbose():
                print(f"[DATAMODULE] WARNING: No pressure_level found for {node_type_target}! Data may not be preprocessed with new code.")

            # Edges - filter lat/lon too
            if ("target_lat_deg_list" in inst_dict and "target_lon_deg_list" in inst_dict):
                target_lat_deg = inst_dict["target_lat_deg_list"][step][keep_np]
                target_lon_deg = inst_dict["target_lon_deg_list"][step][keep_np]

                # Keep lat/lon on the target nodes (used by FSOI matching)
                data[node_type_target].lat = _t32(target_lat_deg)
                data[node_type_target].lon = _t32(target_lon_deg)

                if len(target_lat_deg) > 0:
                    edge_index_decoder, edge_attr_decoder = obs_mesh_conn(
                        target_lat_deg,
                        target_lon_deg,
                        self.mesh_structure["m2m_graphs"],
                        self.mesh_structure["mesh_lat_lon_list"],
                        self.mesh_structure["mesh_list"],
                        o2m=False,
                    )
                    data["mesh", "to", node_type_target].edge_index = edge_index_decoder
                    data["mesh", "to", node_type_target].edge_attr = edge_attr_decoder.to(torch.float16)

    def _create_empty_latent_nodes(self, data, inst_name, inst_cfg, num_latent_steps):
        """Create empty nodes for missing instrument in latent mode."""
        # Create empty input node
        node_type_input = f"{inst_name}_input"
        data[node_type_input].x = torch.empty((0, inst_cfg["input_dim"]), dtype=torch.float32)
        data[node_type_input].lat = torch.empty((0,), dtype=torch.float32)
        data[node_type_input].lon = torch.empty((0,), dtype=torch.float32)
        data[node_type_input, "to", "mesh"].edge_index = torch.empty((2, 0), dtype=torch.long)
        data[node_type_input, "to", "mesh"].edge_attr = torch.empty((0, 4), dtype=torch.float32)

        # Create empty target nodes for all latent steps
        for step in range(num_latent_steps):
            node_type_target = f"{inst_name}_target_step{step}"
            data[node_type_target].y = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.float32)
            # Get scan angle dimension from config
            obs_type = "satellite" if inst_name in self.hparams.observation_config.get("satellite", {}) else "conventional"
            scan_angle_dim = self.hparams.observation_config[obs_type][inst_name].get("scan_angle_channels", 1)
            data[node_type_target].x = torch.empty((0, scan_angle_dim), dtype=torch.float32)
            # lat/lon + instrument metadata + appended target time features
            metadata_dim = len(inst_cfg.get("metadata", [])) + LAT_LON_COLUMNS + 5
            data[node_type_target].target_metadata = torch.empty((0, metadata_dim), dtype=torch.float32)
            data[node_type_target].instrument_ids = torch.empty((0,), dtype=torch.long)
            data[node_type_target].target_channel_mask = torch.empty((0, inst_cfg["target_dim"]), dtype=torch.bool)
            data[node_type_target].target_pressure_hpa = torch.empty((0,), dtype=torch.float32)
            data["mesh", "to", node_type_target].edge_index = torch.empty((2, 0), dtype=torch.long)
            data["mesh", "to", node_type_target].edge_attr = torch.empty((0, 4), dtype=torch.float32)
            data[node_type_target].pos = torch.empty((0, LAT_LON_COLUMNS), dtype=torch.float32)  # from standard mode, seems unused
            data[node_type_target].num_nodes = 0  # from standard mode, seems unused
            data[node_type_target].lat = torch.empty((0,), dtype=torch.float32)
            data[node_type_target].lon = torch.empty((0,), dtype=torch.float32)

    def _make_dataset(self, 
                      bin_names, 
                      data_summary, tag: str, 
                      require_targets: bool, 
                      cache_kind: str, 
                      include_persistence_inputs: bool=False) -> BinDataset:
        return BinDataset(
            bin_names,
            data_summary,
            self.z,
            self._create_graph_structure,
            self.hparams.observation_config,
            feature_stats=self.feature_stats,
            require_targets=require_targets,
            include_persistence_inputs=include_persistence_inputs,
            tag=tag,
            verbose=bool(getattr(self.hparams, "verbose", False)),
            graph_cache_path_fn=self._make_graph_cache_path_fn(cache_kind),
            graph_cache_read=bool(getattr(self.hparams, "graph_cache_read", False)),
            graph_cache_write=bool(getattr(self.hparams, "graph_cache_write", False)),
        )

    def _precompute_dataset_cache(self, kind: str, ds: BinDataset) -> None:
        rank, _ = get_rank_world_size()
        key = (str(kind), int(rank), id(ds.data_summary), len(ds.bin_names))
        if key in self._precomputed_graph_cache_keys:
            return
        self._precomputed_graph_cache_keys.add(key)

        if not self._graph_cache_enabled() or not bool(getattr(self.hparams, "graph_cache_write", False)):
            return

        missing = []
        for bin_name in ds.bin_names:
            cache_path = ds.graph_cache_path_fn(bin_name) if ds.graph_cache_path_fn is not None else None
            if cache_path and not os.path.exists(cache_path):
                missing.append(bin_name)

        if not missing:
            if rank == 0:
                print(f"[GraphCache] {kind}: all {len(ds.bin_names)} rank-local shards already cached")
            return

        print(f"[Rank {rank}] [GraphCache] precomputing {len(missing)}/{len(ds.bin_names)} {kind} shards")
        index_by_bin_name = {bin_name: idx for idx, bin_name in enumerate(ds.bin_names)}
        for idx, bin_name in enumerate(missing):
            original_idx = index_by_bin_name[bin_name]
            _ = ds[original_idx]
            if self._is_verbose() and rank == 0 and (idx + 1) % 8 == 0:
                print(f"[GraphCache] {kind}: cached {idx + 1}/{len(missing)}")

    def precompute_graph_cache(self, stage=None) -> None:
        if not self._graph_cache_enabled():
            return
        train_ds = self._make_dataset(self.train_bin_names, self.train_data_summary, "TRAIN", True, "train")
        val_ds = self._make_dataset(self.val_bin_names, self.val_data_summary, "VAL", self.require_targets, "val")
        if stage in (None, "fit"):
            self._precompute_dataset_cache("train", train_ds)
            if self.val_bin_names:
                self._precompute_dataset_cache("val", val_ds)

    # ------------- DataLoaders -------------
    def _worker_init(self, worker_id):
        import numpy as np
        base_seed = int(torch.initial_seed()) % 2**31
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if self._is_verbose():
            print(
                f"[WorkerInit] rank={rank} worker={worker_id} pid={os.getpid()} seed={base_seed} "
                f"train_sum_id={id(self.train_data_summary)} val_sum_id={id(self.val_data_summary)}"
            )

    def _loader_kwargs(self, num_workers: int, num_bins: int | None = None) -> dict:
        # Cap workers to the actual amount of work available on this rank.
        # Spawning more workers than bins/batches just multiplies the
        # copy-on-write fork footprint without speeding anything up, and has
        # been implicated in node-level OOMs during DataLoader fork.
        if num_bins is not None and num_bins > 0:
            num_workers = min(int(num_workers), int(num_bins))
        num_workers = max(0, int(num_workers))
        kwargs = {
            "num_workers": num_workers,
            "pin_memory": bool(self.hparams.pin_memory),
            # persistent_workers avoids tearing down/respawning worker
            # processes (and re-importing/re-opening zarr stores) between
            # epochs. Note: Lightning's `reload_dataloaders_every_n_epochs`
            # will still rebuild the DataLoader at epoch boundaries, so this
            # is effective only when that reload is disabled or infrequent.
            "persistent_workers": num_workers > 0,
        }
        if num_workers > 0:
            kwargs["worker_init_fn"] = self._worker_init
            kwargs["prefetch_factor"] = int(self.hparams.dataloader_prefetch_factor)
        return kwargs

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        if hasattr(batch, "to"):
            return batch.to(device)
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    def train_dataloader(self):
        self._ensure_domain_sharder()
        ds = self._make_dataset(self.train_bin_names, self.train_data_summary, "TRAIN", True, "train")

        is_dist = bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
        use_domain = bool(self.domain_sharder is not None and self.domain_sharder.is_enabled)
        sampler = None
        shuffle = True
        generator = None
        if use_domain:
            generator = torch.Generator()
            generator.manual_seed(self.data_loader_seed + self._train_version)
        elif is_dist:
            sampler = DistributedSampler(ds, shuffle=True)
            shuffle = False

        loader = PyGDataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            generator=generator,
            **self._loader_kwargs(self.hparams.train_num_workers, num_bins=len(self.train_bin_names)),
        )
        print(f"[DL] TRAIN v{self._train_version} loader_id={id(loader)} ds_id={id(ds)} "
              f"sum_id={id(self.train_data_summary)} bins={len(self.train_bin_names)}")
        return loader

    def val_dataloader(self):
        if not self.val_bin_names:
            return None

        self._ensure_domain_sharder()
        ds = self._make_dataset(self.val_bin_names, 
                                self.val_data_summary,
                                tag="VAL", 
                                require_targets=True, 
                                cache_kind="val", 
                                include_persistence_inputs=self.include_persistence_inputs,)

        is_dist = bool(dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1)
        use_domain = bool(self.domain_sharder is not None and self.domain_sharder.is_enabled)
        sampler = None
        if is_dist and not use_domain:
            sampler = DistributedSampler(ds, shuffle=False)

        loader = PyGDataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            sampler=sampler,
            **self._loader_kwargs(self.hparams.val_num_workers, num_bins=len(self.val_bin_names)),
        )
        print(f"[DL] VAL   v{self._val_version} loader_id={id(loader)} ds_id={id(ds)} "
              f"sum_id={id(self.val_data_summary)} bins={len(self.val_bin_names)}")
        return loader

    def predict_dataloader(self):
        """Create dataloader for prediction/inference mode."""
        print("\n[PREDICT] Setting up prediction dataloader")
        self._ensure_domain_sharder()

        # Use val_data_summary for prediction
        if not hasattr(self, 'val_data_summary') or not self.val_data_summary:
            print("[PREDICT] Building prediction data summary...")
            self._rebuild_val_summary()

        if not self.val_bin_names:
            print("[WARN] No bins found for prediction!")
            return None

        # Route through _make_dataset so the graph_cache is reused during inference.
        ds = self._make_dataset(self.val_bin_names, 
                                self.val_data_summary, 
                                tag="PREDICT", 
                                require_targets=self.require_targets, 
                                cache_kind="val",
                                include_persistence_inputs=self.include_persistence_inputs)

        # Create dataloader
        loader = PyGDataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            **self._loader_kwargs(self.hparams.predict_num_workers, num_bins=len(self.val_bin_names)),
        )

        print(f"[PREDICT] Dataloader created: {len(self.val_bin_names)} bins")
        print(f"[PREDICT] require_targets={self.require_targets}")

        return loader

    def fsoi_dataloader(self):
        """Deterministic dataloader for FSOI.

        Uses the same bin ordering as prediction, but enforces batch_size=1.
        """
        if not hasattr(self, 'val_data_summary') or not self.val_data_summary:
            self._rebuild_val_summary()

        self._ensure_domain_sharder()

        if not self.val_bin_names:
            return None

        # Route through _make_dataset so the graph_cache is reused during FSOI.
        ds = self._make_dataset(self.val_bin_names, 
                                self.val_data_summary, 
                                tag="FSOI", 
                                require_targets=self.require_targets, 
                                cache_kind="val", 
                                include_persistence_inputs=self.include_persistence_inputs)

        return PyGDataLoader(
            ds,
            batch_size=1,
            shuffle=False,
            **self._loader_kwargs(self.hparams.predict_num_workers, num_bins=len(self.val_bin_names)),
        )
