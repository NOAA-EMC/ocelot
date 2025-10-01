import hashlib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from timing_utils import timing_resource_decorator


def _subsample_by_mode(indices: np.ndarray, mode: str, stride: int, seed: int | None):
    """
    Return a subsampled, **sorted** view of `indices` according to `mode`.

    - mode == "stride": keep every `stride`th index
    - mode == "random": keep ~1/stride * len(indices) uniformly at random (no replacement)
    - mode == "none"  : keep all
    """
    idx = np.asarray(indices)
    n = idx.size
    if n == 0:
        return idx

    stride = max(1, int(stride))

    if mode == "none" or stride == 1:
        return np.sort(idx)

    if mode == "stride":
        return np.sort(idx[::stride])

    if mode == "random":
        # choose ceil(n/stride) to keep at least the intended fraction
        k = max(1, int(np.ceil(n / stride)))
        rng = np.random.default_rng(seed)
        take = rng.choice(n, size=k, replace=False)
        return np.sort(idx[take])

    raise ValueError(f"Unknown subsample mode: {mode!r}")


def _to_utc(ts) -> pd.Timestamp:
    """Return a UTC-aware Timestamp for strings or Timestamps."""
    t = pd.Timestamp(ts)
    return t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")


def _stable_seed(seed_base: int, bin_time: pd.Timestamp, obs_type: str, key: str, is_target: bool) -> int:
    """
    Make a stable, per-bin seed so runs are reproducible given the same inputs.
    Uses bin epoch seconds + obs_type + key + target/input flag.
    """
    ts_sec = int(_to_utc(bin_time).timestamp())
    payload = f"{seed_base}|{ts_sec}|{obs_type}|{key}|{int(is_target)}".encode()
    h = hashlib.blake2b(payload, digest_size=8).digest()
    return int(np.frombuffer(h, dtype=np.uint64)[0] % (2**32))


def _resolve_stride_mode(subs_cfg: dict, obs_type: str, inst_name: str, default_stride: int, default_mode: str) -> tuple[int, str]:
    """
    Resolve (stride, mode) for a given obs_type ('satellite'|'conventional') and instrument name.
    Supports legacy ints/strings and new per-instrument dicts with optional '_default'.
    """
    # stride
    stride_spec = subs_cfg.get(obs_type, default_stride)
    if isinstance(stride_spec, dict):
        stride = int(stride_spec.get(inst_name, stride_spec.get("_default", default_stride)))
    else:
        stride = int(stride_spec)

    # mode
    mode_cfg = subs_cfg.get("mode", {}) or {}
    mode_spec = mode_cfg.get(obs_type, default_mode)
    if isinstance(mode_spec, dict):
        mode = str(mode_spec.get(inst_name, mode_spec.get("_default", default_mode)))
    else:
        mode = str(mode_spec)

    return max(1, stride), mode


@timing_resource_decorator
def organize_bins_times(
    z_dict,
    start_date,
    end_date,
    observation_config,
    pipeline_cfg=None,
    window_size="12h",
    latent_step_hours=None,  # NEW PARAMETER
    verbose=False,
):
    """
    Bin definition: a bin consists of a pair of input and targets, each covers window_size.
    Organizes observation times into time bins and creates input-target pairs.

    Uses chunked scans to avoid loading entire arrays into memory.

    ## MODIFIED to support latent rollout (multiple target sub-windows).
    When latent_step_hours is provided, splits target window into sub-windows.
    """
    # --- normalize inputs to UTC-aware ---
    start_date = _to_utc(start_date)
    end_date = _to_utc(end_date)

    # normalize window unit (avoid pandas 'H' deprecation)
    window_size = window_size.lower()
    if not window_size.endswith("h"):
        raise ValueError("window_size must end with 'h' (e.g., '6h', '12h').")

    # subsampling config
    subs_cfg = (pipeline_cfg or {}).get("subsample", {}) or {}
    seed_base = int(subs_cfg.get("seed", 12345))

    # defaults if not specified (mode defaults to "random" for both)
    DEFAULTS = {
        "satellite": {"stride": 25, "mode": "random"},
        "conventional": {"stride": 20, "mode": "random"},
    }

    # START: LATENT ROLLOUT SETUP
    use_latent_rollout = latent_step_hours is not None
    if use_latent_rollout:
        target_hours = int(window_size[:-1])
        if target_hours % latent_step_hours != 0:
            raise ValueError(f"target_hours ({target_hours}) must be divisible by latent_step_hours ({latent_step_hours})")
        num_latent_steps = target_hours // latent_step_hours
        sub_window_freq = f"{latent_step_hours}h"
        if verbose:
            print(f"Latent rollout enabled: {num_latent_steps} steps of {latent_step_hours}h each.")
    else:
        num_latent_steps = 1
    # END: LATENT ROLLOUT SETUP

    t0 = int(start_date.timestamp())
    t1 = int(end_date.timestamp())

    data_summary = {}

    for obs_type in observation_config.keys():
        for key in observation_config[obs_type].keys():
            z = z_dict[obs_type][key]

            # --- Chunked scan to find candidate indices (time + optional sat filter) ---
            time_arr = z["time"]
            n_total = len(time_arr)
            chunk = getattr(time_arr, "chunks", (2_000_000,))[0]  # safe default if not chunked

            idx_parts = []
            if obs_type == "satellite":
                conf_sat_ids = np.asarray(observation_config[obs_type][key]["sat_ids"])
                for i0 in range(0, n_total, chunk):
                    i1 = min(i0 + chunk, n_total)
                    t = time_arr[i0:i1]
                    m_time = (t >= t0) & (t < t1)
                    if not m_time.any():
                        continue
                    sats = z["satelliteId"][i0:i1]
                    m = m_time & np.isin(sats, conf_sat_ids)
                    if m.any():
                        idx_parts.append(np.flatnonzero(m) + i0)
            else:
                for i0 in range(0, n_total, chunk):
                    i1 = min(i0 + chunk, n_total)
                    t = time_arr[i0:i1]
                    m_time = (t >= t0) & (t < t1)
                    if m_time.any():
                        idx_parts.append(np.flatnonzero(m_time) + i0)

            if not idx_parts:
                if verbose:
                    print(f"No observations for {obs_type}.{key} in {start_date} → {end_date}")
                continue

            idx_all = np.concatenate(idx_parts)

            # --- Sort by zar_time (or time) with minimal copies ---
            if "zar_time" in z:
                zar = z["zar_time"][idx_all]
            else:
                zar = time_arr[idx_all]
            order = np.argsort(zar, kind="stable")
            idx_all = idx_all[order]

            # --- Build window labels without a big DataFrame ---
            time_ts = pd.to_datetime(time_arr[idx_all], unit="s", utc=True)
            win = time_ts.floor(window_size)  # tz-aware

            # unique ordered windows + integer codes for each row's window
            uniq_win = pd.Index(win).unique().sort_values()
            codes = pd.Categorical(win, categories=uniq_win, ordered=True).codes

            n_bins = len(uniq_win) - 1
            if n_bins <= 0:
                if verbose:
                    print(f"Not enough windows to form input/target pairs for {obs_type}.{key}")
                continue

            # Resolve subsampling policy for this instrument
            if obs_type == "satellite":
                stride, mode = _resolve_stride_mode(subs_cfg, "satellite", key, DEFAULTS["satellite"]["stride"], DEFAULTS["satellite"]["mode"])
            else:
                stride, mode = _resolve_stride_mode(
                    subs_cfg, "conventional", key, DEFAULTS["conventional"]["stride"], DEFAULTS["conventional"]["mode"]
                )

            # --- Build bins; reproducible per-bin subsampling ---
            for bi in range(n_bins):  # exclude last window as target-only
                t_in = uniq_win[bi]

                m_in = codes == bi
                input_indices = idx_all[m_in]

                if use_latent_rollout:
                    # LATENT ROLLOUT: Split target window into sub-windows
                    t_target_start = uniq_win[bi + 1]

                    # Get all indices in the main target window
                    m_target_full = codes == (bi + 1)
                    idx_target_full = idx_all[m_target_full]
                    ts_target_full = time_ts[m_target_full]

                    # Generate the start/end times for each sub-window
                    target_sub_window_times = pd.date_range(start=t_target_start, periods=num_latent_steps + 1, freq=sub_window_freq)

                    target_indices_list = []

                    # Subsample input once
                    seed_in = _stable_seed(seed_base, t_in, obs_type, key, is_target=False)
                    input_indices = _subsample_by_mode(input_indices, mode, stride, seed_in)

                    # For each sub-window, filter and subsample
                    for step in range(num_latent_steps):
                        t_step_start, t_step_end = target_sub_window_times[step], target_sub_window_times[step+1]

                        m_step = (ts_target_full >= t_step_start) & (ts_target_full < t_step_end)
                        target_indices_step = idx_target_full[m_step]

                        seed_out = _stable_seed(seed_base, t_step_start, obs_type, key, is_target=True)
                        subsampled_indices = _subsample_by_mode(target_indices_step, mode, stride, seed_out)
                        target_indices_list.append(subsampled_indices)

                    if input_indices.size == 0 and all(t.size == 0 for t in target_indices_list):
                        continue

                    bin_name = f"bin{bi+1}"
                    data_summary.setdefault(bin_name, {}).setdefault(obs_type, {})[key] = {
                        "input_time": t_in,
                        "input_time_index": input_indices,
                        "target_times": list(target_sub_window_times[:-1]),  # List of timestamps
                        "target_time_indices": target_indices_list,         # List of index arrays
                        "num_latent_steps": num_latent_steps,
                        "is_latent_rollout": True,
                    }

                else:
                    # STANDARD ROLLOUT: Single target window (original behavior)
                    t_out = uniq_win[bi + 1]
                    m_out = codes == bi + 1
                    target_indices = idx_all[m_out]

                    # Per-bin stable seeds
                    seed_in = _stable_seed(seed_base, t_in, obs_type, key, is_target=False)
                    seed_out = _stable_seed(seed_base, t_out, obs_type, key, is_target=True)

                    # Apply subsampling
                    input_indices = _subsample_by_mode(input_indices, mode, stride, seed_in)
                    target_indices = _subsample_by_mode(target_indices, mode, stride, seed_out)

                    # Skip empty bin if both sides empty after subsampling
                    if input_indices.size == 0 and target_indices.size == 0:
                        continue

                    bin_name = f"bin{bi+1}"
                    data_summary.setdefault(bin_name, {}).setdefault(obs_type, {})[key] = {
                        "input_time": t_in,
                        "target_time": t_out,
                        "input_time_index": input_indices,
                        "target_time_index": target_indices,
                        "is_latent_rollout": False,
                    }

            if verbose:
                total_bins = sum(
                    1
                    for _ in range(n_bins)
                    if f"bin{_+1}" in data_summary and obs_type in data_summary[f"bin{_+1}"] and key in data_summary[f"bin{_+1}"][obs_type]
                )
                print(f"Created {total_bins} bins (pairs of input-target) for {obs_type}.{key}.")

    return data_summary


def _name2id(observation_config):
    order = []
    for obs_type in ("satellite", "conventional"):
        if obs_type in observation_config:
            order += sorted(observation_config[obs_type].keys())
    return {name: i for i, name in enumerate(order)}


# Helper that returns an empty (N,0) if there are no keys
def _stack_or_empty(arrs, keys, idx):
    if not keys:
        return np.empty((len(idx), 0), dtype=np.float32)
    return np.column_stack([arrs[k][idx] for k in keys]).astype(np.float32)


def _stats_from_cfg(feature_stats, inst_name, feat_keys):
    """Return (means, stds) for this instrument/feature order or (None, None) if missing."""
    if feature_stats is None or inst_name not in feature_stats:
        return None, None
    try:
        means = np.array([feature_stats[inst_name][k][0] for k in feat_keys], dtype=np.float32)
        stds = np.array([feature_stats[inst_name][k][1] for k in feat_keys], dtype=np.float32)
    except Exception:
        return None, None
    stds[(stds == 0) | ~np.isfinite(stds)] = 1.0
    means[~np.isfinite(means)] = 0.0
    return means, stds


@timing_resource_decorator
def extract_features(z_dict, data_summary, bin_name, observation_config, feature_stats=None):
    """
    Loads and normalizes features for each time bin.
    Adds per-channel masks for inputs and targets so features can be missing independently.
    Inputs: keep a row if ANY feature channel is valid; metadata can be missing (imputed later).
    Targets: require metadata row to be valid; features may be missing per-channel.

    ## MODIFIED to support latent rollout (multiple target windows).
    """
    print(f"\nProcessing {bin_name}...")
    for obs_type in list(data_summary[bin_name].keys()):
        for inst_name in list(data_summary[bin_name][obs_type].keys()):
            z = z_dict[obs_type][inst_name]
            data_summary_bin = data_summary[bin_name][obs_type][inst_name]

            input_idx = np.asarray(data_summary_bin["input_time_index"])

            # Detect if this is latent rollout or standard rollout
            is_latent_rollout = data_summary_bin.get("is_latent_rollout", False)

            if is_latent_rollout:
                target_indices_list = [np.asarray(ti) for ti in data_summary_bin["target_time_indices"]]
                num_latent_steps = data_summary_bin["num_latent_steps"]
            else:
                target_indices_list = [np.asarray(data_summary_bin["target_time_index"])]
                num_latent_steps = 1

            orig_in = input_idx.size
            orig_tg_sizes = [idx.size for idx in target_indices_list]

            if input_idx.size == 0 and all(idx.size == 0 for idx in target_indices_list):
                del data_summary[bin_name][obs_type][inst_name]
                continue

            # --- Config & feature ordering ---
            obs_cfg = observation_config[obs_type][inst_name]
            qc_filters = obs_cfg.get("qc_filters") or obs_cfg.get("qc")
            feat_keys = observation_config[obs_type][inst_name]["features"]
            meta_keys = observation_config[obs_type][inst_name]["metadata"]
            feat_pos = {k: i for i, k in enumerate(feat_keys)}
            n_ch = len(feat_keys)

            # Per-channel validity masks (inputs + ALL targets)
            input_valid_ch = np.ones((input_idx.size, n_ch), dtype=bool)
            target_valid_ch_list = [np.ones((idx.size, n_ch), dtype=bool) for idx in target_indices_list]

            # Track aux QC to propagate to wind_u / wind_v
            ws_ok_in = wd_ok_in = None
            ws_ok_tg_list = [None] * num_latent_steps
            wd_ok_tg_list = [None] * num_latent_steps

            # -------------------- QC (per-channel; apply to input + ALL targets simultaneously) --------------------
            if qc_filters:
                print(f"Applying QC for {inst_name}...")
                for var, cfg in qc_filters.items():
                    rng = cfg.get("range") if isinstance(cfg, dict) else (cfg if isinstance(cfg, (list, tuple)) else None)
                    flag_col = cfg.get("qm_flag_col") if isinstance(cfg, dict) else None
                    keep = set(cfg.get("keep", [])) if isinstance(cfg, dict) else None
                    pos = feat_pos.get(var, None)

                    # --- range QC ---
                    if rng is not None and var in z:
                        lo, hi = rng

                        # Apply to inputs
                        in_vals = z[var][input_idx]
                        if pos is not None:
                            input_valid_ch[:, pos] &= (in_vals >= lo) & (in_vals <= hi)
                        else:
                            # accumulate aux for u/v
                            if var == "windSpeed":
                                ws_ok_in = (in_vals >= lo) & (in_vals <= hi) if ws_ok_in is None else (ws_ok_in & ((in_vals >= lo) & (in_vals <= hi)))
                            if var == "windDirection":
                                wd_ok_in = (in_vals >= lo) & (in_vals <= hi) if wd_ok_in is None else (wd_ok_in & ((in_vals >= lo) & (in_vals <= hi)))

                        # Apply to ALL target windows
                        for step, target_idx in enumerate(target_indices_list):
                            if target_idx.size == 0:
                                continue
                            tg_vals = z[var][target_idx]
                            if pos is not None:
                                target_valid_ch_list[step][:, pos] &= (tg_vals >= lo) & (tg_vals <= hi)
                            else:
                                # accumulate aux for u/v
                                if var == "windSpeed":
                                    ws_ok_tg_list[step] = (tg_vals >= lo) & (tg_vals <= hi) if ws_ok_tg_list[step] is None else (
                                        ws_ok_tg_list[step] & ((tg_vals >= lo) & (tg_vals <= hi)))
                                if var == "windDirection":
                                    wd_ok_tg_list[step] = (tg_vals >= lo) & (tg_vals <= hi) if wd_ok_tg_list[step] is None else (
                                        wd_ok_tg_list[step] & ((tg_vals >= lo) & (tg_vals <= hi)))

                    # --- flag QC ---
                    if isinstance(cfg, dict) and flag_col and ("keep" in cfg) and (flag_col in z):
                        # Apply to inputs
                        in_flags = z[flag_col][input_idx]
                        keep_in = np.isin(in_flags, list(keep)) | (in_flags < 0)
                        if pos is not None:
                            input_valid_ch[:, pos] &= keep_in
                        else:
                            if var == "windSpeed":
                                ws_ok_in = keep_in if ws_ok_in is None else (ws_ok_in & keep_in)
                            if var == "windDirection":
                                wd_ok_in = keep_in if wd_ok_in is None else (wd_ok_in & keep_in)

                        # Apply to ALL target windows
                        for step, target_idx in enumerate(target_indices_list):
                            if target_idx.size == 0:
                                continue
                            tg_flags = z[flag_col][target_idx]
                            keep_tg = np.isin(tg_flags, list(keep)) | (tg_flags < 0)
                            if pos is not None:
                                target_valid_ch_list[step][:, pos] &= keep_tg
                            else:
                                if var == "windSpeed":
                                    ws_ok_tg_list[step] = keep_tg if ws_ok_tg_list[step] is None else (ws_ok_tg_list[step] & keep_tg)
                                if var == "windDirection":
                                    wd_ok_tg_list[step] = keep_tg if wd_ok_tg_list[step] is None else (wd_ok_tg_list[step] & keep_tg)

                # Wind component propagation
                if ("wind_u" in feat_pos) and ("wind_v" in feat_pos):
                    # Apply to inputs
                    if (ws_ok_in is not None) or (wd_ok_in is not None):
                        cond_in = np.ones(input_idx.size, dtype=bool)
                        if ws_ok_in is not None:
                            cond_in &= ws_ok_in
                        if wd_ok_in is not None:
                            cond_in &= wd_ok_in
                        input_valid_ch[:, feat_pos["wind_u"]] &= cond_in
                        input_valid_ch[:, feat_pos["wind_v"]] &= cond_in

                    # Apply to ALL target windows
                    for step in range(num_latent_steps):
                        if target_indices_list[step].size == 0:
                            continue
                        if (ws_ok_tg_list[step] is not None) or (wd_ok_tg_list[step] is not None):
                            cond_tg = np.ones(target_indices_list[step].size, dtype=bool)
                            if ws_ok_tg_list[step] is not None:
                                cond_tg &= ws_ok_tg_list[step]
                            if wd_ok_tg_list[step] is not None:
                                cond_tg &= wd_ok_tg_list[step]
                            target_valid_ch_list[step][:, feat_pos["wind_u"]] &= cond_tg
                            target_valid_ch_list[step][:, feat_pos["wind_v"]] &= cond_tg

            # -------------------- Feature extraction (following original pattern) --------------------
            def _get_feature(arrs, name, idx):
                if name in arrs:
                    return arrs[name][idx]
                if name in ("wind_u", "wind_v") and ("windSpeed" in arrs and "windDirection" in arrs):
                    ws = arrs["windSpeed"][idx].astype(np.float32)
                    wd = arrs["windDirection"][idx].astype(np.float32)
                    wd_rad = wd if np.nanmax(wd) <= (2 * np.pi + 0.1) else wd * (np.pi / 180.0)
                    u = -ws * np.sin(wd_rad)
                    v = -ws * np.cos(wd_rad)
                    return u if name == "wind_u" else v
                raise KeyError(f"Requested feature '{name}' not found in Zarr and no fallback rule defined.")

            # Extract input features
            input_features_raw = np.column_stack([_get_feature(z, k, input_idx) for k in feat_keys]).astype(np.float32)
            input_metadata_raw = _stack_or_empty(z, meta_keys, input_idx)
            input_lat_raw = z["latitude"][input_idx]
            input_lon_raw = z["longitude"][input_idx]
            input_times_raw = z["time"][input_idx]

            # Extract ALL target features
            target_features_raw_list = []
            target_metadata_raw_list = []
            target_lat_raw_list = []
            target_lon_raw_list = []
            target_times_raw_list = []

            for target_idx in target_indices_list:
                if target_idx.size == 0:
                    target_features_raw_list.append(np.empty((0, n_ch), dtype=np.float32))
                    target_metadata_raw_list.append(np.empty((0, len(meta_keys)), dtype=np.float32))
                    target_lat_raw_list.append(np.array([], dtype=np.float32))
                    target_lon_raw_list.append(np.array([], dtype=np.float32))
                    target_times_raw_list.append(np.array([], dtype=np.float32))
                else:
                    target_features_raw_list.append(np.column_stack([_get_feature(z, k, target_idx) for k in feat_keys]).astype(np.float32))
                    target_metadata_raw_list.append(_stack_or_empty(z, meta_keys, target_idx))
                    target_lat_raw_list.append(z["latitude"][target_idx])
                    target_lon_raw_list.append(z["longitude"][target_idx])
                    target_times_raw_list.append(z["time"][target_idx])

            # Replace fill values with NaN
            FILL_VALUE = 3.402823e38
            input_features_raw[input_features_raw >= FILL_VALUE] = np.nan
            if input_metadata_raw.size:
                input_metadata_raw[input_metadata_raw >= FILL_VALUE] = np.nan

            for target_features_raw in target_features_raw_list:
                if target_features_raw.size:
                    target_features_raw[target_features_raw >= FILL_VALUE] = np.nan

            for target_metadata_raw in target_metadata_raw_list:
                if target_metadata_raw.size:
                    target_metadata_raw[target_metadata_raw >= FILL_VALUE] = np.nan

            # -------------------- EXTRA CROSS-VARIABLE QC (following original pattern) --------------------
            rel = obs_cfg.get("qc_relations") or {}

            def _es_hpa(Tc):
                # Magnus (over water); Tc in °C → hPa
                # Clip Tc first to prevent overflow in the calculation
                Tc_safe = np.clip(Tc, -100.0, 100.0)  # Reasonable temperature range in Celsius
                x = 17.67 * Tc_safe / (Tc_safe + 243.5)
                x = np.clip(x, -50.0, 50.0)   # keeps exp argument in a safe numeric range
                return 6.112 * np.exp(x)

            def _apply_relational_qc():
                # Apply to input + ALL targets
                for step in range(num_latent_steps):
                    target_features_raw = target_features_raw_list[step]
                    target_metadata_raw = target_metadata_raw_list[step]
                    target_valid_ch = target_valid_ch_list[step]

                    if target_features_raw.size == 0:
                        continue

                    # -- Td ≤ T (+0.5) and spread cap --
                    if rel.get("dewpoint_le_temp", False) and "airTemperature" in feat_pos and "dewPointTemperature" in feat_pos:
                        jT = feat_pos["airTemperature"]
                        jTd = feat_pos["dewPointTemperature"]
                        for arr, mask in ((input_features_raw, input_valid_ch), (target_features_raw, target_valid_ch)):
                            if arr.shape[0] == 0:
                                continue
                            T, Td = arr[:, jT], arr[:, jTd]
                            m = np.isfinite(T) & np.isfinite(Td)
                            bad_hi = m & (Td > T + 0.5)
                            bad_spread = m & ((T - Td) > float(rel.get("max_temp_dewpoint_spread", 60.0)))
                            bad = bad_hi | bad_spread
                            if np.any(bad):
                                mask[bad, jTd] = False

                    # -- RH vs Td consistency --
                    if (
                        np.isfinite(float(rel.get("rh_from_td_consistency_pct", np.nan))) and
                        "relativeHumidity" in feat_pos and
                        "airTemperature" in feat_pos and
                        "dewPointTemperature" in feat_pos
                    ):
                        jRH, jT, jTd = feat_pos["relativeHumidity"], feat_pos["airTemperature"], feat_pos["dewPointTemperature"]
                        for arr, mask in ((input_features_raw, input_valid_ch), (target_features_raw, target_valid_ch)):
                            if arr.shape[0] == 0:
                                continue
                            RH, T, Td = arr[:, jRH], arr[:, jT], arr[:, jTd]
                            m = np.isfinite(RH) & np.isfinite(T) & np.isfinite(Td)
                            if not m.any():
                                continue
                            RH_star = 100.0 * (_es_hpa(Td[m]) / _es_hpa(T[m]))
                            bad = np.zeros(RH.shape, dtype=bool)
                            bad[m] = np.abs(RH[m] - RH_star) > float(rel.get("rh_from_td_consistency_pct"))
                            if np.any(bad):
                                mask[bad, jRH] = False

                    # -- Pressure vs height --
                    pvh = rel.get("pressure_vs_height") or {}
                    if pvh.get("enable", False) and "airPressure" in feat_pos and ("height" in meta_keys):
                        H, tol_hpa = float(pvh.get("scale_height_m", 8000.0)), float(pvh.get("tolerance_hpa", 100.0))
                        jP, jH = feat_pos["airPressure"], meta_keys.index("height")
                        for feat_arr, meta_arr, vmask in (
                            (input_features_raw, input_metadata_raw, input_valid_ch),
                            (target_features_raw, target_metadata_raw, target_valid_ch)
                        ):
                            if feat_arr.shape[0] == 0 or meta_arr.size == 0:
                                continue
                            z_h, p = meta_arr[:, jH], feat_arr[:, jP]
                            m = np.isfinite(p) & np.isfinite(z_h)
                            if m.any():
                                p_exp = 1013.25 * np.exp(-np.clip(z_h[m], -500.0, 9000.0) / H)
                                bad = np.zeros_like(p, dtype=bool)
                                bad[m] = np.abs(p[m] - p_exp) > tol_hpa
                                if np.any(bad):
                                    vmask[bad, jP] = False

            # Treat 9999 height as missing
            if "height" in meta_keys:
                j = meta_keys.index("height")
                if input_metadata_raw.size:
                    input_metadata_raw[:, j] = np.where(input_metadata_raw[:, j] >= 9999.0, np.nan, input_metadata_raw[:, j])
                for target_metadata_raw in target_metadata_raw_list:
                    if target_metadata_raw.size:
                        target_metadata_raw[:, j] = np.where(target_metadata_raw[:, j] >= 9999.0, np.nan, target_metadata_raw[:, j])

            # Apply relational QC
            _apply_relational_qc()

            # -------------------- Continue with original processing pattern --------------------
            # The rest follows the original extract_features logic but handles multiple targets

            # Row keeping for INPUTS (same as original)
            observed_in = ~np.isnan(input_features_raw)
            keep_inputs = (observed_in & input_valid_ch).any(axis=1)

            if not keep_inputs.any():
                del data_summary[bin_name][obs_type][inst_name]
                continue

            # Process targets and check if we have any valid targets
            valid_targets_exist = False
            for step in range(num_latent_steps):
                target_metadata_raw = target_metadata_raw_list[step]
                if target_metadata_raw.size > 0:
                    valid_target_meta = ~np.isnan(target_metadata_raw).any(axis=1)
                    if valid_target_meta.any():
                        valid_targets_exist = True
                        break
                elif target_indices_list[step].size > 0:  # Empty metadata but non-empty targets
                    valid_targets_exist = True
                    break

            if not valid_targets_exist:
                del data_summary[bin_name][obs_type][inst_name]
                continue

            # -------------------- Row filtering and final cleaning --------------------

            # Filter inputs (same as original)
            input_idx_clean = input_idx[keep_inputs]
            input_features_raw_clean = input_features_raw[keep_inputs]
            input_metadata_raw_clean = input_metadata_raw[keep_inputs] if input_metadata_raw.size else input_metadata_raw
            input_lat_raw_clean = input_lat_raw[keep_inputs]
            input_lon_raw_clean = input_lon_raw[keep_inputs]
            input_times_clean = input_times_raw[keep_inputs]
            input_valid_ch_clean = input_valid_ch[keep_inputs]

            # Filter ALL targets based on metadata validity
            target_data_cleaned = []
            for step in range(num_latent_steps):
                target_idx = target_indices_list[step]
                target_features_raw = target_features_raw_list[step]
                target_metadata_raw = target_metadata_raw_list[step]
                target_lat_raw = target_lat_raw_list[step]
                target_lon_raw = target_lon_raw_list[step]
                target_times_raw = target_times_raw_list[step]
                target_valid_ch = target_valid_ch_list[step]

                if target_idx.size == 0:
                    target_data_cleaned.append({
                        'indices': np.array([], dtype=int),
                        'features': np.empty((0, n_ch), dtype=np.float32),
                        'metadata': np.empty((0, len(meta_keys)), dtype=np.float32),
                        'lat': np.array([], dtype=np.float32),
                        'lon': np.array([], dtype=np.float32),
                        'times': np.array([], dtype=np.float32),
                        'valid_ch': np.empty((0, n_ch), dtype=bool),
                    })
                    continue

                # TARGETS: require metadata only; allow per-channel NaNs
                valid_target_meta = ~np.isnan(target_metadata_raw).any(axis=1) if target_metadata_raw.size else np.ones(target_idx.size, bool)

                target_data_cleaned.append({
                    'indices': target_idx[valid_target_meta],
                    'features': target_features_raw[valid_target_meta],
                    'metadata': target_metadata_raw[valid_target_meta] if target_metadata_raw.size else target_metadata_raw,
                    'lat': target_lat_raw[valid_target_meta],
                    'lon': target_lon_raw[valid_target_meta],
                    'times': target_times_raw[valid_target_meta],
                    'valid_ch': target_valid_ch[valid_target_meta],
                })

            # Apply per-channel invalidation: set bad channels to NaN
            input_features_raw_clean[~input_valid_ch_clean] = np.nan
            for step in range(num_latent_steps):
                target_data = target_data_cleaned[step]
                if target_data['features'].shape[0] > 0:
                    target_data['features'][~target_data['valid_ch']] = np.nan

            # Check if we have any valid data left
            if input_features_raw_clean.shape[0] == 0 or all(td['features'].shape[0] == 0 for td in target_data_cleaned):
                del data_summary[bin_name][obs_type][inst_name]
                continue

            # -------------------- Feature engineering (following original pattern) --------------------

            # Input lat/lon/time encoding
            lat_rad_input = np.radians(input_lat_raw_clean)[:, None]
            lon_rad_input = np.radians(input_lon_raw_clean)[:, None]
            input_sin_lat, input_cos_lat = np.sin(lat_rad_input), np.cos(lat_rad_input)
            input_sin_lon, input_cos_lon = np.sin(lon_rad_input), np.cos(lon_rad_input)

            input_timestamps = pd.to_datetime(input_times_clean, unit="s")
            input_dayofyear = np.array([(ts.timetuple().tm_yday - 1 + (ts.hour * 3600 + ts.minute * 60 +
                                       ts.second) / 86400) / 365.24219 for ts in input_timestamps])[:, None]
            input_time_fraction = np.array([(ts.hour * 3600 + ts.minute * 60 + ts.second) / 86400 for ts in input_timestamps])
            input_sin_time = np.sin(2 * np.pi * input_time_fraction)[:, None]
            input_cos_time = np.cos(2 * np.pi * input_time_fraction)[:, None]

            # -------------------- Normalization (using ALL target data for stats) --------------------
            means, stds = _stats_from_cfg(feature_stats, inst_name, feat_keys)

            if means is None or stds is None:
                # Fallback: compute per-bin stats using input + ALL targets combined
                all_features = [input_features_raw_clean]
                for target_data in target_data_cleaned:
                    if target_data['features'].shape[0] > 0:
                        all_features.append(target_data['features'])

                if len(all_features) > 1:
                    combined_features = np.vstack(all_features)
                else:
                    combined_features = all_features[0]

                means = np.nanmean(combined_features, axis=0).astype(np.float32)
                stds = np.nanstd(combined_features, axis=0).astype(np.float32)
                stds[(stds == 0) | ~np.isfinite(stds)] = 1.0
                means[~np.isfinite(means)] = 0.0

            # -------------------- Process based on observation type --------------------
            if obs_type == "satellite":
                # Input normalization
                input_features_norm = (input_features_raw_clean - means) / stds

                # Input metadata: angles → cos; impute NaN with column mean (cos-space)
                if input_metadata_raw_clean.size:
                    input_metadata_rad = np.deg2rad(input_metadata_raw_clean)
                    input_metadata_cos = np.cos(input_metadata_rad)
                    col_means = np.nanmean(input_metadata_cos, axis=0)
                    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
                    input_metadata = np.where(np.isnan(input_metadata_cos), col_means, input_metadata_cos).astype(np.float32)
                else:
                    input_metadata = np.empty((input_features_norm.shape[0], 0), dtype=np.float32)

                # Assemble input features: geo/time + metadata + standardized features (NaN→0)
                input_features_final = np.column_stack([
                    input_sin_lat, input_cos_lat, input_sin_lon, input_cos_lon,
                    input_sin_time, input_cos_time, input_dayofyear,
                    input_metadata,
                    np.nan_to_num(input_features_norm, nan=0.0)
                ]).astype(np.float32)

                # Process ALL target windows
                target_features_final_list = []
                target_metadata_list = []
                scan_angle_list = []
                target_channel_mask_list = []
                target_lat_deg_list = []
                target_lon_deg_list = []

                for step in range(num_latent_steps):
                    target_data = target_data_cleaned[step]

                    if target_data['features'].shape[0] == 0:
                        target_features_final_list.append(torch.empty(0, n_ch, dtype=torch.float32))
                        target_metadata_list.append(torch.empty(0, len(meta_keys) + 2, dtype=torch.float32))
                        scan_angle_list.append(torch.empty(0, 1, dtype=torch.float32))
                        target_channel_mask_list.append(torch.empty(0, n_ch, dtype=torch.bool))
                        target_lat_deg_list.append(np.array([], dtype=np.float32))
                        target_lon_deg_list.append(np.array([], dtype=np.float32))
                        continue

                    # Target normalization
                    target_features_norm = (target_data['features'] - means) / stds

                    # Target metadata handling
                    lat_rad_target = np.radians(target_data['lat'])[:, None]
                    lon_rad_target = np.radians(target_data['lon'])[:, None]

                    if target_data['metadata'].size:
                        target_metadata_cos = np.cos(np.deg2rad(target_data['metadata']))
                    else:
                        target_metadata_cos = np.empty((target_features_norm.shape[0], 0), dtype=np.float32)

                    # Targets: build mask then NaN→0
                    target_channel_mask = ~np.isnan(target_features_norm)
                    target_features_final = np.nan_to_num(target_features_norm, nan=0.0).astype(np.float32)

                    scan_angle = target_metadata_cos[:, 0:1] if target_metadata_cos.shape[1] > 0 else np.zeros(
                        (target_features_final.shape[0], 1), dtype=np.float32)
                    target_metadata_final = np.column_stack([lat_rad_target, lon_rad_target, target_metadata_cos])

                    target_features_final_list.append(torch.tensor(target_features_final, dtype=torch.float32))
                    target_metadata_list.append(torch.tensor(target_metadata_final, dtype=torch.float32))
                    scan_angle_list.append(torch.tensor(scan_angle, dtype=torch.float32))
                    target_channel_mask_list.append(torch.tensor(target_channel_mask, dtype=torch.bool))
                    target_lat_deg_list.append(target_data['lat'])
                    target_lon_deg_list.append(target_data['lon'])

            else:
                # Conventional processing
                input_features_norm = (input_features_raw_clean - means) / stds
                input_channel_mask = ~np.isnan(input_features_norm)

                # Clip and use sentinel values (following original pattern)
                ZLIM, SENT = 6.0, -9.0
                x_in = np.clip(input_features_norm, -ZLIM, ZLIM)
                x_in = np.where(input_channel_mask, x_in, SENT).astype(np.float32)

                # Input metadata normalization
                if input_metadata_raw_clean.size:
                    meta = input_metadata_raw_clean.copy()
                    col_means = np.nanmean(meta, axis=0)
                    col_means = np.where(np.isfinite(col_means), col_means, 0.0)
                    meta = np.where(np.isnan(meta), col_means, meta)
                    input_metadata_norm = StandardScaler().fit_transform(meta)
                else:
                    input_metadata_norm = np.empty((x_in.shape[0], 0), dtype=np.float32)

                input_features_final = np.column_stack([
                    input_sin_lat, input_cos_lat, input_sin_lon, input_cos_lon,
                    input_sin_time, input_cos_time, input_dayofyear,
                    input_metadata_norm, x_in
                ]).astype(np.float32)

                # Process ALL target windows
                target_features_final_list = []
                target_metadata_list = []
                scan_angle_list = []
                target_channel_mask_list = []
                target_lat_deg_list = []
                target_lon_deg_list = []

                for step in range(num_latent_steps):
                    target_data = target_data_cleaned[step]

                    if target_data['features'].shape[0] == 0:
                        target_features_final_list.append(torch.empty(0, n_ch, dtype=torch.float32))
                        target_metadata_list.append(torch.empty(0, len(meta_keys) + 2, dtype=torch.float32))
                        scan_angle_list.append(torch.empty(0, 1, dtype=torch.float32))
                        target_channel_mask_list.append(torch.empty(0, n_ch, dtype=torch.bool))
                        target_lat_deg_list.append(np.array([], dtype=np.float32))
                        target_lon_deg_list.append(np.array([], dtype=np.float32))
                        continue

                    # Target normalization with clipping (conventional style)
                    target_features_norm = (target_data['features'] - means) / stds
                    target_channel_mask = ~np.isnan(target_features_norm)
                    target_features_final = np.clip(target_features_norm, -ZLIM, ZLIM)
                    target_features_final = np.where(target_channel_mask, target_features_final, SENT).astype(np.float32)

                    # Target metadata
                    lat_rad_target = np.radians(target_data['lat'])[:, None]
                    lon_rad_target = np.radians(target_data['lon'])[:, None]

                    if target_data['metadata'].size:
                        target_metadata_norm = StandardScaler().fit_transform(target_data['metadata'])
                    else:
                        target_metadata_norm = np.empty((target_features_final.shape[0], 0), dtype=np.float32)

                    target_metadata_final = np.column_stack([lat_rad_target, lon_rad_target, target_metadata_norm])
                    scan_angle = np.zeros((target_features_final.shape[0], 1), dtype=np.float32)

                    target_features_final_list.append(torch.tensor(target_features_final, dtype=torch.float32))
                    target_metadata_list.append(torch.tensor(target_metadata_final, dtype=torch.float32))
                    scan_angle_list.append(torch.tensor(scan_angle, dtype=torch.float32))
                    target_channel_mask_list.append(torch.tensor(target_channel_mask, dtype=torch.bool))
                    target_lat_deg_list.append(target_data['lat'])
                    target_lon_deg_list.append(target_data['lon'])

            # -------------------- Store final results --------------------
            data_summary_bin["input_features_final"] = torch.tensor(input_features_final, dtype=torch.float32)
            data_summary_bin["input_metadata"] = torch.tensor(np.column_stack([lat_rad_input, lon_rad_input]), dtype=torch.float32)
            data_summary_bin["input_lat_deg"] = input_lat_raw_clean
            data_summary_bin["input_lon_deg"] = input_lon_raw_clean

            if is_latent_rollout:
                data_summary_bin.update({
                    "target_features_final_list": target_features_final_list,
                    "target_metadata_list": target_metadata_list,
                    "scan_angle_list": scan_angle_list,
                    "target_channel_mask_list": target_channel_mask_list,
                    "target_lat_deg_list": target_lat_deg_list,
                    "target_lon_deg_list": target_lon_deg_list
                })
            else:
                data_summary_bin.update({
                    "target_features_final": target_features_final_list[0],
                    "target_metadata": target_metadata_list[0],
                    "scan_angle": scan_angle_list[0],
                    "target_channel_mask": target_channel_mask_list[0],
                    "target_lat_deg": target_lat_deg_list[0],
                    "target_lon_deg": target_lon_deg_list[0]
                })

            NAME2ID = _name2id(observation_config)
            data_summary_bin["instrument_id"] = NAME2ID[inst_name]

            print(f"[{bin_name}] {inst_name}: input {orig_in} -> "
                  f"{input_features_final.shape[0]}, targets {orig_tg_sizes} -> "
                  f"{[t.shape[0] for t in target_features_final_list]}")

        if not data_summary[bin_name].get(obs_type):
            del data_summary[bin_name][obs_type]

    return data_summary
