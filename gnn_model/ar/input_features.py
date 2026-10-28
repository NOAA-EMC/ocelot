"""Input-row packing for AR synthetic feedback.

Builds rows in the same column layout as real encoder inputs assembled in
process_timeseries.py: [7 geo/time | metadata | channels | sat_id one-hot].
"""

from typing import Optional
import numpy as np
import torch


GEO_TIME_DIM = 7          # sin/cos lat, sin/cos lon, sin/cos time-of-day, day-of-year fraction
CONV_ZLIM, CONV_SENT = 6.0, -9.0   # conventional clip and missing-channel sentinel (process_timeseries)


def _instrument_layout(observation_config: dict, inst_name: str, input_dim: int) -> dict:
    """Column layout of a real input row, as assembled in process_timeseries.py:
        [7 geo/time | metadata | channels | sat_id one-hot (satellite only)]
    """
    for obs_type in ("satellite", "conventional"):
        cfg = (observation_config or {}).get(obs_type, {}).get(inst_name)
        if cfg is not None:
            break
    else:
        raise KeyError(f"{inst_name}: not found in observation_config")

    n_meta = len(cfg.get("metadata") or [])
    n_ch = len(cfg.get("features") or [])
    sat_ids = [int(s) for s in (cfg.get("sat_ids") or [])] if obs_type == "satellite" else []
    k = len(sat_ids)
    if GEO_TIME_DIM + n_meta + n_ch + k != input_dim:
        raise ValueError(
            f"{inst_name}: layout 7 + {n_meta} meta + {n_ch} ch + {k} sat = "
            f"{GEO_TIME_DIM + n_meta + n_ch + k} does not match input_dim={input_dim}"
        )
    meta0 = GEO_TIME_DIM
    ch0 = meta0 + n_meta
    sat0 = ch0 + n_ch
    return dict(obs_type=obs_type, n_meta=n_meta, n_ch=n_ch, k=k,
                meta0=meta0, ch0=ch0, sat0=sat0, sat_ids=sat_ids)


def _geo_time_block(lat_deg: torch.Tensor, lon_deg: torch.Tensor, times_unix: Optional[torch.Tensor]) -> torch.Tensor:
    """Same encoding as process_timeseries.py input rows. Returns [N, 7] float32, or [N, 4] if no times."""
    lat = torch.deg2rad(lat_deg.float())
    lon = torch.deg2rad(lon_deg.float())
    geo = torch.stack([torch.sin(lat), torch.cos(lat), torch.sin(lon), torch.cos(lon)], dim=1)
    if times_unix is None:
        return geo
    t = times_unix.detach().cpu().numpy().astype(np.int64)
    dt = t.astype("datetime64[s]")
    year_start = dt.astype("datetime64[Y]").astype("datetime64[s]")
    secs_into_year = (dt - year_start).astype(np.int64)
    day_of_year = secs_into_year / 86400.0 / 365.24219              # (yday-1 + frac_of_day) / 365.24219
    frac_of_day = (t % 86400) / 86400.0
    tt = np.stack([np.sin(2 * np.pi * frac_of_day), np.cos(2 * np.pi * frac_of_day), day_of_year], axis=1)
    return torch.cat([geo, torch.from_numpy(tt.astype(np.float32)).to(geo.device)], dim=1)


def _build_input_rows(
    pred: torch.Tensor,
    chan_mask: torch.Tensor,
    tgt,
    template_x: torch.Tensor,
    layout: dict,
    inst_name: str,
) -> torch.Tensor:
    """Pack predictions at prev-window target nodes into rows with the real input layout."""
    N = pred.shape[0]
    device, dtype = template_x.device, template_x.dtype
    row = torch.zeros(N, template_x.shape[1], dtype=torch.float32, device=device)
    tmpl = template_x.float()

    # geo/time (cols 0-6) from the prediction's own location and valid time
    times = getattr(tgt, "obs_time_unix", None)
    times = times[:N] if times is not None and times.numel() >= N and bool((times[:N] >= 0).all()) else None
    gt = _geo_time_block(tgt.lat[:N].to(device), tgt.lon[:N].to(device), times)
    row[:, :gt.shape[1]] = gt
    if gt.shape[1] < GEO_TIME_DIM:   # no valid times: fall back to template time columns
        row[:, 4:GEO_TIME_DIM] = tmpl[:, 4:GEO_TIME_DIM].mean(dim=0)

    # metadata: target_metadata = [lat_rad, lon_rad, meta (same transform as inputs), 5 time feats]
    n_meta, m0 = layout["n_meta"], layout["meta0"]
    if n_meta:
        tm = getattr(tgt, "target_metadata", None)
        if tm is not None and tm.ndim == 2 and tm.shape[0] >= N and tm.shape[1] >= 2 + n_meta:
            meta = tm[:N, 2:2 + n_meta].to(device).float().clone()
            if inst_name == "ssmis" and n_meta > 1:
                # inputs keep ssmis metadata col 1 raw; targets store cos(deg2rad(raw))
                meta[:, 1] = torch.rad2deg(torch.arccos(meta[:, 1].clamp(-1.0, 1.0)))
            row[:, m0:m0 + n_meta] = meta
        else:
            row[:, m0:m0 + n_meta] = tmpl[:, m0:m0 + n_meta].mean(dim=0)

    # channels: predictions, missing channels encoded as in preprocessing
    c0, n_ch = layout["ch0"], layout["n_ch"]
    p = pred[:, :n_ch].float()
    m = chan_mask[:, :n_ch].to(device)
    if layout["obs_type"] == "conventional":
        p = torch.where(m, p.clamp(-CONV_ZLIM, CONV_ZLIM), torch.full_like(p, CONV_SENT))
    else:
        p = torch.where(m, p, torch.zeros_like(p))
    row[:, c0:c0 + n_ch] = p

    # sat_id one-hot from each target row's own satellite ID
    k, s0 = layout["k"], layout["sat0"]
    if k:
        sids = getattr(tgt, "sat_ids", None)
        lut = torch.full((max(layout["sat_ids"]) + 1,), -1, dtype=torch.long, device=device)
        lut[torch.tensor(layout["sat_ids"], device=device)] = torch.arange(k, device=device)
        if sids is not None and sids.numel() >= N:
            s = sids[:N].to(device).long()
            pos = torch.full_like(s, -1)
            ok = (s >= 0) & (s < lut.numel())
            pos[ok] = lut[s[ok]]
        else:
            pos = torch.full((N,), -1, dtype=torch.long, device=device)
        unknown = pos < 0
        if bool(unknown.any()):
            # No ID available: fall back to the template window's most common satellite
            onehot = tmpl[:, s0:s0 + k]
            pos[unknown] = int(onehot.sum(dim=0).argmax()) if onehot.shape[0] else 0
            print(f"[SATDIAG] {inst_name}: {int(unknown.sum())}/{N} rows without a usable sat_id "
                  f"(sat_ids attr {'missing' if sids is None else tuple(sids.shape)}); used fallback", flush=True)
        row[torch.arange(N, device=device), s0 + pos] = 1.0

    return row.to(dtype)
