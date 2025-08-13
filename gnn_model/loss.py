# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def normalized_level_weights(pressure_levels: torch.Tensor) -> torch.Tensor:
    """Weights proportional to pressure at each level (normalized by mean)."""
    return pressure_levels / (pressure_levels.mean() + 1e-8)


def level_weighted_mse(predictions: torch.Tensor,
                       targets: torch.Tensor,
                       pressure_levels: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute level-weighted MSE loss.

    If pressure_levels is None, creates an evenly spaced vector from 1000→200 mb
    with length equal to predictions.shape[-1].
    """
    if pressure_levels is None:
        level_num = predictions.shape[-1]
        pressure_levels = torch.linspace(1000, 200, level_num, device=predictions.device)

    weights = normalized_level_weights(pressure_levels)              # [C]
    weights = weights.view(*([1] * (predictions.dim() - 1)), -1)     # broadcast to [..., C]

    sq = (predictions - targets) ** 2
    return (sq * weights).mean()


def huber_per_element(pred: torch.Tensor,
                      target: torch.Tensor,
                      delta: float = 0.1) -> torch.Tensor:
    """
    Per-element Huber loss, shape == pred.shape.
    """
    # torch.nn.functional.huber_loss exists, but we need elementwise -> reduction='none'
    return F.huber_loss(pred, target, delta=delta, reduction="none")


def weighted_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    instrument_ids: torch.Tensor | None = None,
    channel_weights=None,           # dict[int|str->Tensor[C]] or Tensor[C] or None
    delta: float = 0.1,
    rebalancing: bool = True,       # average equally across instruments if True
) -> torch.Tensor:
    """
    Huber loss with optional per-instrument and per-channel weights.

    pred/target: [N, C]
    instrument_ids: [N] or None
    channel_weights:
        - dict keyed by instrument id (int or str) -> Tensor[C]
        - OR a single Tensor[C] to apply to all
        - OR None for uniform channel weighting
    """
    device = pred.device
    C = pred.shape[1]
    huber = nn.HuberLoss(delta=delta, reduction="none")(pred, target)  # [N, C]

    def _broadcast_w(w: torch.Tensor) -> torch.Tensor:
        w = w.to(device).flatten()
        if w.numel() != C:
            if w.numel() < C:
                w = torch.cat([w, torch.ones(C - w.numel(), device=device)], dim=0)
            else:
                w = w[:C]
        return w.view(1, C)  # for broadcasting over batch

    # No instrument IDs: apply a single (possibly global) channel weight or mean.
    if instrument_ids is None:
        # Accept dict with 'global', a Tensor[C], or None
        if isinstance(channel_weights, dict):
            w = channel_weights.get("global", None)
        else:
            w = channel_weights  # could be Tensor[C] or None

        if w is None:
            return huber.mean()

        w = _broadcast_w(w)
        return (huber * w).mean()

    # Per-instrument weighting
    total = torch.tensor(0.0, device=device)
    denom = 0.0
    unique_ids = torch.unique(instrument_ids)

    for inst in unique_ids:
        mask = (instrument_ids == inst)
        if not mask.any():
            continue

        key_int = int(inst.item())
        w = None
        if isinstance(channel_weights, dict):
            if key_int in channel_weights:
                w = channel_weights[key_int]
            else:
                key_str = str(key_int)
                if key_str in channel_weights:
                    w = channel_weights[key_str]
        elif channel_weights is not None:
            w = channel_weights  # Tensor[C] broadcast to all instruments

        if w is None:
            w = torch.ones(C, device=device)

        w = _broadcast_w(w)
        loss_i = (huber[mask] * w).mean()   # mean over samples and channels

        if rebalancing:
            total = total + loss_i
            denom += 1.0
        else:
            # sample-weighted average
            n_i = mask.sum().item()
            total = total + loss_i * n_i
            denom += n_i

    if denom == 0:
        return torch.tensor(0.0, device=device)
    return total / denom


def ocelot_loss(pred: torch.Tensor,
                target: torch.Tensor,
                instrument_ids: torch.Tensor,
                instrument_weights: dict[int, float] | dict[str, float] | None,
                channel_weights: dict[int, torch.Tensor] | dict[str, torch.Tensor] | None,
                channel_masks: dict[int, torch.Tensor] | None = None,
                channel_mean: torch.Tensor | None = None,
                channel_std: torch.Tensor | None = None) -> torch.Tensor:
    """
    Legacy Ocelot-style per-instrument MSE with channel masking/weights and optional normalization.

    - If channel_mean/std are provided (shape [C]), applies (x - mean) / std before loss.
    - If channel_masks contains boolean masks per instrument id, only masked channels are used.
    """
    device = pred.device
    total = 0.0
    denom = 0

    for inst in instrument_ids.unique():
        inst_id = int(inst.item())
        m = (instrument_ids == inst)

        y_p = pred[m]        # [n_i, C]
        y_t = target[m]      # [n_i, C]

        if channel_mean is not None and channel_std is not None:
            mean = channel_mean.to(device=device, dtype=y_p.dtype)
            std = channel_std.to(device=device, dtype=y_p.dtype)
            y_p = (y_p - mean) / (std + 1e-8)
            y_t = (y_t - mean) / (std + 1e-8)

        # weights & masks
        w_c = None
        if channel_weights is not None:
            w_c = (channel_weights.get(inst_id, None)
                   or channel_weights.get(str(inst_id), None))
            if w_c is not None and not torch.is_tensor(w_c):
                w_c = torch.as_tensor(w_c, device=device, dtype=y_p.dtype)
        if w_c is None:
            w_c = torch.ones(y_p.shape[1], device=device, dtype=y_p.dtype)

        if channel_masks is not None and inst_id in channel_masks:
            ch_mask = channel_masks[inst_id].to(device=device)
            y_p = y_p[:, ch_mask]
            y_t = y_t[:, ch_mask]
            w_c = w_c[ch_mask]

        per_ch_mse = ((y_p - y_t) ** 2).mean(dim=0)           # [C_used]
        weighted = (per_ch_mse * w_c).sum()                    # scalar

        w_i = 1.0
        if instrument_weights is not None:
            w_i = (instrument_weights.get(inst_id, None)
                   or instrument_weights.get(str(inst_id), 1.0))

        total = total + w_i * weighted
        denom += max(y_p.shape[0], 1)

    return total / max(denom, 1)
