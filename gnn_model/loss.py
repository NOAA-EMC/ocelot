# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def normalized_level_weights(pressure_levels: torch.Tensor) -> torch.Tensor:
    """Weights proportional to pressure at each level (normalized by mean)."""
    return pressure_levels / (pressure_levels.mean() + 1e-8)


def level_weighted_mse(predictions: torch.Tensor, targets: torch.Tensor, pressure_levels: torch.Tensor | None = None) -> torch.Tensor:
    """
    Compute level-weighted MSE loss.

    If pressure_levels is None, creates an evenly spaced vector from 1000→200 mb
    with length equal to predictions.shape[-1].
    """
    if pressure_levels is None:
        level_num = predictions.shape[-1]
        pressure_levels = torch.linspace(1000, 200, level_num, device=predictions.device)

    weights = normalized_level_weights(pressure_levels)  # [C]
    weights = weights.view(*([1] * (predictions.dim() - 1)), -1)  # broadcast to [..., C]

    sq = (predictions - targets) ** 2
    return (sq * weights).mean()


def huber_per_element(pred: torch.Tensor, target: torch.Tensor, delta: float = 0.1) -> torch.Tensor:
    """
    Per-element Huber loss, shape == pred.shape.
    """
    # torch.nn.functional.huber_loss exists, but we need elementwise -> reduction='none'
    return F.huber_loss(pred, target, delta=delta, reduction="none")


def weighted_huber_loss(
    pred: torch.Tensor,  # [N, C]
    target: torch.Tensor,  # [N, C]
    instrument_ids: Optional[torch.Tensor] = None,  # [N] or None
    channel_weights=None,  # dict[int|str->Tensor[C]] or Tensor[C] or None
    delta: float = 0.1,
    rebalancing: bool = True,  # average equally across instruments if True
    valid_mask: Optional[torch.Tensor] = None,  # [N, C] bool; False = ignore element
) -> torch.Tensor:
    """
    Huber loss with optional per-instrument, per-channel weights and an optional [N, C] mask.
    When valid_mask is provided, loss is averaged ONLY over True elements.

    Behavior:
      - If instrument_ids is None → single group over all rows.
      - If rebalancing=True → mean of per-instrument means (each instrument contributes equally
        if it has at least 1 valid element).
      - If rebalancing=False → global mean over all valid elements across instruments.
    """
    device = pred.device
    N, C = pred.shape

    # elementwise Huber [N, C]
    huber = nn.HuberLoss(delta=delta, reduction="none")(pred, target)

    # Optional elementwise mask
    if valid_mask is not None:
        if valid_mask.shape != huber.shape:
            raise ValueError(f"valid_mask shape {valid_mask.shape} must match pred/target {huber.shape}.")
        vm = valid_mask.to(dtype=huber.dtype, device=device)
    else:
        vm = None

    # Helper to broadcast a channel-weight vector to [*, C]
    def _broadcast_w(w: torch.Tensor) -> torch.Tensor:
        w = w.to(device=device, dtype=huber.dtype).flatten()
        if w.numel() != C:
            if w.numel() < C:
                w = torch.cat([w, torch.ones(C - w.numel(), device=device, dtype=huber.dtype)], dim=0)
            else:
                w = w[:C]
        return w.view(1, C)  # broadcast over batch rows

    # Helper to fetch weights for an instrument (or global)
    def _get_weights_for_inst(inst_key) -> torch.Tensor:
        if isinstance(channel_weights, dict):
            w = channel_weights.get(inst_key)
            if w is None:
                w = channel_weights.get(str(inst_key))
            if w is None:
                w = channel_weights.get("global")
        else:
            w = channel_weights  # Tensor[C] or None
        if w is None:
            w = torch.ones(C, device=device, dtype=huber.dtype)
        w = torch.as_tensor(w, device=device, dtype=huber.dtype)
        w = torch.clamp(w, min=0)  # <- no negative weights
        return _broadcast_w(w)  # [1, C]

    eps = torch.finfo(huber.dtype).eps

    # Single-group path (no instrument ids)
    if instrument_ids is None:
        w = _get_weights_for_inst("global")  # [1, C]
        loss_mat = huber * w  # [N, C]
        if vm is not None:
            vm = vm.to(dtype=huber.dtype, device=device)
            active = vm * (w > 0).to(vm.dtype)  # exclude zero-weight chans
            loss_mat = loss_mat * active
            denom = active.sum()
        else:
            active = (w > 0).to(loss_mat.dtype, device=device).expand_as(loss_mat)
            denom = active.sum()
        if denom <= 0:
            return torch.tensor(0.0, device=device)
        return loss_mat.sum() / (denom + eps)

    # Per-instrument path
    total = torch.tensor(0.0, device=device, dtype=huber.dtype)
    denom_total = torch.tensor(0.0, device=device, dtype=huber.dtype)

    for inst in torch.unique(instrument_ids):
        mask_rows = instrument_ids == inst
        if not mask_rows.any():
            continue
        w = _get_weights_for_inst(int(inst.item()))  # [1, C]
        h_i = huber[mask_rows] * w  # [Ni, C]
        if vm is not None:
            vm_i = vm[mask_rows]
            active_i = vm_i * (w > 0).to(vm_i.dtype)
            h_i = h_i * active_i
            denom_i = active_i.sum()
        else:
            active_i = (w > 0).to(h_i.dtype).expand_as(h_i)
            denom_i = active_i.sum()

        if denom_i <= 0:
            continue

        if rebalancing:
            total = total + h_i.sum() / (denom_i + eps)  # equal weight per instrument
            denom_total = denom_total + 1.0
        else:
            total = total + h_i.sum()
            denom_total = denom_total + denom_i

    if denom_total <= 0:
        return torch.tensor(0.0, device=device)
    return total / (denom_total + eps)


def ocelot_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    instrument_ids: torch.Tensor,
    instrument_weights: dict[int, float] | dict[str, float] | None,
    channel_weights: dict[int, torch.Tensor] | dict[str, torch.Tensor] | None,
    channel_masks: dict[int, torch.Tensor] | None = None,
    channel_mean: torch.Tensor | None = None,
    channel_std: torch.Tensor | None = None,
) -> torch.Tensor:
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
        m = instrument_ids == inst

        y_p = pred[m]  # [n_i, C]
        y_t = target[m]  # [n_i, C]

        if channel_mean is not None and channel_std is not None:
            mean = channel_mean.to(device=device, dtype=y_p.dtype)
            std = channel_std.to(device=device, dtype=y_p.dtype)
            y_p = (y_p - mean) / (std + 1e-8)
            y_t = (y_t - mean) / (std + 1e-8)

        # weights & masks
        w_c = None
        if channel_weights is not None:
            w_c = channel_weights.get(inst_id, None) or channel_weights.get(str(inst_id), None)
            if w_c is not None and not torch.is_tensor(w_c):
                w_c = torch.as_tensor(w_c, device=device, dtype=y_p.dtype)
        if w_c is None:
            w_c = torch.ones(y_p.shape[1], device=device, dtype=y_p.dtype)

        if channel_masks is not None and inst_id in channel_masks:
            ch_mask = channel_masks[inst_id].to(device=device)
            y_p = y_p[:, ch_mask]
            y_t = y_t[:, ch_mask]
            w_c = w_c[ch_mask]

        per_ch_mse = ((y_p - y_t) ** 2).mean(dim=0)  # [C_used]
        weighted = (per_ch_mse * w_c).sum()  # scalar

        w_i = 1.0
        if instrument_weights is not None:
            w_i = instrument_weights.get(inst_id, None) or instrument_weights.get(str(inst_id), 1.0)

        total = total + w_i * weighted
        denom += max(y_p.shape[0], 1)

    return total / max(denom, 1)
