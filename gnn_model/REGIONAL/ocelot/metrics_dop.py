# Third-party
import torch
from ocelot.logger import logger

# logger = setup_logger(__name__)


def get_metric(metric_name):
    """
    Get a defined metric with given name

    metric_name: str, name of the metric

    Returns:
    metric: function implementing the metric
    """
    metric_name_lower = metric_name.lower()
    assert (
        metric_name_lower in DEFINED_METRICS
    ), f"Unknown metric: {metric_name}"
    return DEFINED_METRICS[metric_name_lower]


def mask_and_reduce_metric(metric_entry_vals, mask, average_grid, sum_vars):
    """
    Masks and (optionally) reduces entry-wise metric values

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    metric_entry_vals: (..., N, d_state), prediction
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Apply the mask to select only the valid entries.
    if mask is not None:
        # This will produce a 1D tensor of the valid metric entries.
        metric_entry_vals = metric_entry_vals[mask]

    # If the mask was all False, the tensor will be empty. Return 0.
    if metric_entry_vals.numel() == 0:
        return torch.tensor(0.0, device=metric_entry_vals.device)

    # Reduce the remaining values to a single mean value.
    if average_grid:
        metric_entry_vals = torch.mean(metric_entry_vals)
    # The sum_vars argument is no longer needed as we are averaging over
    # everything.

    return metric_entry_vals


def wmse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Weighted Mean Squared Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Log shapes before computing MSE
    logger.debug(
        f"wmse - pred shape: {tuple(pred.shape)}, target shape: {tuple(target.shape)}, "
        f"pred_std shape: {tuple(pred_std.shape)}")

    entry_mse = torch.nn.functional.mse_loss(
        pred, target, reduction="none"
    )  # (..., N, d_state)

    print("broadcasted pred_std:",
          (pred_std[None, None, :]).shape)  # (1, 1, 22)
    entry_mse_weighted = entry_mse / (pred_std**2)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_mse_weighted,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def mse(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    (Unweighted) Mean Squared Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Log shapes before computing MSE
    logger.debug(
        f"mse - pred shape: {tuple(pred.shape)}, target shape: {tuple(target.shape)}, "
        f"pred_std shape: {tuple(pred_std.shape)}")

    # Replace pred_std with constant ones
    return wmse(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def wmae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Weighted Mean Absolute Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    entry_mae = torch.nn.functional.l1_loss(
        pred, target, reduction="none"
    )  # (..., N, d_state)
    entry_mae_weighted = entry_mae / pred_std  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_mae_weighted,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars,
    )


def mae(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    (Unweighted) Mean Absolute Error

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Replace pred_std with constant ones
    return wmae(
        pred, target, torch.ones_like(pred_std), mask, average_grid, sum_vars
    )


def nll(pred, target, pred_std, mask=None, average_grid=True, sum_vars=True):
    """
    Negative Log Likelihood loss, for isotropic Gaussian likelihood

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    # Broadcast pred_std if shaped (d_state,), done internally in Normal class
    dist = torch.distributions.Normal(pred, pred_std)  # (..., N, d_state)
    entry_nll = -dist.log_prob(target)  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_nll, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


def crps_gauss(
    pred, target, pred_std, mask=None, average_grid=True, sum_vars=True
):
    """
    (Negative) Continuous Ranked Probability Score (CRPS)
    Closed-form expression based on Gaussian predictive distribution

    (...,) is any number of batch dimensions, potentially different
            but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    std_normal = torch.distributions.Normal(
        torch.zeros((), device=pred.device), torch.ones((), device=pred.device)
    )
    target_standard = (target - pred) / pred_std  # (..., N, d_state)

    entry_crps = -pred_std * (
        torch.pi ** (-0.5)
        - 2 * torch.exp(std_normal.log_prob(target_standard))
        - target_standard * (2 * std_normal.cdf(target_standard) - 1)
    )  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_crps, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


def huber(
        pred,
        target,
        pred_std,
        mask=None,
        average_grid=True,
        sum_vars=True,
        delta=1.0):
    """
    Huber loss (Smooth L1 loss)
    Combines MSE for small errors and MAE for large errors

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev. (not used)
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)
    delta: float, threshold where loss transitions from quadratic to linear

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    diff = pred - target  # (..., N, d_state)
    abs_diff = torch.abs(diff)

    # Huber loss formula
    entry_huber = torch.where(
        abs_diff <= delta,
        0.5 * diff ** 2,  # Quadratic for small errors
        delta * (abs_diff - 0.5 * delta)  # Linear for large errors
    )  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_huber, mask=mask, average_grid=average_grid, sum_vars=sum_vars
    )


def whuber(
        pred,
        target,
        pred_std,
        mask=None,
        average_grid=True,
        sum_vars=True,
        delta=1.0):
    """
    Weighted Huber loss (uncertainty-weighted)
    Divides Huber loss by predicted standard deviation

    (...,) is any number of batch dimensions, potentially different
        but broadcastable
    pred: (..., N, d_state), prediction
    target: (..., N, d_state), target
    pred_std: (..., N, d_state) or (d_state,), predicted std.-dev.
    mask: (N,), boolean mask describing which grid nodes to use in metric
    average_grid: boolean, if grid dimension -2 should be reduced (mean over N)
    sum_vars: boolean, if variable dimension -1 should be reduced (sum
        over d_state)
    delta: float, threshold where loss transitions from quadratic to linear

    Returns:
    metric_val: One of (...,), (..., d_state), (..., N), (..., N, d_state),
    depending on reduction arguments.
    """
    diff = pred - target  # (..., N, d_state)
    abs_diff = torch.abs(diff)

    # Huber loss formula
    entry_huber = torch.where(
        abs_diff <= delta,
        0.5 * diff ** 2,  # Quadratic for small errors
        delta * (abs_diff - 0.5 * delta)  # Linear for large errors
    )  # (..., N, d_state)

    # Weight by uncertainty
    entry_huber_weighted = entry_huber / pred_std  # (..., N, d_state)

    return mask_and_reduce_metric(
        entry_huber_weighted,
        mask=mask,
        average_grid=average_grid,
        sum_vars=sum_vars)


def gradient_loss(pred, target, edge_index, mask=None):
    """
    Edge/gradient loss: scores the error in the value *difference* between
    neighboring grid nodes, rather than the raw per-node value. A plain per-node
    MSE is minimized by matching the mean and tends to blur sharp transitions
    (fronts, coastlines); penalizing the mismatch in neighbor-to-neighbor
    differences pushes the model to reproduce those discontinuities directly.

    pred, target: (N, d_state)
    edge_index: (2, E) neighbor pairs (i, j) indexing into pred/target's node
        ordering, e.g. a k-NN graph over the target grid. None or empty disables
        the term (returns 0).
    mask: (N, d_state) boolean valid mask, optional. An edge only contributes
        where both endpoints are valid.

    Returns a scalar tensor.
    """
    if edge_index is None or edge_index.numel() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    i, j = edge_index[0], edge_index[1]
    pred_diff = pred[i] - pred[j]
    target_diff = target[i] - target[j]
    entry_sq = (pred_diff - target_diff) ** 2
    if mask is not None:
        edge_valid = mask[i] & mask[j]
        entry_sq = entry_sq[edge_valid]
    if entry_sq.numel() == 0:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return entry_sq.mean()


def multiscale_loss(pred, target, pos, mask=None, scales=(0.5, 1.0, 2.0, 4.0)):
    """
    Multi-scale loss: bins nodes by lon/lat into cells at each of ``scales``
    (degrees) and scores the error between the cell-averaged prediction and
    cell-averaged target, in addition to the main per-node loss. This pushes the
    model to get large-scale (synoptic) structure right, not just the per-node
    average, which a per-node loss alone can satisfy while still misplacing
    coarse-scale features.

    pred, target: (N, d_state)
    pos: (N, 2) [lon, lat] in degrees, same node ordering as pred/target.
    mask: (N, d_state) boolean valid mask, optional.
    scales: bin sizes in degrees.

    Returns a scalar tensor (mean over scales).
    """
    n = pred.shape[0]
    if n == 0:
        return torch.zeros((), device=pred.device, dtype=torch.float32)
    # Accumulate in float32 regardless of pred's dtype: index_add requires the
    # accumulator and source to match dtype exactly, and under AMP autocast pred
    # is Half while target (built directly from the dataset) is Float32 — mixing
    # them crashes index_add even though ordinary arithmetic would silently
    # promote. Float32 also avoids overflow summing many nodes into one bin.
    pred = pred.float()
    target = target.float()
    lon, lat = pos[:, 0], pos[:, 1]
    valid = mask.float() if mask is not None else torch.ones_like(pred)
    pred_valid = pred * valid
    target_valid = target * valid

    losses = []
    for scale in scales:
        lon_bin = torch.floor(lon / scale).long()
        lat_bin = torch.floor(lat / scale).long()
        lon_bin = lon_bin - lon_bin.min()
        lat_bin = lat_bin - lat_bin.min()
        # Row-major combination into a single bin id (like
        # np.ravel_multi_index).
        bin_id = lon_bin * (lat_bin.max() + 1) + lat_bin
        _, inverse = torch.unique(bin_id, return_inverse=True)
        n_bins = int(inverse.max().item()) + 1

        sums_pred = torch.zeros(
            n_bins,
            pred.shape[1],
            device=pred.device,
            dtype=torch.float32)
        sums_target = torch.zeros(
            n_bins,
            pred.shape[1],
            device=pred.device,
            dtype=torch.float32)
        counts = torch.zeros(
            n_bins,
            pred.shape[1],
            device=pred.device,
            dtype=torch.float32)
        sums_pred = sums_pred.index_add(0, inverse, pred_valid)
        sums_target = sums_target.index_add(0, inverse, target_valid)
        counts = counts.index_add(0, inverse, valid)

        has_data = counts > 0
        if not torch.any(has_data):
            continue
        pred_avg = sums_pred[has_data] / counts[has_data]
        target_avg = sums_target[has_data] / counts[has_data]
        losses.append(torch.nn.functional.mse_loss(pred_avg, target_avg))

    if not losses:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return torch.stack(losses).mean()


DEFINED_METRICS = {
    "mse": mse,
    "mae": mae,
    "wmse": wmse,
    "wmae": wmae,
    "nll": nll,
    "crps_gauss": crps_gauss,
    "huber": huber,
    "whuber": whuber,
}
