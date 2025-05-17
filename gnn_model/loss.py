import torch

def normalized_level_weights(pressure_levels):
    """Weights proportional to pressure at each level."""
    return pressure_levels / pressure_levels.mean()

def level_weighted_mse(predictions, targets, pressure_levels=None):
    """
    Compute level-weighted MSE loss.
    Args:
        predictions (torch.Tensor): Predicted values
        targets (torch.Tensor): Target values
        pressure_levels (torch.Tensor, optional): Pressure values for each channel.
            If None, will use default values from 1000mb to 200mb for 22 channels.
    Returns:
        torch.Tensor: Scalar loss value
    """
    # Default pressure levels if none provided
    # We don't have data with levels now. Using 22channels as levels to trick it
    # modify accordingly when dealing with data with levels
    if pressure_levels is None:
        level_num = predictions.shape[-1]  # 22
        # For 22 channels, create evenly spaced values from 1000 to 200
        pressure_levels = torch.linspace(1000, 200, level_num).to(predictions.device)

    print("MK LOSS DEBUDDING:")
    print(f"pressure levels: {pressure_levels}")
    print(f"prediction shape: {predictions.shape}")
    print(f"target shape: {targets.shape}")

    # Get normalized weights
    weights = normalized_level_weights(pressure_levels)
    print(f"weights shape: {weights.shape} \n weights:")
    print(weights)

    # Reshape weights for broadcasting
    weight_shape = [1] * (predictions.dim() - 1) + [weights.size(0)]
    weights = weights.view(*weight_shape)
    print(f"reshaped weights shape: {weights.shape}")

    # Compute squared difference and apply weights
    squared_diff = (predictions - targets)**2
    weighted_squared_diff = squared_diff * weights

    print(f"squared error shape: {squared_diff.shape}")
    print(squared_diff[:5, :])
    print(f"weighted squared error shape: {weighted_squared_diff.shape}")
    print(weighted_squared_diff[:5, :])

    # Calculate mean loss
    loss = weighted_squared_diff.mean()
    print(f"loss shape: {loss.shape} \n loss: {loss}")

    return loss
