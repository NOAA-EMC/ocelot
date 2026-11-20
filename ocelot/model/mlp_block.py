"""Shared tensor and neural-network utilities for OCELOT.

Author: Azadeh Gholoubi

This module contains the dtype constant and MLP builder used by the model and
mesh graph construction code.
"""

import torch
from torch import nn

DEFAULT_DTYPE = torch.float32


def make(blueprint, layer_norm=True, output_activation=None):
    """
    Create MLP from list blueprint, with
    input dimensionality: blueprint[0]
    output dimensionality: blueprint[-1] and
    hidden layers of dimensions: blueprint[1], ..., blueprint[-2]

    if layer_norm is True, includes a LayerNorm layer at
    the output (as used in GraphCast)
    """
    hidden_layers = len(blueprint) - 2
    assert hidden_layers >= 0, "Invalid MLP blueprint"

    layers = []
    for layer_i, (dim1, dim2) in enumerate(zip(blueprint[:-1], blueprint[1:])):
        layers.append(nn.Linear(dim1, dim2))
        if layer_i != hidden_layers:
            layers.append(nn.LayerNorm(dim2))  # Normalize before activation
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

    # Optionally add output activation
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)
