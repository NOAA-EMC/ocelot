# Standard library

import importlib
import torch
from torch import nn

DEFAULT_DTYPE = torch.float32


def to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.cpu()
    if isinstance(obj, dict):
        return {k: to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_cpu(i) for i in obj]
    return obj


def make_mlp(blueprint, layer_norm=True, output_activation=None):
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
            # Add a small epsilon to LayerNorm to prevent division by zero
            layers.append(nn.LayerNorm(dim2, eps=1e-6))  # Normalize after activation
            layers.append(nn.SiLU())  # Swish activation
            

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1], eps=1e-6))

    # Optionally add output activation
    if output_activation is not None:
        layers.append(output_activation)

    return nn.Sequential(*layers)


def load_class_from_path(class_path: str):
    """
    Dynamically load a class from a full class path string.

    Args:
        class_path (str): Full path to the class, e.g. "mypackage.mymodule.MyClass"

    Returns:
        type: The class object.
    """
    try:
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        cls = getattr(module, class_name)
        return cls
    except (ImportError, AttributeError, ValueError) as e:
        raise ImportError(f"Could not import '{class_path}': {e}")


def instantiate_class_from_path(class_path: str, *args, **kwargs):
    cls = load_class_from_path(class_path)
    return cls(*args, **kwargs)

