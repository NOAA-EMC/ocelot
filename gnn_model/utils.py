from torch import nn
import numpy as np

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
            layers.append(nn.LayerNorm(dim2))  # Normalize before activation
            layers.append(nn.SiLU())  # Swish activation

    # Optionally add layer norm to output
    if layer_norm:
        layers.append(nn.LayerNorm(blueprint[-1]))

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


def random_keep_fraction_and_reindex(arrays, edge_index, edge_attr=None, keep_frac=0.8):
    """
    Randomly keep a fraction of rows from each array in arrays.
    Also reindex and mask the edge_index accordingly.
    If edge_attr is provided, mask it as well.
    Returns masked arrays, masked & reindexed edge_index, and (if given) masked edge_attr.
    """
    N = arrays[0].shape[0]
    if N == 0:
        if edge_attr is not None:
            return arrays, edge_index, edge_attr
        return arrays, edge_index
    keep_n = int(N * keep_frac)
    keep_idx = np.sort(np.random.choice(N, size=keep_n, replace=False))

    # Mask arrays
    arrays_masked = [arr[keep_idx] for arr in arrays]

    # Build mapping from old to new index
    old_to_new = -np.ones(N, dtype=int)
    old_to_new[keep_idx] = np.arange(keep_n)

    # Mask and remap edges
    src, dst = edge_index[0], edge_index[1]
    mask = np.isin(src, keep_idx) & np.isin(dst, keep_idx)
    src_new = old_to_new[src[mask]]
    dst_new = old_to_new[dst[mask]]
    edge_index_masked = np.stack([src_new, dst_new], axis=0)

    if edge_attr is not None:
        edge_attr_masked = edge_attr[mask]
        return arrays_masked, edge_index_masked, edge_attr_masked
    else:
        return arrays_masked, edge_index_masked
