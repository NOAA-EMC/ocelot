"""Small GraphCast mesh helpers used by OCELOT.

Author: Azadeh Gholoubi

Only the active mesh-distance helper is kept for the v1.0 model path.
"""

import numpy as np

from modules.mesh import icosahedral_mesh


def _get_max_edge_distance(mesh):
    """Return the maximum Euclidean edge length in a triangular mesh."""
    senders, receivers = icosahedral_mesh.faces_to_edges(mesh.faces)
    edge_distances = np.linalg.norm(
        mesh.vertices[senders] - mesh.vertices[receivers], axis=-1
    )
    return edge_distances.max()
