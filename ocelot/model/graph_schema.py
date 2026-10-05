"""Node and edge type naming for the OCELOT heterogeneous observation graph."""

from typing import Iterable, List, Tuple

EdgeType = Tuple[str, str, str]


class GraphSchema:
    """Single owner of the node and edge type names shared across model components."""

    MESH = "mesh"

    def __init__(self, instrument_names: Iterable[str]):
        self.instrument_names = list(instrument_names)

    @staticmethod
    def input_node(inst_name: str) -> str:
        return f"{inst_name}_input"

    @staticmethod
    def target_node(inst_name: str) -> str:
        return f"{inst_name}_target"

    def encoder_edge(self, inst_name: str) -> EdgeType:
        return (self.input_node(inst_name), "to", self.MESH)

    def decoder_edge(self, inst_name: str) -> EdgeType:
        return (self.MESH, "to", self.target_node(inst_name))

    @property
    def mesh_edge(self) -> EdgeType:
        return (self.MESH, "to", self.MESH)

    @property
    def processor_edge_types(self) -> List[EdgeType]:
        return [self.mesh_edge] + [self.encoder_edge(inst_name) for inst_name in self.instrument_names]

    @property
    def node_types(self) -> List[str]:
        types = [self.MESH]
        for inst_name in self.instrument_names:
            types.extend([self.input_node(inst_name), self.target_node(inst_name)])
        return types

    @property
    def edge_types(self) -> List[EdgeType]:
        types = [self.mesh_edge]
        for inst_name in self.instrument_names:
            types.extend([self.encoder_edge(inst_name), self.decoder_edge(inst_name)])
        return types
