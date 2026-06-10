
from ..mesh.fixed_mesh import FixedMesh
from .processor_base import ProcessorBase


class FlatProcessorBase(ProcessorBase):
    def __init__(
        self,
        mesh : FixedMesh
    ):
        if not isinstance(mesh, FixedMesh):
            raise TypeError(f"Expected FixedMesh, got {type(mesh).__name__}")
        
        super().__init__(mesh)
    