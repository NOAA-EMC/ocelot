from .processor_base import ProcessorBase
from .interaction_processor import InteractionProcessor
from .sliding_window_transformer import SlidingWindowTransformer
from .hierarchical_interaction_processor import HierarchicalInteractionProcessor
from .hierarchical_sliding_window_transformer import HierarchicalSlidingWindowTransformer
from ..mesh.mesh import Mesh

processor_types = {
    "interaction": InteractionProcessor,
    "sliding_window": SlidingWindowTransformer,
    "hierarchical_interaction": HierarchicalInteractionProcessor,
    "hierarchical_sliding_window": HierarchicalSlidingWindowTransformer
}

class ProcessorFactory:
    @staticmethod
    def build(processor_type: str, mesh : Mesh, params: dict) -> ProcessorBase:
        if processor_type not in processor_types:
            raise ValueError(f"Unknown processor_type: {processor_type}")
        print (f"Created {processor_type}.")
        return processor_types[processor_type](mesh, **params)
