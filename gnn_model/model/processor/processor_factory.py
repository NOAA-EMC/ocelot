from configs.model_config import ProcessorConfig
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
    def build(mesh : Mesh, hidden_dim: int, processor_config: ProcessorConfig) -> ProcessorBase:
        if processor_config.type not in processor_types:
            raise ValueError(f"Unknown processor_type: {processor_config.type}")
            
        print (f"Created {processor_config.type}.")

        return processor_types[processor_config.type](mesh, hidden_dim, processor_config)
