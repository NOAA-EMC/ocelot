from ocelot.configs.model_config import ProcessorConfig
from ocelot.model.processor.processor_base import ProcessorBase
from ocelot.model.processor.interaction_processor import InteractionProcessor
from ocelot.model.processor.sliding_window_transformer import SlidingWindowTransformer
from ocelot.model.processor.hierarchical_interaction_processor import HierarchicalInteractionProcessor
from ocelot.model.processor.hierarchical_sliding_window_transformer import HierarchicalSlidingWindowTransformer
from ocelot.model.mesh.mesh import Mesh

processor_types = {
    "interaction": InteractionProcessor,
    "sliding_window": SlidingWindowTransformer,
    "hierarchical_interaction": HierarchicalInteractionProcessor,
    "hierarchical_sliding_window": HierarchicalSlidingWindowTransformer
}

def make(mesh : Mesh, config: ProcessorConfig) -> ProcessorBase:
    if config.type not in processor_types:
        raise ValueError(f"Unknown processor_type: {config.type}")
        
    print (f"Created {config.type}.")
    return processor_types[config.type](mesh, config)
