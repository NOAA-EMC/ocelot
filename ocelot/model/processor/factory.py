from ocelot.configs.model_config import ProcessorConfig
from ocelot.model.graph_schema import GraphSchema
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

def make(mesh : Mesh, processor_config: ProcessorConfig, graph_schema: GraphSchema) -> ProcessorBase:
    if processor_config.type not in processor_types:
        raise ValueError(f"Unknown processor_type: {processor_config.type}")
        
    print (f"Created {processor_config.type}.")
    return processor_types[processor_config.type](mesh, processor_config, graph_schema)
