
import yaml

from ocelot.configs.config_base import (
    BoolField,
    Choices,
    ConfigBase,
    FloatField,
    IntField,
    ListField,
    Optional,
)

from .observation_config import ObservationConfig


## Mesh configurations ##

class MeshConfig(ConfigBase):
    levels = IntField()
    resolution = IntField()


class FixedMeshConfig(MeshConfig):
    pass


class HierarchicalMeshConfig(MeshConfig):
    pass


## Coder (Encoder/Decoder) configurations ##

class CoderConfig(ConfigBase):
    rec_dim = IntField()
    send_dim = IntField()


class GatCoderConfig(CoderConfig):
    send_dim: IntField()
    rec_dim: IntField()
    hidden_dim = IntField()
    layers = Optional(IntField(), default=2)
    heads = Optional(IntField(), default=4)
    dropout = Optional(FloatField(), default=0.0)
    edge_dim = Optional(IntField())
    dst_chunk_size = Optional(IntField())
    dst_chunk_threshold = Optional(IntField(), default=20_000)
    use_activation_checkpointing = Optional(BoolField(), default=True)


class InteractionCoderConfig(CoderConfig):
    send_dim = IntField()
    rec_dim = IntField()
    edge_index = Optional(ListField(ListField(IntField())), default=None)  # Edge indices in PyG format (2, M)
    update_edges = Optional(BoolField(), default=False)
    hidden_layers = Optional(IntField(), default=1)
    hidden_dim = Optional(IntField())
    edge_chunk_sizes = Optional(ListField(IntField()))
    aggr_chunk_sizes = Optional(ListField(IntField()))
    aggr = Optional(Choices(['sum', 'mean']), default='sum')

    def load(self, config_dict: dict) -> None:
        super().load(config_dict)
        if self.hidden_dim is None:
            self.hidden_dim = self.rec_dim


## Processor configurations ##

class ProcessorConfig(ConfigBase):
    pass


class TransformerProcessorConfig(ProcessorConfig):
    hidden_dim = IntField()
    window = Optional(IntField(), default=4)
    depth = Optional(IntField(), default=2)
    num_heads = Optional(IntField(), default=4)
    dropout = Optional(FloatField(), default=0.0)
    use_causal_mask = Optional(BoolField(), default=True)
    spatial_mixing_steps = Optional(IntField(), default=1)


class SlidingWindowProcessorConfig(TransformerProcessorConfig):
    pass


class InteractionProcessorConfig(ProcessorConfig):
    num_message_passing_steps = IntField()


class HierarchicalInteractionProcessorConfig(ProcessorConfig):
    num_message_passing_steps = Optional(IntField(), default=4)


class HierarchicalSlidingWindowProcessorConfig(TransformerProcessorConfig):
    use_cross_scale = Optional(BoolField(), default=True)


## Embeddings configurations ##

class EmbeddingsConfig(ConfigBase):
    scan_angle_dim = Optional(IntField(), default=8)
    scan_angle_conditioning = Optional(Choices(['pad', 'project']), default='project')
    pressure_level_dim = Optional(IntField(), default=8)
    pressure_level_conditioning = Optional(Choices(['pad', 'project']), default='project')
    target_time_dim = Optional(IntField(), default=8)
    num_pressure_levels = Optional(IntField(), default=16)


## Model configuration ##

class ModelConfig(ConfigBase):
    hidden_dim = IntField()
    latent_step_hours = IntField()
    
    mesh = Choices({
        'fixed': FixedMeshConfig(),
        'hierarchical': HierarchicalMeshConfig(),
    })
    encoder = Choices({
        'gat': GatCoderConfig(),
        'interaction': InteractionCoderConfig(),
    })
    processor = Choices({
        'interaction': InteractionProcessorConfig(),
        'sliding_window': SlidingWindowProcessorConfig(),
        'hierarchical_interaction': HierarchicalInteractionProcessorConfig(),
        'hierarchical_sliding_window': HierarchicalSlidingWindowProcessorConfig(),
    })
    decoder = Choices({
        'gat': GatCoderConfig(),
        'interaction': InteractionCoderConfig(),
    })
    embeddings = EmbeddingsConfig()

    def __init__(self, config_path: str):
        super().__init__()
        with open(config_path) as config_file:
            self.load(yaml.safe_load(config_file))

        self.flat_processors = {'interaction', 'sliding_window'}
        self.mesh_is_fixed = self.mesh.type == 'fixed'
        self.processor_is_flat = self.processor.type in self.flat_processors
        if self.mesh_is_fixed != self.processor_is_flat:
            raise ValueError(
                f"Processor type '{self.processor.type}' is incompatible with "
                f"mesh type '{self.mesh.type}'"
            )
