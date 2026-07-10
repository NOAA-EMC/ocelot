
import yaml
from .config_base import ConfigBase, Choices, IntField, FloatField, BoolField, StrField, Optional


class MeshConfig(ConfigBase):
    type = Choices(['fixed', 'hierarchical'])
    resolution = IntField()
    levels = IntField()


class CoderConfig(ConfigBase):
    type = Choices(['gat', 'interaction'])
    layers = IntField()
    heads = IntField()
    dropout = FloatField()
    scan_angle_conditioning = Optional(Choices(['pad', 'project']), default='project')
    pressure_level_conditioning = Optional(Choices(['pad', 'project']), default='project')
    disable_bipartite_edge_attr = Optional(BoolField(), default=False)
    bipartite_edge_attr_dim = Optional(IntField(), default=4)


class ProcessorConfig(ConfigBase):
    type = Choices(['sliding_transformer', 
                    'interaction', 
                    'hierarchical', 
                    'hierarchical_transformer'])
    num_layers = IntField()
    depth = IntField()
    heads = IntField()
    window = IntField()
    dropout = FloatField()


class EmbeddingsConfig(ConfigBase):
    scan_angle_dim = IntField()
    pressure_level_dim = IntField()
    target_time_dim = IntField()
    num_pressure_levels = IntField()


class ModelConfig(ConfigBase):
    hidden_dim = IntField()
    mesh = MeshConfig()
    encoder = CoderConfig()
    processor = ProcessorConfig()
    decoder = CoderConfig()
    embeddings = EmbeddingsConfig()

    def __init__(self, config_path: str):
        super().load(yaml.safe_load(open(config_path)))
