
import yaml
from .config_base import ConfigBase, Choices, IntField, FloatField, StrField, Optional


class MeshConfig(ConfigBase):
    type = Choices(['fixed', 'hierarchical'])
    resolution = IntField()
    levels = IntField()


class CoderConfig(ConfigBase):
    type = Choices(['gat', 'interaction'])
    layers = IntField()
    heads = IntField()
    dropout = FloatField()


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


model_config = ModelConfig("gnn_model/configs/model_config.yaml")
