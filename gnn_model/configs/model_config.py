
import yaml


class ConfigField:
    def __init__(self, value):
        self.value = value
        self._validate(value)

    def _validate(self, value):
        raise NotImplementedError("Must implement _validate method")

    def __call__(self):
        return self.value

class Choices(ConfigField):
    def __init__(self, choices: list[str], value):
        self.choices = choices
        super().__init__(value)

    def _validate(self, value):
        if value not in self.choices:
            raise ValueError(f"Value '{value}' not in allowed choices: {self.choices}")
        
class Optional(ConfigField):
    def __init__(self, inner_type, default=None):
        self.inner_type = inner_type
        self.default = default
        super().__init__(default)

    def _validate(self, value):
        if value is not None:
            self.inner_type._validate(value)

class IntField(ConfigField):
    def _validate(self, value):
        if not isinstance(value, int):
            raise ValueError(f"Expected int but got {type(value)}")
        
class FloatField(ConfigField):
    def __init__(self, value):
        super().__init__(value)

    def _validate(self, value):
        if not isinstance(value, float):
            raise ValueError(f"Expected float but got {type(value)}")

class StrField(ConfigField):
    def __init__(self, value):
        super().__init__(value)

    def _validate(self, value):
        if not isinstance(value, str):
            raise ValueError(f"Expected str but got {type(value)}")

# # meta class for all config classes, which will validate the fields and provide a way to access the fields
# class ConfigMeta(type):
#     def __new__(cls, name, bases, attrs):
#         fields = {k: v for k, v in attrs.items() if isinstance(v, ConfigField)}
#         attrs['_fields'] = fields
#         return super().__new__(cls, name, bases, attrs)
    
#     def __call__(cls, *args, **kwargs):
#         instance = super().__call__(*args, **kwargs)
#         instance._validate()
#         return instance

class ConfigBase:
    def __init__(self, name: str):
        self._name = name
        
    def _validate(self):
        for field_name, field_type in self._fields.items():
            if isinstance(field_type, ConfigField):
                value = getattr(self, field_name)
                field_type._validate(value)

    def __post_init__(self):
        self._validate()
    
    # return the value of the field
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if isinstance(attr, ConfigField):
            return attr()
        return attr


class MeshConfig(ConfigBase):
    def __init__(self, config_dict):
        self.type = Choices(['fixed', 'hierarchical'], config_dict['type'])
        self.resolution = IntField(config_dict['resolution'])
        self.levels = IntField(config_dict['levels'])


class CoderConfig(ConfigBase):
    def __init__(self, config_dict):
        self.type = Choices(['gat', 'interaction'], config_dict['type']) # gat | interaction
        self.layers = IntField(config_dict['layers'])
        self.heads = IntField(config_dict['heads'])
        self.dropout = FloatField(config_dict['dropout'])


class ProcessorConfig(ConfigBase):
    def __init__(self, config_dict):
        self.type = Choices(['sliding_transformer', 'interaction', 'hierarchical', 'hierarchical_transformer'], config_dict['type'])
        self.num_layers = IntField(config_dict['num_layers'])
        self.depth = IntField(config_dict['depth'])
        self.heads = IntField(config_dict['heads'])
        self.window = IntField(config_dict['window'])
        self.dropout = FloatField(config_dict['dropout'])


class EmbeddingsConfig(ConfigBase):
    def __init__(self, config_dict):
        self.scan_angle_dim = IntField(config_dict['scan_angle_dim'])
        self.pressure_level_dim = IntField(config_dict['pressure_level_dim'])
        self.target_time_dim = IntField(config_dict['target_time_dim'])
        self.num_pressure_levels = IntField(config_dict['num_pressure_levels'])


class ModelConfig(ConfigBase):
    def __init__(self, config_dict):
        self.hidden_dim = IntField(config_dict['hidden_dim'])
        self.mesh = MeshConfig(config_dict['mesh'])
        self.encoder = CoderConfig(config_dict['encoder'])
        self.processor = ProcessorConfig(config_dict['processor'])
        self.decoder = CoderConfig(config_dict['decoder'])
        self.embeddings = EmbeddingsConfig(config_dict['embeddings'])


model_config = ModelConfig(yaml.safe_load(open("gnn_model/configs/model_config.yaml")))
