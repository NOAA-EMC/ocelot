
import yaml

class ConfigItem:
    pass

class ConfigField(ConfigItem):
    def __init__(self):
        self.value = None

    def _validate(self, value):
        raise NotImplementedError("Must implement _validate method")

    def __call__(self):
        return self.value

class Choices(ConfigField):
    def __init__(self, choices: list[str]):
        self.choices = choices
        super().__init__()

    def _validate(self):
        if self.value not in self.choices:
            raise ValueError(f"Value '{self.value}' not in allowed choices: {self.choices}")
        
class Optional(ConfigField):
    def __init__(self, inner_type, default=None):
        self.inner_type = inner_type
        self.default = default
        super().__init__(default)

    def _validate(self):
        if self.value is not None:
            self.inner_type._validate(self.value)

class IntField(ConfigField):
    def _validate(self):
        if not isinstance(self.value, int):
            raise ValueError(f"Expected int but got {type(self.value)}")
        
class FloatField(ConfigField):
    def _validate(self):
        if not isinstance(self.value, float):
            raise ValueError(f"Expected float but got {type(self.value)}")

class StrField(ConfigField):
    def _validate(self):
        if not isinstance(self.value, str):
            raise ValueError(f"Expected str but got {type(self.value)}")

# meta class for all config classes, which will validate the fields and provide a way to access the fields
class ConfigMeta(type):
    def __new__(cls, name, bases, attrs):
        fields = {k: v for k, v in attrs.items() if isinstance(v, ConfigItem)}
        attrs['_fields'] = fields
        return super().__new__(cls, name, bases, attrs)


class ConfigBase(ConfigItem, metaclass=ConfigMeta):

    def load(self, config_dict: dict) -> None:
        for field_name in self._fields:
            if field_name not in config_dict:
                raise ValueError(f"Missing required field '{field_name}' in config")
            field = self._fields[field_name]
            if isinstance(field, ConfigBase):
                field.load(config_dict[field_name])
            elif isinstance(field, ConfigField):
                field.value = config_dict[field_name]
                self.__dict__[field_name] = field
                field._validate()

    # return the value of the field
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if isinstance(attr, ConfigField):
            return attr()
        return attr


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
