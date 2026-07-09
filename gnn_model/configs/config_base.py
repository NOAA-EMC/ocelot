class ConfigItem:
    pass

class ConfigField(ConfigItem):

    @property
    def value(self):
        return self._value
    
    @value.setter
    def value(self, value):
        self._value = self._get_type()(value) if value is not None else None

    def _validate(self):
        if not isinstance(self._value, self._get_type()):
            raise ValueError(f"Expected value of type {self._get_type()} but got {type(self._value)}")

    def __call__(self):
        return self.value

    def _get_type(self):
        return None

class Choices(ConfigField):
    def __init__(self, choices: list[str]):
        self.choices = choices
        super().__init__()

    def _validate(self):
        if self.value not in self.choices:
            raise ValueError(f"Value '{self.value}' not in allowed choices: {self.choices}")
        
    def _get_type(self):
        return str
        
class Optional(ConfigField):
    type = None

    def __init__(self, inner_type, default=None):
        self.inner_type = inner_type
        self.value = default
        super().__init__()

    @property
    def value(self):
        return self.inner_type.value

    @value.setter
    def value(self, value):
        self.inner_type.value = value

    def _validate(self):
        if self.value is not None:
            self.inner_type._validate()

class IntField(ConfigField):
    def _get_type(self):
        return int
        
class BoolField(ConfigField):
    def _get_type(self):
        return bool
        
class FloatField(ConfigField):
    def _get_type(self):
        return float

class StrField(ConfigField):
    def _get_type(self):
        return str

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
                if isinstance(self._fields[field_name], Optional):
                    continue            
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
    