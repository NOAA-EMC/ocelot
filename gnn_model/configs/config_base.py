from datetime import datetime, date
from typing import Any


class ConfigItem:
    pass

class ConfigField(ConfigItem):
    def __init__(self):
        self.value = None

    def load(self, value: Any) -> None:
        assert False, f"load method not implemented for field type {self.__class__.__name__}"

    def __call__(self):
        return self.value


# meta class for all config classes, which will validate the fields and provide a way to access the fields
class ConfigMeta(type):
    def __new__(cls, name, bases, attrs):
        fields = {k: v for k, v in attrs.items() if isinstance(v, ConfigItem)}
        attrs['_fields'] = fields
        return super().__new__(cls, name, bases, attrs)


class ConfigBase(ConfigItem, metaclass=ConfigMeta):

    def load(self, config_dict: dict) -> None:
        for field_name in self._fields:
            field = self._fields[field_name]

            if field_name not in config_dict:
                if isinstance(field, Optional):
                    continue            
                raise ValueError(f"Missing required field '{field_name}' in config")
            if isinstance(field, ConfigBase):
                field.load(config_dict[field_name])
            elif isinstance(field, ConfigField):
                field.load(config_dict[field_name])
                self.__dict__[field_name] = field

    # return the value of the field
    def __getattribute__(self, name):
        attr = super().__getattribute__(name)
        if isinstance(attr, ConfigField):
            return attr()
        return attr
    

class Choices(ConfigField):
    def __init__(self, choices: dict[str, ConfigBase] | list[str]):
        self.choices = choices
        self.choice = None

    @property
    def value(self):
        return self.choice

    def load(self, value: dict | str) -> None:
        if isinstance(self.choices, list):
            if value not in self.choices:
                raise ValueError(f"Value '{value}' not in allowed choices: {self.choices}")
            self.choice = value

        elif isinstance(self.choices, dict):
            if not isinstance(value, dict):
                raise ValueError(f"Expected value of type dict but got {type(value)}")
            
            if 'type' not in value:
                raise ValueError(f"Missing 'type' key in value: {value}. Possible types are: {list(self.choices.keys())}")

            if value['type'] not in self.choices.keys():
                raise ValueError(f"Value '{value['type']}' not in allowed choices: {list(self.choices.keys())}")
        
            self.choice = self.choices[value['type']]
            self.choice.load(value)
            self.choice.type = value['type']
    
        
class Optional(ConfigField):
    type = None

    def __init__(self, inner_type, default=None):
        self.inner_type = inner_type
        self.default = default

    @property
    def value(self):
        return self.inner_type.value
    
    def load(self, value=None):    
        if value is not None:
            self.inner_type.load(value)
        elif self.default is not None:
            self.inner_type.load(self.default)


class IntField(ConfigField):
    def load(self, value):
        self.value = int(value)

        
class BoolField(ConfigField):
    def load(self, value):
        self.value = bool(value)

        
class FloatField(ConfigField):
    def load(self, value):
        self.value = float(value)


class StrField(ConfigField):
    def load(self, value):
        self.value = str(value)


class DatetimeField(ConfigField):
    def load(self, value):
        if isinstance(value, str):
            self.value = datetime.fromisoformat(value)
        elif isinstance(value, datetime):
            self.value = value
        elif isinstance(value, date):
            self.value = datetime(value.year, value.month, value.day)
        else:
            raise ValueError(f"Expected value of type str or datetime but got {type(value)}")

