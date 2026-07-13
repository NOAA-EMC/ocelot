from copy import deepcopy
from datetime import datetime, date
from typing import Any


_MISSING = object()


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
        fields = {}
        for base in bases:
            fields.update(getattr(base, '_fields', {}))
        fields.update({k: v for k, v in attrs.items() if isinstance(v, ConfigItem)})
        attrs['_fields'] = fields
        return super().__new__(cls, name, bases, attrs)


class ConfigBase(ConfigItem, metaclass=ConfigMeta):

    def __init__(self):
        self._initialize_fields()

    def _initialize_fields(self) -> None:
        fields = deepcopy(type(self)._fields)
        self.__dict__['_fields'] = fields
        self.__dict__.update(fields)

    def load(self, config_dict: dict) -> None:
        if '_fields' not in self.__dict__:
            self._initialize_fields()

        if not isinstance(config_dict, dict):
            raise ValueError(f"Expected config dictionary but got {type(config_dict).__name__}")

        unknown_fields = config_dict.keys() - self._fields.keys()
        if unknown_fields:
            names = ', '.join(sorted(unknown_fields))
            raise ValueError(f"Unknown field(s) for {type(self).__name__}: {names}")

        for field_name in self._fields:
            field = self._fields[field_name]

            if field_name not in config_dict:
                if isinstance(field, Optional):
                    field.load()
                    continue     
                raise ValueError(f"Missing required field '{field_name}' in config")
            if isinstance(field, ConfigBase):
                field.load(config_dict[field_name])
            elif isinstance(field, ConfigField):
                field.load(config_dict[field_name])

    def to_dict(self) -> dict[str, Any]:
        config_dict = {}
        for field_name, field in self._fields.items():
            value = getattr(self, field_name)
            if isinstance(value, ConfigBase):
                value_dict = value.to_dict()
                choice_field = field.inner_type if isinstance(field, Optional) else field
                if isinstance(choice_field, Choices):
                    value_dict = {'type': value.type, **value_dict}
                value = value_dict
            config_dict[field_name] = value
        return config_dict

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
        
            choice_type = value['type']
            self.choice = deepcopy(self.choices[choice_type])
            self.choice.load({key: item for key, item in value.items() if key != 'type'})
            self.choice.type = choice_type
    
        
class Optional(ConfigField):
    type = None

    def __init__(self, inner_type, default=None):
        self.inner_type = inner_type
        self.default = default
        self._has_value = False

    @property
    def value(self):
        if not self._has_value:
            return None
        if isinstance(self.inner_type, ConfigField):
            return self.inner_type.value
        return self.inner_type
    
    def load(self, value=_MISSING):
        if value is _MISSING:
            value = self.default
        if value is None:
            self._has_value = False
            return
        self.inner_type.load(value)
        self._has_value = True


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


class ListField(ConfigField):
    def __init__(self, inner_type: ConfigField):
        super().__init__()
        self.inner_type = inner_type

    def load(self, value):
        if not isinstance(value, list):
            raise ValueError(f"Expected value of type list but got {type(value).__name__}")

        items = []
        for item in value:
            field = deepcopy(self.inner_type)
            field.load(item)
            items.append(field.value)
        self.value = items


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

