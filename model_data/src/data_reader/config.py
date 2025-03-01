# read the tank yaml configuration
import os
import yaml
from .. import settings


class OperationConfig:
    def __init__(self, config):
        self.config = config

    @property
    def type(self):
        return self.config['type']

    @property
    def parameters(self):
        return self.config['parameters']


class DataTypeConfig:
    def __init__(self, config):
        self.config = config

    @property
    def name(self):
        return self.config['name']

    @property
    def mapping(self):
        return self.config['mapping']

    @property
    def paths(self):
        return self.config['paths']

    @property
    def operations(self):
        if not 'operations' in self.config:
            return []

        return [OperationConfig(op) for op in self.config['operations']]


class Config:
    def __init__(self, yaml_path=''):
        if not yaml_path:
            yaml_path = settings.BUFR_TANK_YAML

        self.config = yaml.load(open(yaml_path), Loader=yaml.Loader)['data types']
        self.data_types = self._get_data_types()

    def get_data_type_names(self):
        return [data_type.name for data_type in self.data_types]

    def get_data_type(self, name):
        for data_type in self.data_types:
            if data_type.name == name:
                return data_type
        return None

    def get_map_path(self, name):
        type_config = self.get_data_type(name)
        map_path = os.path.join(settings.MAPPING_FILE_DIR, type_config.mapping)

        return map_path

    def _get_data_types(self):
        data_types = []
        for data_type in self.config:
            data_types.append(DataTypeConfig(data_type))
        return data_types

    def __repr__(self):
        return f"TankConfig(data_types={self.data_types})"


if __name__ == '__main__':
    tank_config = Config('../../test/testinput/local_emc_tank.yaml')

    for data_type in tank_config.data_types:
        print(data_type.name)
        print(data_type.mapping)
        print(data_type.operations)
        print(data_type.paths)
        print()
