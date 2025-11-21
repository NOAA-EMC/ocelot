# read the tank yaml configuration
import os
import yaml
import settings


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
    def __init__(self, config, type_name):
        self.config = config
        self.type = type_name

    @property
    def name(self):
        return self.config['name']

    @property
    def mapping(self):
        return self.config['mapping']

    @property
    def num_tasks(self):
        return self.config['num_tasks']

    @property
    def batch_days(self):
        return self.config['batch_days']

    @property
    def memory(self):
        if 'memory' not in self.config:
            return None
        return self.config['memory']

    @property
    def operations(self):
        if 'operations' not in self.config:
            return []

        return [OperationConfig(op) for op in self.config['operations']]


class TankConfig(DataTypeConfig):
    def __init__(self, config):
        super().__init__(config, 'tank')

    @property
    def paths(self) -> str | dict:
        return self.config['paths']


class PcaConfig(DataTypeConfig):
    def __init__(self, config):
        super().__init__(config, 'pca')

    @property
    def directory(self):
        return self.config['directory']

    @property
    def filename_regex(self):
        return self.config['filename_regex']


class Config:
    def __init__(self, yaml_path=''):
        if not yaml_path:
            yaml_path = settings.BUFR_TANK_YAML

        self.config = yaml.load(open(yaml_path), Loader=yaml.Loader)['data types']
        self.data_types = self._get_data_types()

    def get_data_type_names(self):
        return [data_type.name for data_type in self.data_types]

    def get_data_type(self, name) -> DataTypeConfig:
        for data_type in self.data_types:
            if data_type.name == name:
                return data_type
        assert False, f"Data type {name} not found in config"

    def get_map_path(self, name):
        type_config = self.get_data_type(name)
        map_path = os.path.join(settings.MAPPING_FILE_DIR, type_config.mapping)

        return map_path

    def _get_data_types(self):
        data_types = []
        for data_type in self.config:
            if data_type['type'] == 'tank':
                data_types.append(TankConfig(data_type))
            elif data_type['type'] == 'pca':
                data_types.append(PcaConfig(data_type))
            else:
                assert False, f"Unknown data type {data_type['type']} in config"

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
