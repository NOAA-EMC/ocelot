import os
import sys
sys.path.insert(0, os.path.realpath('.'))
sys.path.insert(0, os.path.realpath('..'))

import sys
import importlib.util
from datetime import datetime, timedelta
import argparse

import bufr

from . import config
import settings

class Parameters:
    def __init__(self):
        self.start_time = None
        self.stop_time = None

class Runner(object):
    def __init__(self, data_type, config):
        self.tank_config = config

        if data_type not in self.tank_config.get_data_type_names():
            raise ValueError(f"Data type {data_type} not found in tank")

        self.type_config = self.tank_config .get_data_type(data_type)
        self.map_path = os.path.join(settings.MAPPING_FILE_DIR, self.type_config.mapping)

    def run(self, comm, parameters:Parameters) -> bufr.DataContainer:
        # Parse the relevant BUFR files
        combined_container = bufr.DataContainer()
        for path in self.type_config.paths:
            for day_str in self._day_strs(parameters.start_time, parameters.stop_time):
                input_path = os.path.join(settings.TANK_PATH, day_str, path)

                if not os.path.exists(input_path):
                    continue

                container = self._make_obs(comm, input_path)

                # if parameters.category is not None:
                #     container = container.get_sub_container(parameters.category)

                container.gather(comm)
                combined_container.append(container)

        return combined_container

    def get_encoder_description(self) -> bufr.encoders.Description:
        raise NotImplementedError

    def _make_obs(self, comm, input_path:str) -> bufr.DataContainer:
        raise NotImplementedError

    def _day_strs(self, start: datetime, end: datetime) -> list[str]:
        '''
        Loop through each day in the window and return a list of formatted strings
        :return:
        '''

        day = start
        days = []
        while day <= end:
            days.append(day.strftime(settings.DATETIME_DIR_FORMAT))
            day += timedelta(days=1)

        return days


class YamlRunner(Runner):
    def __init__(self, data_type, config):
        super().__init__(data_type, config)

    def get_encoder_description(self) -> bufr.encoders.Description:
        return bufr.encoders.Description(self.map_path)

    def _make_obs(self, comm, input_path:str) -> bufr.DataContainer:
        return bufr.Parser(input_path, self.map_path).parse(comm)


class ScriptRunner(Runner):
    def __init__(self,data_type:str, config=config.Config()):
        super().__init__(data_type, config)
        self.script = self._load_script()
        self.obs_builder = self.script.make_obs_builder()

    def get_encoder_description(self) -> bufr.encoders.Description:
        return self.obs_builder.make_description()

    def _make_obs(self, comm, input_path:str) -> bufr.DataContainer:
        return self.obs_builder.make_obs(comm, input_path)

    def _load_script(self):
        module_name = os.path.splitext(os.path.basename(self.map_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, self.map_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, 'make_obs') or not hasattr(module, 'make_encoder_description'):
            raise ValueError(f"Script {self.map_path} must have a make_obs and make_encoder_description.")

        return module


def run(comm, data_type, parameters:Parameters, tank_config = config.Config()) -> (bufr.encoders.Description, bufr.DataContainer):
    if data_type not in tank_config.get_data_type_names():
        raise ValueError(f"Data type {data_type} not found in tank")

    if  os.path.splitext(tank_config.get_data_type(data_type).mapping)[1] == '.py':
        runner = ScriptRunner(data_type, tank_config)
    else:
        runner = YamlRunner(data_type, tank_config)

    return (runner.get_encoder_description(), runner.run(comm, parameters))


if __name__ == '__main__':
    sys.path.insert(0, os.path.realpath('..'))
    from zarr_encoder import Encoder as ZarrEncoder

    bufr.mpi.App(sys.argv)
    comm = bufr.mpi.Comm("world")

    parameters = Parameters()
    parameters.start_time = datetime(2020, 10, 12)
    parameters.stop_time = datetime(2020, 10, 13)

    description, combined_container = run(comm, 'atms', parameters)

    if combined_container is None:
        raise ValueError("No data found")

    if comm.rank() == 0:
        ZarrEncoder(description).encode(combined_container, "output_atms.zarr", append=True)
        print("Output written to output_atms.zarr")
