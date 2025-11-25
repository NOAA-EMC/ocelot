import os
import sys
import importlib.util
import re
from datetime import datetime, timedelta
import bufr

sys.path.insert(0, os.path.realpath('.'))
sys.path.insert(0, os.path.realpath('..'))

import config  # noqa: E402
import settings  # noqa: E402


class Parameters:
    def __init__(self):
        self.start_time = None
        self.stop_time = None


class Runner(object):
    """Base class for all data readers."""

    def __init__(self, data_type, cfg):
        self.config = cfg

        if data_type not in self.config.get_data_type_names():
            raise ValueError(f"Data type {data_type} not found in config")

        self.type_config = self.config.get_data_type(data_type)
        self.map_path = os.path.join(settings.MAPPING_FILE_DIR, self.type_config.mapping)

        # Determine if we're using a python script or yaml mapping
        if os.path.splitext(self.map_path)[1] == ".py":
            self.script = self._load_script()
            self.obs_builder = self.script.make_obs_builder()
        else:
            self.script = None
            self.obs_builder = None

    def get_encoder_description(self) -> bufr.encoders.Description:
        if self.obs_builder:
            return self.obs_builder.description
        return bufr.encoders.Description(self.map_path)

    def _make_obs(self, comm, input_path: str) -> bufr.DataContainer:
        if self.obs_builder:
            return self.obs_builder.make_obs(comm, input_path)
        return bufr.Parser(input_path, self.map_path).parse(comm)

    def _load_script(self):
        module_name = os.path.splitext(os.path.basename(self.map_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, self.map_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, 'make_obs_builder'):
            raise ValueError(f"Script {self.map_path} must define make_obs_builder.")

        return module


class TankRunner(Runner):
    """Runner for traditional BUFR tank files."""

    def run(self, comm, parameters: Parameters) -> bufr.DataContainer:
        combined_container = bufr.DataContainer()

        if isinstance(self.type_config.paths, list):
            for day_str in self._day_strs(parameters.start_time, parameters.stop_time):
                for path in self.type_config.paths:
                    input_path = os.path.join(settings.TANK_PATH, day_str, path)

                    if not os.path.exists(input_path):
                        print(f"Input path {input_path} does not exist! Skipping it.")
                        continue

                    container = self._make_obs(comm, input_path)
                    container.gather(comm)

                    if comm.rank() == 0:
                        combined_container.append(container)

        elif isinstance(self.type_config.paths, dict):
            for day_str in self._day_strs(parameters.start_time, parameters.stop_time):
                for path_idx in range(len(list(self.type_config.paths.values())[0])):
                    input_dict = {}
                    for key in self.type_config.paths.keys():
                        rel_path = self.type_config.paths[key][path_idx]
                        input_path = os.path.join(settings.TANK_PATH, day_str, rel_path)

                        if not os.path.exists(input_path):
                            print(f"Input path {input_path} does not exist! Skipping it.")
                            continue

                        input_dict[key] = input_path

                    container = self._make_obs(comm, input_dict)
                    container.gather(comm)

                    if comm.rank() == 0:
                        combined_container.append(container)

        return combined_container

    def _day_strs(self, start: datetime, end: datetime) -> list:
        day = start
        days = []
        while day <= end:
            days.append(day.strftime(settings.DATETIME_DIR_FORMAT))
            day += timedelta(days=1)

        return days


class PcaRunner(Runner):
    """Runner for PCA BUFR files with time stamped names."""

    def __init__(self, data_type, cfg):
        super().__init__(data_type, cfg)
        self.regex = re.compile(self.type_config.filename_regex)

    def run(self, comm, parameters: Parameters) -> bufr.DataContainer:
        combined_container = bufr.DataContainer()
        directory = self.type_config.directory

        for fname in os.listdir(directory):
            match = self.regex.match(fname)
            if not match:
                continue

            start_str = match.group("start_time")
            end_str = match.group("end_time")
            file_start = datetime.strptime(start_str, "%Y%m%d%H%M%S")
            file_end = datetime.strptime(end_str, "%Y%m%d%H%M%S")

            if file_end < parameters.start_time or file_start > parameters.stop_time:
                continue

            input_path = os.path.join(directory, fname)
            container = self._make_obs(comm, input_path)
            container.gather(comm)

            if comm.rank() == 0:
                combined_container.append(container)

        return combined_container


def run(comm, data_type, parameters: Parameters, cfg=config.Config()) -> (bufr.encoders.Description, bufr.DataContainer):
    type_cfg = cfg.get_data_type(data_type)
    if type_cfg.type == 'tank':
        runner = TankRunner(data_type, cfg)
    elif type_cfg.type == 'pca':
        runner = PcaRunner(data_type, cfg)
    else:
        raise ValueError(f"Unknown data type {type_cfg.type}")

    return (runner.get_encoder_description(), runner.run(comm, parameters))
