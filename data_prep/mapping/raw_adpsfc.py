#!/usr/bin/env python3

import os
import numpy as np
import re
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


DumpKey = 'dump_map_path'
PrepbufrKey = 'prepbufr_map_path'
DumpMapPath = map_path('raw_adpsfc_dump.yaml')
PrepbufrMapPath = map_path('raw_adpsfc_prepbufr.yaml')

class RawAdpsfcBuilder(ObsBuilder):
    def __init__(self):
        super().__init__({DumpKey: DumpMapPath,
                          PrepbufrKey: PrepbufrMapPath}, log_name=os.path.basename(__file__))

    # Override
    def make_obs(self, comm, input_dict) -> bufr.DataContainer:
        dump_container = bufr.Parser(input_dict[DumpKey], self.map_dict[DumpKey]).parse(comm)
        prepbufr_container = bufr.Parser(input_dict[PrepbufrKey], self.map_dict[PrepbufrKey]).parse(comm)

        container = bufr.DataContainer()

        return container

    def _make_description(self):
        description = super()._make_description()

        description.add_variables([
            {
                'name': "time",
                'source': 'timestamp',
                'longName': "Datetime",
                'units': "seconds since 1970-01-01T00:00:00Z"
            }
        ])

        return description

    def _get_reference_time(self, input_path) -> np.datetime64:
        path_components = Path(input_path).parts
        m = re.match(r'\w+\.(?P<year>\d{4})(?P<month>\d{2})(?P<day>\d{2})', path_components[-4])

        if not m.groups():
            raise Exception("Error: Path string did not match the expected pattern.")

        return np.datetime64(datetime(year=int(m.group('year')),
                                      month=int(m.group('month')),
                                      day=int(m.group('day')),
                                      hour=int(path_components[-3])))

    def _add_timestamp(self, container: bufr.DataContainer, reference_time: np.datetime64) -> np.array:
        cycle_times = np.array([3600 * t for t in container.get('obsTimeMinusCycleTime')]).astype('timedelta64[s]')
        time = (reference_time + cycle_times).astype('datetime64[s]').astype('int64')
        container.add('timestamp', time, ['*'])



add_main_functions(RawAdpsfcBuilder)
