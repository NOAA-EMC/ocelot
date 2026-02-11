#!/usr/bin/env python3

import os
import numpy as np
import re
from datetime import datetime
from pathlib import Path

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


PrepbufrMapPath = map_path('bufr_aircraft_prepbufr.yaml')


class AircraftBuilder(ObsBuilder):

    def __init__(self):
        super().__init__(PrepbufrMapPath, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path) -> bufr.DataContainer:
        if not input_path:
            return bufr.DataContainer()

        prep_container = bufr.Parser(input_path, PrepbufrMapPath).parse(comm)
        prep_container.apply_mask(~prep_container.get('driftCycleTime').mask)
        prep_container.apply_mask(~prep_container.get('driftLatitude').mask)

        reference_time = self._get_reference_time(input_path)

        self._add_timestamp('driftCycleTime',
                            'timestamp',
                            prep_container,
                            reference_time)

        return prep_container

    def _make_description(self):
        description = bufr.encoders.Description(PrepbufrMapPath)

        # Add the quality flag variables
        description.add_variables([
            {
                'name': "time",
                'source': 'timestamp',
                'longName': "Datetime",
                'units': "seconds since 1970-01-01T00:00:00Z"
            }])

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

    def _add_timestamp(self,
                       input_name: str,
                       output_name: str,
                       container: bufr.DataContainer,
                       reference_time: np.datetime64) -> None:
        cycle_times = np.array([3600 * t for t in container.get(input_name)]).astype('timedelta64[s]')
        time = (reference_time + cycle_times).astype('datetime64[s]').astype('int64')
        container.add(output_name, time, ['*'])


# Add main functions create_obs_file and create_obs_group
add_main_functions(AircraftBuilder)
