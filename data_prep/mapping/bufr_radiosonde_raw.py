#!/usr/bin/env python3

import os
import numpy as np
import re
from datetime import datetime
from pathlib import Path

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


PrepbufrKey = 'prepbufr'
DumpKey = 'dump'

PrepbufrMapPath = map_path('bufr_radiosonde_prepbufr.yaml')
DumpMapPath = map_path('bufr_radiosonde_dump.yaml')

class RawRadiosondeBuilder(ObsBuilder):
    def __init__(self):
        super().__init__({PrepbufrKey: PrepbufrMapPath,
                          DumpKey: DumpMapPath}, log_name=os.path.basename(__file__))

    # Override
    def make_obs(self, comm, input_dict) -> bufr.DataContainer:
        prepbufr_container = bufr.Parser(input_dict[PrepbufrKey], self.map_dict[PrepbufrKey]).parse(comm)
        container = bufr.Parser(input_dict[DumpKey], self.map_dict[DumpKey]).parse(comm)

        # Mask out missing time stamps
        # Note, in numpy masked arrays "mask == True" means to mask out. So we must invert the mask.
        prepbufr_container.apply_mask(~prepbufr_container.get('launchCycleTime').mask)

        # Add timestamps to the prepbufr container
        reference_time = self._get_reference_time(input_dict[PrepbufrKey])
        self._add_timestamp('launchCycleTime',
                            'launchTime',
                            prepbufr_container,
                            reference_time)

        self._add_timestamp('driftCycleTime',
                            'driftTime',
                            prepbufr_container,
                            reference_time)

        # Add the prepbufr quality flag fields to the combined container
        # Use the timestamp, latitude and longitude to match the observations
        prepbufr_time = prepbufr_container.get('launchTime').filled()
        prepbufr_lat = prepbufr_container.get('launchLatitude').filled()
        prepbufr_lon = prepbufr_container.get('launchLongitude').filled()

        # Make hash table for fast lookup
        prepbufr_dict = {}
        for i, (t, lat, lon) in enumerate(zip(prepbufr_time, prepbufr_lat, prepbufr_lon)):
            key = (t, np.round(lat, 2), np.round(lon, 2))
            prepbufr_dict[key] = i

        container_time = container.get('timestamp').filled()
        container_lat = container.get('latitude').filled()
        container_lon = container.get('longitude').filled()

        # Use hash table to find matching indices in combined container
        indices = [-1] * len(container_time)
        for i, (t, lat, lon) in enumerate(zip(container_time, container_lat, container_lon)):
            key = (t, np.round(lat, 2), np.round(lon, 2))
            if key in prepbufr_dict:
                indices[i] = prepbufr_dict[key]

        indices = np.array(indices)
        valid_mask = indices != -1
        indices = indices[valid_mask]
        container.apply_mask(~valid_mask)

        # Add the quality flags to the container
        for var in ['driftTime',
                    'driftLatitude',
                    'driftLongitude',
                    'height',
                    'airTemperatureQuality',
                    'specificHumidityQuality',
                    'dewPointTemperatureQuality',
                    'windQuality',
                    'airPressureQuality',
                    'heightQuality']:
            quality_flags = prepbufr_container.get(var)[indices]
            container.add(var, quality_flags, ['*'])

        return container

    def _make_description(self):
        description = bufr.encoders.Description(self.map_dict[DumpKey])

        # Add the quality flag variables
        description.add_variables([
            {
                'name': "time",
                'source': 'driftTime',
                'longName': "Datetime",
                'units': "seconds since 1970-01-01T00:00:00Z"
            },
            {
                'name': "latitude",
                'source': 'driftLatitude',
                'longName': "Latitude",
                'units': "degree_north"
            },
            {
                'name': "longitude",
                'source': 'driftLongitude',
                'longName': "Longitude",
                'units': "degree_east"
            },
            {
                'name': "height",
                'source': 'height',
                'longName': "Height",
                'units': "meters"
            },
            {
                'name': "airTemperatureQuality",
                'source': 'airTemperatureQuality',
                'longName': "Air Temperature Quality Marker",
                'units': "quality_marker"
            },
            {
                'name': "specificHumidityQuality",
                'source': 'specificHumidityQuality',
                'longName': "Specific Humidity Quality Marker",
                'units': "quality_marker"
            },
            {
                'name': "dewPointTemperatureQuality",
                'source': 'dewPointTemperatureQuality',
                'longName': "Dew Point Temperature Quality Marker",
                'units': "quality_marker"
            },
            {
                'name': "windQuality",
                'source': 'windQuality',
                'longName': "Wind Quality Marker",
                'units': "quality_marker"
            },
            {
                'name': "airPressureQuality",
                'source': 'airPressureQuality',
                'longName': "Air Pressure Quality Marker",
                'units': "quality_marker"
            },
            {
                'name': "heightQuality",
                'source': 'heightQuality',
                'longName': "Height Quality Marker",
                'units': "quality_marker"
            }
        ])

        description.add_dimension('event', ['*', '*/EVENT'])

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

add_main_functions(RawRadiosondeBuilder)

