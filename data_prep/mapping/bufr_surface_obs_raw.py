#!/usr/bin/env python3

import os
import numpy as np
import re
from datetime import datetime
from pathlib import Path

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


PrepbufrKey = 'prepbufr'
AdpsfcKey = 'adpsfc'
SfcshpKey = 'sfcshp'

PrepbufrMapPath = map_path('bufr_surface_obs_prepbufr.yaml')
AdpsfcMapPath = map_path('bufr_surface_obs_adpsfc.yaml')
SfcshpMapPath = map_path('bufr_surface_obs_sfcshp.yaml')

class RawAdpsfcBuilder(ObsBuilder):
    def __init__(self):
        super().__init__({PrepbufrKey: PrepbufrMapPath,
                          AdpsfcKey: AdpsfcMapPath,
                          SfcshpKey: SfcshpMapPath}, log_name=os.path.basename(__file__))

    # Override
    def make_obs(self, comm, input_dict) -> bufr.DataContainer:
        prepbufr_container = bufr.Parser(input_dict[PrepbufrKey], self.map_dict[PrepbufrKey]).parse(comm)
        adpsfc_container = bufr.Parser(input_dict[AdpsfcKey], self.map_dict[AdpsfcKey]).parse(comm)
        sfcshp_container = bufr.Parser(input_dict[SfcshpKey], self.map_dict[SfcshpKey]).parse(comm)

        # Mask out missing time stamps
        # Note, in numpy masked arrays "mask == True" means to mask out. So we must invert the mask.
        prepbufr_container.apply_mask(~prepbufr_container.get('obsTimeMinusCycleTime').mask)

        # Add timestamps to the prepbufr container
        reference_time = self._get_reference_time(input_dict[PrepbufrKey])
        self._add_timestamp(prepbufr_container, reference_time)

        # Create output container
        container = bufr.DataContainer()

        # Combine the ADPSFC and SFCSHP containers
        container.append(adpsfc_container)
        container.append(sfcshp_container)

        # Add the prepbufr quality flag fields to the combined container
        # Use the timestamp, latitude and longitude to match the observations
        prepbufr_time = prepbufr_container.get('timestamp').filled()
        prepbufr_lat = prepbufr_container.get('latitude').filled()
        prepbufr_lon = prepbufr_container.get('longitude').filled()

        container_time = container.get('timestamp').filled()
        container_lat = container.get('latitude').filled()
        container_lon = container.get('longitude').filled()

        # Make hash table for fast lookup
        prepbufr_dict = {}
        for i, (t, lat, lon) in enumerate(zip(prepbufr_time, prepbufr_lat, prepbufr_lon)):
            key = (t, np.round(lat, 2), np.round(lon, 2))
            prepbufr_dict[key] = i

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
        for var in ['airTemperatureQuality',
                    'specificHumidityQuality',
                    'windQuality',
                    'airPressureQuality',
                    'dewPointTemperatureQuality',
                    'heightQuality']:
            quality_flags = prepbufr_container.get(var)[indices]
            container.add(var, quality_flags, ['*'])

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
