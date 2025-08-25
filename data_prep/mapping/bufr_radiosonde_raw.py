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
        prepbufr_container.apply_mask(~prepbufr_container.get('driftTime').mask)

        # Add timestamps to the prepbufr container
        reference_time = self._get_reference_time(input_dict[PrepbufrKey])
        self._add_timestamp(prepbufr_container, reference_time)

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

