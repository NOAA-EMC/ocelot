#!/usr/bin/env python3

import os
import numpy as np
import re
from datetime import datetime
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions


script_dir = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.path.join(script_dir, 'bufr_surface_obs.yaml')

OBS_TYPES = np.array([180, 181, 183, 187, 120])


class PressureObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(MAP_PATH, log_name=os.path.basename(__file__))

    # Override
    def make_obs(self, comm, input_path) -> bufr.DataContainer:
        container = super().make_obs(comm, input_path)

        # Apply Masks

        # Filter according to thw obs type
        obs_type = container.get('observationType')
        container.apply_mask(np.isin(obs_type, OBS_TYPES))

        # Mask out missing time stamps
        # Note, in numpy masked arrays "mask == True" means to mask out. So we must invert the mask.
        container.apply_mask(~container.get('obsTimeMinusCycleTime').mask)

        # Apply Quality Masks
        quality_mask = container.get('airTemperatureQuality') <= 3 & \
                       container.get('specificHumidityQuality') <= 3 & \
                       container.get('windQuality') <= 3 & \
                       container.get('airPressureQuality') <= 3 & \
                       container.get('heightQuality') <= 3 & \
                       container.get('seaTemperatureQuality') <= 3

        container.apply_mask(quality_mask)

        # Add timestamps
        reference_time = self._get_reference_time(input_path)
        self._add_timestamp(container, reference_time)

        # Convert stationIdentification into integer field
        stationIdentification = container.get('stationIdentification')
        encoder = LabelEncoder()
        stationIdentification = encoder.fit_transform(stationIdentification)
        container.replace('stationIdentification', stationIdentification)

        # Add global attribute for stationIdentification labels
        self.description.add_global('stationIdentificationLabels', list(encoder.classes_))

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


# Add main functions create_obs_file and create_obs_group
add_main_functions(PressureObsBuilder)
