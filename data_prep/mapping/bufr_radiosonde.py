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
MAP_PATH = os.path.join(script_dir, 'bufr_radiosonde.yaml')


class RadiosondeObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(MAP_PATH, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path):
        container = super().make_obs(comm, input_path)

        # Apply Masks

        # Mask out bad time values
        # Note, in numpy masked arrays "mask == True" means to mask out. So we must invert the mask.
        container.apply_mask(~container.get('obsTimeMinusCycleTime').mask)

        self._apply_quality_flag(container, 'airTemperature', 'airTemperatureQuality')
        self._apply_quality_flag(container, 'specificHumidity', 'specificHumidityQuality')
        self._apply_quality_flag(container, 'northwardWind', 'windQuality')
        self._apply_quality_flag(container, 'eastwardWind',  'windQuality')
        self._apply_quality_flag(container, 'airPressure', 'airPressureQuality')
        self._apply_quality_flag(container, 'height', 'heightQuality')

        # Add timestamps
        reference_time = self._get_reference_time(input_path)
        self._add_timestamp(container, reference_time)

        # Replace virtual temperature with computed air temperature values
        temp = container.get('airTemperature')
        temp_event_code = container.get('temperatureEventCode')
        specific_humidity = container.get('specificHumidity')

        virt_temp_mask = temp_event_code == 8
        mixing_ratio = specific_humidity / (1 - specific_humidity)
        temp[virt_temp_mask] = temp[virt_temp_mask] / (1 + 0.61 * mixing_ratio[virt_temp_mask])
        container.replace('airTemperature', temp)

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

    def _apply_quality_flag(self, container, target_field_name, quality_field_name):
        data = container.get(target_field_name)
        data.mask[container.get(quality_field_name) > 3] = True  # True means mask out
        container.replace(target_field_name, data)


add_main_functions(RadiosondeObsBuilder)
