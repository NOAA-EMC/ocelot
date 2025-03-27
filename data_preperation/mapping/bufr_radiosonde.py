#!/usr/bin/env python3

import os
import numpy as np

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions

script_dir = os.path.dirname(os.path.abspath(__file__))
MAP_PATH = os.path.join(script_dir, 'bufr_radiosonde.yaml')

class RadiosondeObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(MAP_PATH, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path):
        container = super().make_obs(comm, input_path)

        # Replace virtual temperature with computed air temperature values
        temp = container.get('airTemperature')
        temp_event_code = container.get('temperatureEventCode')
        specific_humidity = container.get('specificHumidity')

        virt_temp_mask = temp_event_code == 8

        mixing_ratio = specific_humidity / (1 - specific_humidity)
        temp[virt_temp_mask] = temp[virt_temp_mask] / (1 + 0.61 * mixing_ratio[virt_temp_mask])

        container.replace('airTemperature', temp)

        return container

add_main_functions(RadiosondeObsBuilder)
