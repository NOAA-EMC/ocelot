#!/usr/bin/env python3

import os

from bufr.obs_builder import ObsBuilder, add_main_functions

script_dir = os.path.dirname(os.path.abspath(__file__))
MAPPING_PATH = os.path.join(script_dir, 'bufr_atms.yaml')


class AtmsObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path):
        container = super().make_obs(comm, input_path)

        # Mask out values with non-zero quality flags
        quality_flags = container.get('qualityFlags')

        brightnessTemperature = container.get('brightnessTemperature')
        brightnessTemperature[quality_flags != 0].mask = True
        container.replace('brightnessTemperature', brightnessTemperature)

        return container


add_main_functions(AtmsObsBuilder)
