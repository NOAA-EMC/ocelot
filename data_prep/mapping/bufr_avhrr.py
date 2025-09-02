#!/usr/bin/env python3

import os

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

MAPPING_PATH = map_path('bufr_avhrr.yaml')

AM_KEY = 'am'
PM_KEY = 'pm'

class AvhrrObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__({AM_KEY:MAPPING_PATH,
                          PM_KEY:MAPPING_PATH}, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_dict):
        container = bufr.Container()
        if os.path.exists(input_dict[AM_KEY]):
            container.append(bufr.Parser(input_dict[AM_KEY], self.map_dict[AM_KEY]).parse(comm))

        if os.path.exists(input_dict[PM_KEY]):
            container.append(bufr.Parser(input_dict[PM_KEY], self.map_dict[PM_KEY]).parse(comm))

        return container

    def _make_description(self):
        description = bufr.encoders.Description(self.map_dict[AM_KEY])


# Add main functions create_obs_file and create_obs_group
add_main_functions(AvhrrObsBuilder)