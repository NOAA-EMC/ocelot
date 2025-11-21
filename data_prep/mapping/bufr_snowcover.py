#!/usr/bin/env python3

import os

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

SNOCVR_KEY = 'snocvr'
SFCSNO_KEY = 'sfcsno'

SFCSNO_MAPPING = map_path('bufr_snowcover_sfcsno.yaml')
SNOCVR_MAPPING = map_path('bufr_snowcover_snocvr.yaml')


class SnowCoverObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__({SNOCVR_KEY: SNOCVR_MAPPING,
                          SFCSNO_KEY: SFCSNO_MAPPING}, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_dict):
        container = bufr.DataContainer()

        if SNOCVR_KEY in input_dict:
            container.append(bufr.Parser(input_dict[SNOCVR_KEY], self.map_dict[SNOCVR_KEY]).parse(comm))

        if SFCSNO_KEY in input_dict:
            container.append(bufr.Parser(input_dict[SFCSNO_KEY], self.map_dict[SFCSNO_KEY]).parse(comm))

        # Mask out rows with no valid snow depth
        snow_depth = container.get('snowDepth')
        container.apply_mask(~snow_depth.mask)

        return container


# Add main functions create_obs_file and create_obs_group
add_main_functions(SnowCoverObsBuilder)
