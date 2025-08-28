#!/usr/bin/env python3

import os

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

MAPPING_PATH = map_path('bufr_snowcover.yaml')

class SnowCoverObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))


# Add main functions create_obs_file and create_obs_group
add_main_functions(SnowCoverObsBuilder)