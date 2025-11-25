#!/usr/bin/env python3

import os

from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

MAPPING_PATH = map_path('bufr_satwnd_amv_seviri.yaml')


class SatWndAmvSeviriObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))


# Add main functions create_obs_file and create_obs_group
add_main_functions(SatWndAmvSeviriObsBuilder)
