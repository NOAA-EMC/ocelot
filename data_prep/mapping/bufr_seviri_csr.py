#!/usr/bin/env python3

import os
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


SevCsrMapPath = map_path('bufr_sevcsr.yaml')

class SeviriCsrBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(SevCsrMapPath, log_name=os.path.basename(__file__))


# Add main functions create_obs_file and create_obs_group
add_main_functions(SeviriCsrBuilder)
