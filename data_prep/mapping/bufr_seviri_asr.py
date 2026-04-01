#!/usr/bin/env python3

import os
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


SevAsrMapPath = map_path('bufr_sevasr.yaml')

class SeviriAsrBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(SevAsrMapPath, log_name=os.path.basename(__file__))


# Add main functions create_obs_file and create_obs_group
add_main_functions(SeviriAsrBuilder)
