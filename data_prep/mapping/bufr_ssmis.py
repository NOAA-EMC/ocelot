#!/usr/bin/env python3
import os

from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

MAPPING_PATH = map_path('bufr_ssmis.yaml')


class BufrSsmisObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path):
        container = super().make_obs(comm, input_path)

        # Mask out values with non-zero quality flags
        sat_id = container.get('satalliteIdentifier')
        container.apply_mask(sat_id == 285)

        return container


# Add main functions create_obs_file or create_obs_group
add_main_functions(BufrSsmisObsBuilder)
