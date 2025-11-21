#!/usr/bin/env python3

import os


from bufr.obs_builder import ObsBuilder, add_main_functions, map_path
import settings
MAPPING_PATH = map_path('pca_iasi.yaml')


class IasiPcaObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(MAPPING_PATH, settings.BUFR_TABLE_DIR, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path):
        container = super().make_obs(comm, input_path)

        return container


# Add main functions create_obs_file and create_obs_group
add_main_functions(IasiPcaObsBuilder)
