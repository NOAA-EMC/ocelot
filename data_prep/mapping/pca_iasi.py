#!/usr/bin/env python3

import os

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

MAPPING_PATH = map_path('pca_iasi.yaml')

class IasiPcaObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path):
        container = super().make_obs(comm, input_path)

        # Mask out values with non-zero quality flags
        principleCompScores = container.get('principalComponentScore')

        principleCompScores[0] = principleCompScores[0][0:20]
        principleCompScores[1] = principleCompScores[1][0:10]
        principleCompScores[2] = principleCompScores[2][0:10]

        container.replace('principalComponentScore', principleCompScores)

        return container


# Add main functions create_obs_file and create_obs_group
add_main_functions(IasiPcaObsBuilder)
