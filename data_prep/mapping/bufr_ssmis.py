#!/usr/bin/env python3
import os
import numpy as np
import bufr

from bufr.obs_builder import ObsBuilder, add_main_functions, map_path

MAPPING_PATH = map_path('bufr_ssmis.yaml')


class BufrSsmisObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path) -> bufr.DataContainer:
        container = super().make_obs(comm, input_path)

        latitude1 = container.get('latitude1')
        latitude2 = container.get('latitude2')

        isAscending = np.where(latitude2 > latitude1, 1, -1).astype(np.int32)
        container.add('satelliteAscendingFlag', isAscending, ["*"])

        return container

    def _make_description(self):
        description = super()._make_description()

        description.add_variables([
            {
                'name': "satelliteAscendingFlag",
                'source': 'satelliteAscendingFlag',
                'longName': "Satellite Ascending Flag",
                'units': "boolean",
            }
        ])

        return description


# Add main functions create_obs_file or create_obs_group
add_main_functions(BufrSsmisObsBuilder)
