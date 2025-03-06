#!/usr/bin/env python3

import os
import numpy as np

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions

ADPUPA_MAPPING = './bufr_adpupa_mapping.yaml'
ADPSFC_SFCSHP_MAPPING = './bufr_adpsfc_sfcshp_mapping.yaml'

OBS_TYPES = np.array([180, 181, 183, 187, 120])


class PressureObsBuilder(ObsBuilder):
    def __init__(self, input_path):
        input_dict = {'adpupa': (input_path, ADPUPA_MAPPING),
                      'adpsfc_sfcshp': (input_path, ADPSFC_SFCSHP_MAPPING)}

        super().__init__(input_dict, log_name=os.path.basename(__file__))


    # Virtual Method
    def _make_obs(self, comm) -> bufr.DataContainer:
        container = bufr.Parser(*self.input_dict['adpsfc_sfcshp']).parse(comm)
        adpupa_container = bufr.Parser(*self.input_dict['adpupa']).parse(comm)

        # Mask ADPUPA for station pressure category
        data_level_cat = adpupa_container.get('dataLevelCategory')
        adpupa_container.apply_mask(data_level_cat == 0)

        # Remove extra data field from ADPUPA (make all containers the same)
        adpupa_container.remove('dataLevelCategory')

        # Merge containers
        container.append(adpupa_container)

        # Filter according to thw obs type
        obs_type = container.get('observationType')
        container.apply_mask(np.isin(obs_type, OBS_TYPES))

        return container


# Add main functions create_obs_file and create_obs_group
add_main_functions(PressureObsBuilder)
