#!/usr/bin/env python3

import os
import numpy as np

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions

ADPUPA_MAPPING = './bufr_adpupa_mapping.yaml'
ADPSFC_MAPPING = './bufr_adpsfc_mapping.yaml'
SFCSHP_MAPPING = './bufr_sfcshp_mapping.yaml'

OBS_TYPES = np.array([180, 181, 183, 187, 120])


class PressureObsBuilder(ObsBuilder):
    def __init__(self, input_path):

        input_dict == {'adpupa': (input_path, ADPUPA_MAPPING),
                       'adpsfc': (input_path, ADPSFC_MAPPING),
                       'sfcshp': (input_path, SFCSHP_MAPPING)}

        super().__init__(input_dict, log_name=os.path.basename(__file__))


    # Virtual Method
    def _make_obs(self, comm) -> bufr.DataContainer:
        adpupa_container = bufr.Parser(*self.input_dict['adpupa']).parse(comm)
        adpsfc_container = bufr.Parser(*self.input_dict['adpsfc']).parse(comm)
        sfcshp_container = bufr.Parser(*self.input_dict['sfcshp']).parse(comm)

        # Mask ADPUPA for station pressure category
        data_level_cat = adpupa_container.get('dataLevelCategory')
        adpupa_container.apply_mask(data_level_cat == 0)

        # Remove extra data from ADPUPA (make all containers the same)
        adpupa_container.remove('dataLevelCategory')

        # Merge containers
        container = bufr.DataContainer()
        container.append(adpupa_container)
        container.append(adpsfc_container)
        container.append(sfcshp_container)

        # Filter according to thw obs type
        obs_type = container.get('observationType')
        container.apply_mask(np.isin(obs_type, OBS_TYPES))

        # Wrap the longitude values
        longitude = container.get('longitude')
        longitude = np.where(longitude > 180, longitude - 360, longitude)
        container.replace('longitude', longitude)

        return container


# Add main functions create_obs_file and create_obs_group
add_main_functions(PressureObsBuilder)
