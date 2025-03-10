#!/usr/bin/env python3

import os
import numpy as np
import re
import datetime

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions

script_dir = os.path.dirname(os.path.abspath(__file__))

ADPUPA_MAPPING = os.path.join(script_dir, 'bufr_adpupa_mapping.yaml')
ADPSFC_SFCSHP_MAPPING = os.path.join(script_dir, 'bufr_adpsfc_sfcshp_mapping.yaml')

OBS_TYPES = np.array([180, 181, 183, 187, 120])


class PressureObsBuilder(ObsBuilder):
    def __init__(self):
        map_dict = {'adpupa': ADPUPA_MAPPING,
                    'adpsfc_sfcshp': ADPSFC_SFCSHP_MAPPING}

        super().__init__(map_dict, log_name=os.path.basename(__file__))

    def _get_reference_time(self, input_path) -> datetime.datetime:
        path_components = os.path.split(input_path)

        m = re.match('\w+\.(?<year>\d{4})(?<month>\d{2})(?<day>\d{2})', path_components[-4])
        return np.datetime64(datetime.datetime(year=int(m.group('year')), 
                                               month=int(m.group('month')), 
                                               day=int(m.group('day')), 
                                               hour=int(path_components[-3])))

    # Overrides
    def make_obs(self, comm, input_path) -> bufr.DataContainer:
        container = bufr.Parser(input_path, self.map_dict['adpsfc_sfcshp']).parse(comm)
        cycle_times = np.array([timedelta(seconds=3600*t) for t in container.get('obsTimeMinusCycleTime')])
        times = _get_reference_time(input_path) + cycle_times

        adpupa_container = bufr.Parser(input_path, self.map_dict['adpupa']).parse(comm)

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
