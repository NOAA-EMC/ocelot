#!/usr/bin/env python3

import os

import numpy.ma as ma
import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path


SevCsrKey = 'sevcsr'
SevAsrKey = 'sevasr'

SevCsrMapPath = map_path('bufr_sevcsr.yaml')
SevAsrMapPath = map_path('bufr_sevasr.yaml')


class SeviriBuilder(ObsBuilder):
    def __init__(self):
        super().__init__({SevCsrKey: SevCsrMapPath,
                          SevAsrKey: SevAsrMapPath}, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_dict) -> bufr.DataContainer:
        container = bufr.DataContainer()
        if SevCsrKey in input_dict:
            container.append(bufr.Parser(input_dict[SevCsrKey], self.map_dict[SevCsrKey]).parse(comm))
            bt_all = ma.masked_all_like(container.get('bt_clear'))
            container.add('bt_all', bt_all, container.get_paths('bt_clear'))

        if SevAsrKey in input_dict:
            container.append(bufr.Parser(input_dict[SevAsrKey], self.map_dict[SevAsrKey]).parse(comm))

        return container



# Add main functions create_obs_file and create_obs_group
add_main_functions(SeviriBuilder)
