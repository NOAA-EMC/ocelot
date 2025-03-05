#!/usr/bin/env python3

import os
import numpy as np

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions
from docutils.nodes import description
from earthkit.meteo.thermo.array import virtual_temperature
from sympy.physics.units import temperature

ADPUPA_MAPPING = './bufr_adpupa_mapping.yaml'
ADPSFC_MAPPING = './bufr_adpsfc_mapping.yaml'
SFCSHP_MAPPING = './bufr_sfcshp_mapping.yaml'


def mask_container(container, mask):
    new_container = bufr.DataContainer()
    for field_name in container.list():
        new_container.add(field_name, container.get(field_name)[mask])
    return new_container

class PressureObsBuilder(ObsBuilder):
    def __init__(self, input_path):

        input_dict == {'adpups': (input_path, ADPUPA_MAPPING),
                       'adpsfc': (input_path, ADPSFC_MAPPING),
                       'sfcshp': (input_path, SFCSHP_MAPPING)}

        super().__init__(input_dict, log_name=os.path.basename(__file__))

    # Virtual Method
    def _make_description(self) -> bufr.encoders.Description:
        description = super()._make_description()

        description.add_variables([
            {
                'name': 'virtualTemperature',
                'source': 'virtualTemperature',
                'units': 'K',
                'longName': 'Virtual Temperature'
            }
        ])

        return description


    # Virtual Method
    def _make_obs(self, comm) -> bufr.DataContainer:
        container = super()._make_obs(comm)

        # Wrap the longitude values
        longitude = container.get('longitude')
        longitude = np.where(longitude > 180, longitude - 360, longitude)
        container.replace('longitude', longitude)

        # Filter by type
        obsType = container.get('observationType')
        container = mask_container(container, obsType < 200)

        # Add the virtual temperature
        # temperature = container.get('temperature')
        # virtual_temperature = []
        # container.add('virtualTemperature', virtual_temperature)

        return container


# Add main functions create_obs_file and create_obs_group
add_main_functions(PressureObsBuilder)
