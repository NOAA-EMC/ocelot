#!/usr/bin/env python3
import os
import numpy as np
import numpy.ma as ma

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path
from bufr.obs_builder import nprocs_per_task, add_dummy_variable
from bufr.transforms import compute_solar_angles
from datetime import datetime

MAPPING_PATH = map_path('bufr_ssmis.yaml')


class BufrSsmisObsBuilder(ObsBuilder):
    """
    Class for building observations from ssmis BUFR data.

    This class extends `ObsBuilder` to include specific logic for processing
    SSMIS data such as solar angles and satellite ascending/descending orbits.

    :param mapping_path: Path to the mapping file.
    :type mapping_path: str
    """

    def __init__(self):
        super().__init__(MAPPING_PATH, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_path):

        # Get container from mapping file first
        self.log.info('Get container from bufr')
        container = super().make_obs(comm, input_path)

        self.log.debug(f'container list (original): {container.list()}')
        self.log.debug(f'all_sub_categories =  {container.all_sub_categories()}')
        self.log.debug(f'category map =  {container.get_category_map()}')

        # Add new/derived data into container
        for cat in container.all_sub_categories():

            self.log.debug(f'category = {cat}')

            satId = container.get('satelliteId', cat)
            if not np.any(satId):
                self.log.warning(f'category {cat[0]} does not exist in input file')

            self._add_sensor_zenith_and_solar_angles(container, cat)
            self._add_satellite_ascend_descent_orbit(container, cat)

        # Check
        self.log.debug(f'container list (updated): {container.list()}')
        self.log.debug(f'all_sub_categories {container.all_sub_categories()}')

        return container

    def _make_description(self):
        description = super()._make_description()
        self._add_new_variable_descriptions(description)

        return description

    def _add_new_variable_descriptions(self, description):
        description.add_variables([
            {
                'name': 'MetaData/satelliteAscendingFlag',
                'source': 'satelliteAscendingFlag',
                'longName': 'Satellite Ascending/Descending Orbit Flag (Ascend:1; Descend:-1)',
            },
            {
                'name': 'MetaData/sensorZenithAngle',
                'source': 'sensorZenithAngle',
                'units': 'degree',
                'longName': 'Sensor Zenith Angle',
            },
            {
                'name': 'MetaData/sensorAzimuthAngle',
                'source': 'sensorAzimuthAngle',
                'units': 'degree',
                'longName': 'Sensor Azimuth Angle',
            },
            {
                'name': 'MetaData/solarZenithAngle',
                'source': 'solarZenithAngle',
                'units': 'degree',
                'longName': 'Solar Zenith Angle',
            },
            {
                'name': 'MetaData/solarAzimuthAngle',
                'source': 'solarAzimuthAngle',
                'units': 'degree',
                'longName': 'Solar Azimuth Angle',
            }])

    def _add_satellite_ascend_descent_orbit(self, container, category):
        """
        Determine satellite orbit type (ascending or descending) based on latitude changes.

        :param container: Observation data container.
        :type container: Container
        :param category: Data category to process.
        :type category: str
        """

        satId = container.get('satelliteId', category)

        if not satId.size:
            add_dummy_variable(container, 'satelliteAscendingFlag', category, 'fieldOfViewNumber')
            return

        # Get data from container
        # ephemeris data - latitude values in order of time
        first_lat = container.get('latitude1', category)
        self.log.debug(f'first_lat min/max = {first_lat.min()} {first_lat.max()}')
        second_lat = container.get('latitude2', category)
        self.log.debug(f'second_lat min/max = {second_lat.min()} {second_lat.max()}')
        fovn = container.get('fieldOfViewNumber', category)
        self.log.debug(f'fovn min/max = {fovn.min()} {fovn.max()}')

        # Determine ascending/descending mode
        # Compare latitude between the first and second records
        orbit = np.where(second_lat > first_lat, 1, -1).astype(np.int32)

        self.log.debug(f'orbit min/max = {orbit.min()} {orbit.max()}')

        paths = container.get_paths('fieldOfViewNumber', category)
        self.log.debug(f'paths = {paths}')
        container.add('satelliteAscendingFlag', orbit, paths, category)

    def _add_sensor_zenith_and_solar_angles(self, container, category):
        """
        Compute and add solar zenith and azimuth angles to the observation container.

        :param container: Observation data container.
        :type container: Container
        :param category: Data category to process.
        :type category: str
        """

        satId = container.get('satelliteId', category)
        if not satId.size:
            dummy_mappings = [
                ('solarZenithAngle', 'latitude'),
                ('solarAzimuthAngle', 'latitude'),
                ('sensorZenithAngle', 'latitude'),
                ('sensorAzimuthAngle', 'latitude')
            ]
            for target_var, source_var in dummy_mappings:
                add_dummy_variable(container, target_var, category, source_var)

            return

        # Prepare input arrays
        unix_times = container.get('timestamp', category)
        latitudes = container.get('latitude', category)
        longitudes = container.get('longitude', category)
        self.log.debug(f'latitudes min/max = {latitudes.min()} {latitudes.max()}')
        self.log.debug(f'longitudes min/max = {longitudes.min()} {longitudes.max()}')
        self.log.debug(f'unix_times min/max = {unix_times.min()} {unix_times.max()}')

        # Calculate solar angles
        zenith_angles, azimuth_angles = compute_solar_angles(latitudes, longitudes, unix_times)

        self.log.debug(f'zenith_angles min/max = {zenith_angles.min()} {zenith_angles.max()}')
        self.log.debug(f'azimuth_angles min/max = {azimuth_angles.min()} {azimuth_angles.max()}')

        # Add solar angles
        paths = container.get_paths('latitude', category)
        self.log.debug(f'paths = {paths}')
        container.add('solarZenithAngle', zenith_angles, paths, category)
        container.add('solarAzimuthAngle', azimuth_angles, paths, category)

        # Add sensor angles
        sensor_zenith = np.full_like(latitudes, 53.0)
        sensor_azimuth = np.full_like(latitudes, latitudes.fill_value)
        container.add('sensorZenithAngle', sensor_zenith, paths, category)
        container.add('sensorAzimuthAngle', sensor_azimuth, paths, category)


# Add main functions create_obs_file or create_obs_group
add_main_functions(BufrSsmisObsBuilder)