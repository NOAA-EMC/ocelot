#!/usr/bin/env python3

import os

import bufr
from bufr.obs_builder import ObsBuilder, add_main_functions, map_path
from bufr.encoders import netcdf

MAPPING_PATH = map_path('bufr_avhrr.yaml')

AM_KEY = 'am'
PM_KEY = 'pm'


class AvhrrObsBuilder(ObsBuilder):
    def __init__(self):
        super().__init__({AM_KEY: MAPPING_PATH,
                          PM_KEY: MAPPING_PATH}, log_name=os.path.basename(__file__))

    def make_obs(self, comm, input_dict):
        container = bufr.DataContainer()
        if AM_KEY in input_dict:
            container.append(bufr.Parser(input_dict[AM_KEY], self.map_dict[AM_KEY]).parse(comm))

        if PM_KEY in input_dict:
            container.append(bufr.Parser(input_dict[PM_KEY], self.map_dict[PM_KEY]).parse(comm))

        return container

    def create_obs_file(self, am_input, pm_input, output, type='netcdf', append=False):
        """
        Create an observation file from the input data. Override this method if you want to
        customize the file creation process or if you need a different function signature (ex: you
        need to pass multiple input files). add_main_functions will copy the function signature.

        :param am_input: Input path to the BUFR file.
        :param pm_input: Input path to the BUFR file.
        :param output: Output file name
        :param type: Data type to encode into (optional)
        :param append: Add to the file if it exists or create a new file. (optional)
        """

        comm = bufr.mpi.Comm("world")
        self.log.comm = comm

        container = self.make_obs(comm, {AM_KEY: am_input, PM_KEY: pm_input})
        container.gather(comm)

        # Encode the data
        if comm.rank() == 0:
            self.finalize_container(container)
            netcdf.Encoder(self.description).encode(container, output, append)

        self.log.info(f'Return the encoded data')

    def create_obs_group(self, am_input, pm_input, env):
        """
        Create an observation file from the input data. Override this method if you want to
        customize the file creation process or if you need a different function signature (ex: you
        need to pass multiple input files).

        :param input: Input path to the BUFR file.
        :param env: The IODA environment. Dictionary with keys: start_time, end_time, comm_name
        :return: IODA ObsGroup object.
        """
        from pyioda.ioda.Engines.Bufr import Encoder as iodaEncoder

        comm = bufr.mpi.Comm(env["comm_name"])
        self.log.comm = comm

        container = self.make_obs(comm, {AM_KEY: am_input, PM_KEY: pm_input})
        container.all_gather(comm)

        self.finalize_container(container)

        self.log.info('Encoding')
        data = next(iter(iodaEncoder(self.description).encode(container).values()))

        return data

    def _make_description(self):
        return bufr.encoders.Description(self.map_dict[AM_KEY])


# Add main functions create_obs_file and create_obs_group
add_main_functions(AvhrrObsBuilder)
