#!/usr/bin/env python3
import os
import sys
sys.path.insert(0, os.path.realpath('..'))

import os
import argparse
import time
import numpy as np
import bufr
from wxflow import Logger

from zarr_encoder import Encoder as zarrEncoder

# Initialize Logger
# Get log level from the environment variable, default to 'INFO it not set
log_level = os.getenv('LOG_LEVEL', 'INFO')
logger = Logger('bufr_atms.py', level=log_level, colored_log=False)


def logging(comm, level, message):
    """
  Logs a message to the console or log file, based on the specified logging level.

  This function ensures that logging is only performed by the root process (`rank 0`)
  in a distributed computing environment. The function maps the logging level to
  appropriate logger methods and defaults to the 'INFO' level if an invalid level is provided.

  Parameters:
      comm: object
          The communicator object, typically from a distributed computing framework
          (e.g., MPI). It must have a `rank()` method to determine the process rank.
      level: str
          The logging level as a string. Supported levels are:
              - 'DEBUG'
              - 'INFO'
              - 'WARNING'
              - 'ERROR'
              - 'CRITICAL'
          If an invalid level is provided, a warning will be logged, and the level
          will default to 'INFO'.
      message: str
          The message to be logged.

  Behavior:
      - Logs messages only on the root process (`comm.rank() == 0`).
      - Maps the provided logging level to a method of the logger object.
      - Defaults to 'INFO' and logs a warning if an invalid logging level is given.
      - Supports standard logging levels for granular control over log verbosity.

  Example:
      >>> logging(comm, 'DEBUG', 'This is a debug message.')
      >>> logging(comm, 'ERROR', 'An error occurred!')

  Notes:
      - Ensure that a global `logger` object is configured before using this function.
      - The `comm` object should conform to MPI-like conventions (e.g., `rank()` method).
  """

    if comm.rank() == 0:
        # Define a dictionary to map levels to logger methods
        log_methods = {
            'DEBUG': logger.debug,
            'INFO': logger.info,
            'WARNING': logger.warning,
            'ERROR': logger.error,
            'CRITICAL': logger.critical,
        }

        # Get the appropriate logging method, default to 'INFO'
        log_method = log_methods.get(level.upper(), logger.info)

        if log_method == logger.info and level.upper() not in log_methods:
            # Log a warning if the level is invalid
            logger.warning(f'log level = {level}: not a valid level --> set to INFO')

        # Call the logging method
        log_method(message)


def _make_description(mapping_path, update=False):
    description = bufr.encoders.Description(mapping_path)

    if update:
        # Define the variables to be added in a list of dictionaries
        variables = [
        ]

        # Loop through each variable and add it to the description
        for var in variables:
            description.add_variable(
                name=var['name'],
                source=var['source'],
                units=var['units'],
                longName=var['longName']
            )

    return description


def _make_obs(comm, input_path, mapping_path):
    # Get container from mapping file first
    container = bufr.Parser(input_path, mapping_path).parse(comm)


    return container


def create_obs_file(input_path, mapping_path, output_path):
    comm = bufr.mpi.Comm("world")
    container = _make_obs(comm, input_path, mapping_path)
    container.gather(comm)

    description = _make_description(mapping_path, update=True)

    # Encode the data
    if comm.rank() == 0:
        zarrEncoder(description).encode(container, output_path, append=True)


MAP_PATH = os.path.join(os.path.dirname(__file__), 'bufr_atms_mapping.yaml')

def make_obs(comm, input_path):
    return _make_obs(comm, input_path, MAP_PATH)

def make_encoder_description():
    return _make_description(MAP_PATH)


if __name__ == '__main__':
    start_time = time.time()

    bufr.mpi.App(sys.argv)
    comm = bufr.mpi.Comm("world")

    # Required input arguments as positional arguments
    parser = argparse.ArgumentParser(description="Convert BUFR to NetCDF using a mapping file.")
    parser.add_argument('input', type=str, help='Input BUFR file')
    parser.add_argument('mapping', type=str, help='BUFR2IODA Mapping File')
    parser.add_argument('output', type=str, help='Output NetCDF file')

    args = parser.parse_args()
    infile = args.input
    mapping = args.mapping
    output = args.output

    create_obs_file(infile, mapping, output)

    end_time = time.time()
    running_time = end_time - start_time
    logging(comm, 'INFO', f'Total running time: {running_time}')
