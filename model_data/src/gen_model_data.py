from zarr_encoder import Encoder

import os
import sys
sys.path.insert(0, os.path.realpath('./'))

import argparse
from datetime import datetime, timedelta

import bufr
from zarr_encoder import Encoder
import data_reader
import settings

def create_data_for_day(start_datetime:datetime, end_datetime:datetime, type:str):
    bufr.mpi.App(sys.argv)
    comm = bufr.mpi.Comm("world")

    start_datetime = date
    end_datetime = date + timedelta(hours=23, minutes=59, seconds=59)

    parameters = emctank.Parameters()
    parameters.start_time = start_datetime
    parameters.stop_time = end_datetime

    container, description = data_reader.run(comm, type, parameters)

    if container is None:
        raise ValueError("No data found")

    if comm.rank() == 0:
        output_path = os.path.join(settings.OUTPUT_PATH, f'{type}.zarr')
        Encoder(description).encode(container, output_path, append=True)
        print(f"Output written to {output_path}")
        sys.stdout.flush()


def create_data(start_date: datetime, end_date: datetime, type:str):
    date = start_date
    day = timedelta(days=1)

    while date <= end_date:
        create_data_for_day(date, type)
        date += day


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('start_date')
    parser.add_argument('end_date')
    parser.add_argument('type')

    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d")

    create_data(start_date, end_date, args.type)
